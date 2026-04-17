#!/usr/bin/env python3
"""LoCoMo benchmark runner for taosmd.

Ingests each LoCoMo conversation into a fresh VectorMemory, runs retrieval
against every QA, generates answers via Ollama, and scores with F1, BLEU-1,
and an LLM judge. Output mirrors the Mem0 paper convention.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd.vector_memory import VectorMemory  # noqa: E402
from taosmd.retrieval import retrieve  # noqa: E402

CATEGORY_NAMES = {
    1: "Single-hop (1)",
    2: "Temporal  (2)",
    3: "Multi-hop (3)",
    4: "Open-dom  (4)",
    5: "Adversarial(5)",
}

ANSWER_PROMPT = """You are answering a question using retrieved conversation memory.
Use ONLY the context below. If the answer is not present, reply exactly: I don't know.
Keep your answer to 1-2 short sentences.

Context:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are grading a predicted answer against a reference answer.
Reply with a single token: YES if the predicted answer matches the reference
(same facts, minor wording differences are fine), otherwise NO.

Question: {question}
Reference: {reference}
Predicted: {predicted}

Reply YES or NO only."""

_PUNCT = re.compile(r"[^\w\s]")


def _git_sha() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _tokenize(s: str) -> list[str]:
    return [t for t in _PUNCT.sub(" ", s.lower().strip()).split() if t]


def _f1(pred: str, ref: str) -> float:
    p, r = _tokenize(pred), _tokenize(ref)
    if not p or not r:
        return 0.0
    r_counts: dict[str, int] = {}
    for tok in r:
        r_counts[tok] = r_counts.get(tok, 0) + 1
    overlap = 0
    for tok in p:
        if r_counts.get(tok, 0) > 0:
            overlap += 1
            r_counts[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def _bleu1(pred: str, ref: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        smoothie = SmoothingFunction().method1
        return float(sentence_bleu(
            [_tokenize(ref)], _tokenize(pred),
            weights=(1, 0, 0, 0), smoothing_function=smoothie,
        ))
    except Exception:
        p, r = _tokenize(pred), _tokenize(ref)
        if not p or not r:
            return 0.0
        r_counts: dict[str, int] = {}
        for tok in r:
            r_counts[tok] = r_counts.get(tok, 0) + 1
        matches = 0
        for tok in p:
            if r_counts.get(tok, 0) > 0:
                matches += 1
                r_counts[tok] -= 1
        precision = matches / len(p)
        bp = 1.0 if len(p) >= len(r) else pow(2.718281828, 1 - len(r) / max(len(p), 1))
        return precision * bp


async def _ollama_generate(client: httpx.AsyncClient, url: str, model: str,
                           prompt: str, temperature: float = 0.2) -> str:
    resp = await client.post(
        f"{url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature}},
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


async def _judge(client: httpx.AsyncClient, url: str, model: str,
                 question: str, reference: str, predicted: str) -> float:
    try:
        reply = await _ollama_generate(
            client, url, model,
            JUDGE_PROMPT.format(question=question, reference=reference, predicted=predicted),
            temperature=0.0,
        )
    except Exception:
        return 0.0
    head = reply.strip().upper().split()
    return 1.0 if head and head[0].startswith("YES") else 0.0


def _session_keys(conversation: dict) -> list[tuple[str, str]]:
    pairs = []
    for key in conversation.keys():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        parts = key.split("_")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        if not isinstance(conversation.get(key), list):
            continue
        pairs.append((key, conversation.get(f"{key}_date_time", "")))
    pairs.sort(key=lambda kv: int(kv[0].split("_")[1]))
    return pairs


async def _ingest_conversation(vmem: VectorMemory, conv: dict) -> tuple[int, float]:
    conversation = conv.get("conversation", conv)
    t0 = time.time()
    added = 0
    for session_key, dt in _session_keys(conversation):
        for turn in conversation.get(session_key) or []:
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "")
            await vmem.add(
                f"[{speaker}] {text}",
                metadata={
                    "dia_id": turn.get("dia_id", ""),
                    "session": session_key,
                    "datetime": dt,
                    "speaker": speaker,
                },
            )
            added += 1
    return added, time.time() - t0


async def _retrieve(strategy: str, query: str, vmem: VectorMemory, top_k: int) -> list[dict]:
    if strategy == "vector-only":
        raw = await vmem.search(query, limit=top_k)
        return [{"text": r["text"], "metadata": r.get("metadata", {}),
                 "score": r.get("similarity", 0.0)} for r in raw]
    # TODO: wire cross-encoder reranker for strategy="full"; for now reuse retrieve() with vector source only.
    hits = await retrieve(query, strategy="thorough", sources={"vector": vmem},
                          limit=top_k, agent_name="locomo_eval")
    return [{"text": h.get("text", ""),
             "metadata": h.get("metadata", {}).get("metadata", h.get("metadata", {})),
             "score": h.get("rrf_score", h.get("source_score", 0.0))} for h in hits]


def _build_context(hits: list[dict]) -> str:
    lines = []
    for hit in hits:
        meta = hit.get("metadata", {}) or {}
        dt = meta.get("datetime", "")
        prefix = f"[{dt}] " if dt else ""
        lines.append(f"{prefix}{hit.get('text', '')}")
    return "\n".join(lines)


def _evidence_hits(hits: list[dict], evidence: list[str]) -> int:
    if not evidence:
        return 0
    retrieved = {(h.get("metadata", {}) or {}).get("dia_id") for h in hits}
    retrieved.discard(None)
    return sum(1 for e in evidence if e in retrieved)


async def _process_qa(qa: dict, conv_id: str, vmem: VectorMemory, client: httpx.AsyncClient,
                      ollama_url: str, model: str, top_k: int, strategy: str) -> dict | None:
    if "answer" not in qa:
        return None
    question = qa["question"]
    reference = str(qa["answer"])
    category = int(qa.get("category", 0))
    evidence = qa.get("evidence", []) or []

    t0 = time.time()
    hits = await _retrieve(strategy, question, vmem, top_k)
    retrieval_ms = (time.time() - t0) * 1000.0

    context = _build_context(hits)
    t1 = time.time()
    try:
        predicted = await _ollama_generate(
            client, ollama_url, model,
            ANSWER_PROMPT.format(context=context, question=question),
        )
    except Exception as exc:
        predicted = f"[generation_error: {exc}]"
    gen_ms = (time.time() - t1) * 1000.0

    judge = await _judge(client, ollama_url, model, question, reference, predicted)

    return {
        "conversation_id": conv_id,
        "question": question,
        "reference": reference,
        "predicted": predicted,
        "category": category,
        "f1": round(_f1(predicted, reference), 4),
        "bleu1": round(_bleu1(predicted, reference), 4),
        "judge": round(judge, 4),
        "retrieval_ms": round(retrieval_ms, 2),
        "gen_ms": round(gen_ms, 2),
        "evidence_hits": _evidence_hits(hits, evidence),
        "evidence_total": len(evidence),
    }


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "f1": 0.0, "bleu1": 0.0, "judge": 0.0, "retrieval_recall": 0.0}
    n = len(rows)
    hit_rows = [r for r in rows if r.get("evidence_total", 0) > 0]
    recall = (sum(1 for r in hit_rows if r["evidence_hits"] > 0) / len(hit_rows)) if hit_rows else 0.0
    return {
        "count": n,
        "f1": round(sum(r["f1"] for r in rows) / n, 4),
        "bleu1": round(sum(r["bleu1"] for r in rows) / n, 4),
        "judge": round(sum(r["judge"] for r in rows) / n, 4),
        "retrieval_recall": round(recall, 4),
    }


def _aggregate(results: list[dict]) -> tuple[dict, dict]:
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(str(r["category"]), []).append(r)
    return {cat: _summary(rows) for cat, rows in by_cat.items()}, _summary(results)


def _print_summary(meta: dict, by_category: dict, overall: dict) -> None:
    sep, dash = "=" * 57, "-" * 57
    print(sep)
    print(f"LoCoMo Benchmark \u2014 taosmd (commit {meta.get('git_sha', 'unknown')})")
    print(sep)
    print(f"Conversations: {meta['conversations']} | Questions: {meta['total_qa']} | "
          f"Model: {meta['model']} | Strategy: {meta['strategy']} | Top-K: {meta['top_k']}")
    print(dash)
    print(f"{'Category':<16} {'Count':>5} {'F1':>7} {'BLEU-1':>8} {'Judge':>8} {'R@K':>8}")
    print(dash)
    for cat in sorted(by_category.keys(), key=int):
        s = by_category[cat]
        label = CATEGORY_NAMES.get(int(cat), f"Cat {cat}")
        recall = "-" if int(cat) == 5 else f"{s['retrieval_recall']:.2f}"
        print(f"{label:<16} {s['count']:>5} {s['f1']:>7.2f} {s['bleu1']:>8.2f} "
              f"{s['judge']:>8.2f} {recall:>8}")
    print(dash)
    print(f"{'Overall':<16} {overall['count']:>5} {overall['f1']:>7.2f} "
          f"{overall['bleu1']:>8.2f} {overall['judge']:>8.2f} "
          f"{overall['retrieval_recall']:>8.2f}")
    print(sep)


async def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    conversations = json.loads(dataset_path.read_text())[: args.conversations]

    if args.category == "all":
        include_cats = {1, 2, 3, 4}
        if args.include_adversarial:
            include_cats.add(5)
    else:
        include_cats = {int(args.category)}

    results: list[dict] = []
    total_seen = 0

    async with httpx.AsyncClient(timeout=60) as client:
        for conv in conversations:
            conv_id = conv.get("sample_id", "unknown")
            tmp = tempfile.mkdtemp(prefix=f"locomo_{conv_id}_")
            vmem = VectorMemory(
                db_path=os.path.join(tmp, "vmem.db"),
                qmd_url=args.qmd_url,
                embed_mode=args.embed_mode,
                onnx_path=args.onnx_path,
            )
            await vmem.init(http_client=client)
            added, ingest_s = await _ingest_conversation(vmem, conv)
            print(f"[{conv_id}] ingested {added} turns in {ingest_s:.1f}s", flush=True)

            for qa in conv.get("qa", []) or []:
                cat = int(qa.get("category", 0))
                if cat not in include_cats:
                    continue
                if args.limit and total_seen >= args.limit:
                    break
                record = await _process_qa(qa, conv_id, vmem, client, args.ollama_url,
                                           args.model, args.top_k, args.strategy)
                if record is None:
                    continue
                results.append(record)
                total_seen += 1
                if total_seen % 25 == 0:
                    print(f"  progress: {total_seen} QAs processed", flush=True)
            await vmem.close()
            if args.limit and total_seen >= args.limit:
                break

    by_category, overall = _aggregate(results)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    meta = {
        "run_id": args.run_id or "",
        "timestamp": timestamp,
        "model": args.model,
        "top_k": args.top_k,
        "strategy": args.strategy,
        "total_qa": len(results),
        "categories_included": sorted(include_cats),
        "conversations": min(args.conversations, len(conversations)),
        "git_sha": _git_sha(),
        "dataset": str(dataset_path),
    }

    if args.out:
        out_path = Path(args.out)
    else:
        tag = f"_{args.run_id}" if args.run_id else ""
        out_path = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / f"locomo_{timestamp}{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"meta": meta, "by_category": by_category, "overall": overall, "results": results},
        indent=2,
    ))

    _print_summary(meta, by_category, overall)
    print(f"\nwrote {out_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo benchmark runner for taosmd")
    p.add_argument("--dataset", default="/home/jay/taosmd/data/locomo/data/locomo10.json")
    p.add_argument("--limit", type=int, default=0, help="Cap total QAs (0=all)")
    p.add_argument("--conversations", type=int, default=10)
    p.add_argument("--category", default="all")
    p.add_argument("--include-adversarial", action="store_true")
    p.add_argument("--qmd-url", default="http://localhost:7832")
    p.add_argument("--embed-mode", choices=["qmd", "local", "onnx"], default="onnx",
                   help="Embedding backend. Default onnx (Fedora reference stack).")
    p.add_argument("--onnx-path", default="/home/jay/taosmd/models/minilm-onnx",
                   help="Path to MiniLM ONNX model directory.")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--model", default="qwen3:4b")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--strategy", choices=["vector-only", "full"], default="vector-only")
    p.add_argument("--out", default=None)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))
