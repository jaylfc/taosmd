#!/usr/bin/env python3
"""MemPalace LoCoMo adapter.

Same generator, prompt, judge, and dataset as benchmarks/locomo_runner.py and
benchmarks/mem0_locomo_runner.py — only the retrieval layer changes. Requires
`pip install mempalace`.

Per-conversation palace strategy: each LoCoMo conversation gets its own palace
directory at /tmp/mempalace_{run_id}_{conv_id}/ so search() naturally scopes to
that conversation without wing/room filters. This avoids the need for a shared
palace with wing= scoping and makes it trivial to delete per-conv state
afterwards if desired.

Ingest strategy: if a Python ingest API is available (MemoryStack or similar in
mempalace.layers), we use it. If no clean Python ingest API is found, we write
each conversation's turns to a temporary text file and invoke `mempalace mine
<path>` via subprocess.
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

# ---------------------------------------------------------------------------
# Shared constants (duplicated from locomo_runner.py / mem0_locomo_runner.py to
# keep this file self-contained; all three runners must stay in sync if prompts
# change).
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    1: "Single-hop (1)",
    2: "Temporal  (2)",
    3: "Multi-hop (3)",
    4: "Open-dom  (4)",
    5: "Adversarial(5)",
}

ANSWER_PROMPT = """You are answering a question using retrieved conversation memory.
Use the context below. For date/time questions, always answer with explicit absolute dates (e.g., "7 May 2023" or "June 2023") — never relative references like "last Saturday", "this month", or "next month". Copy the exact date format used in the conversation.
If the answer is not directly stated but can be reasonably inferred from the context, give your best inference. Only reply "I don't know" if the context is unrelated to the question.
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

# MemPalace doesn't round-trip LoCoMo dia_id metadata, so retrieval recall
# cannot be computed. Both fields are reported as None throughout.
_EVIDENCE_UNAVAILABLE: dict = {"evidence_hits": None, "evidence_total": None}


# ---------------------------------------------------------------------------
# Utilities (identical to locomo_runner.py and mem0_locomo_runner.py)
# ---------------------------------------------------------------------------

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
                 question: str, reference: str, predicted: str) -> float | None:
    """Returns 1.0/0.0 on success, None on transport failure."""
    try:
        reply = await _ollama_generate(
            client, url, model,
            JUDGE_PROMPT.format(question=question, reference=reference, predicted=predicted),
            temperature=0.0,
        )
    except Exception:
        return None
    head = reply.strip().upper().split()
    return 1.0 if head and head[0].startswith("YES") else 0.0


# ---------------------------------------------------------------------------
# LoCoMo data helpers (verbatim from locomo_runner.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scoring / aggregate (same as mem0_locomo_runner.py, None-aware per PR #33)
# ---------------------------------------------------------------------------

def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "f1": 0.0, "bleu1": 0.0, "judge": 0.0, "retrieval_recall": 0.0}
    n = len(rows)
    # Filter None values so infra failures don't depress averages.
    f1_vals = [r["f1"] for r in rows if r.get("f1") is not None]
    bleu_vals = [r["bleu1"] for r in rows if r.get("bleu1") is not None]
    judge_vals = [r["judge"] for r in rows if r.get("judge") is not None]
    # MemPalace doesn't carry dia_id so retrieval_recall is always 0.0 here.
    hit_rows = [r for r in rows if r.get("evidence_total") is not None and r["evidence_total"] > 0]
    recall = (
        sum(1 for r in hit_rows if r["evidence_hits"] is not None and r["evidence_hits"] > 0)
        / len(hit_rows)
    ) if hit_rows else 0.0
    return {
        "count": n,
        "f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else 0.0,
        "bleu1": round(sum(bleu_vals) / len(bleu_vals), 4) if bleu_vals else 0.0,
        "judge": round(sum(judge_vals) / len(judge_vals), 4) if judge_vals else 0.0,
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
    print(f"LoCoMo Benchmark \u2014 MemPalace (commit {meta.get('git_sha', 'unknown')})")
    print(sep)
    print(f"Conversations: {meta['conversations']} | Questions: {meta['total_qa']} | "
          f"Model: {meta['model']} | Top-K: {meta['top_k']}")
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


# ---------------------------------------------------------------------------
# MemPalace ingest
# ---------------------------------------------------------------------------

def _palace_path(run_id: str, conv_id: str) -> str:
    """Each conversation gets its own palace under /tmp/."""
    safe_run = run_id.replace("/", "_") if run_id else "bench"
    safe_conv = str(conv_id).replace("/", "_")
    return f"/tmp/mempalace_{safe_run}_{safe_conv}"


def _build_turn_text(dt: str, speaker: str, text: str) -> str:
    prefix = f"[{dt}] " if dt else ""
    return f"{prefix}[{speaker}] {text}"


def _ingest_conversation_mempalace(conv: dict, conv_id: str, palace: str) -> tuple[int, float]:
    """Write conversation turns to a temp file and ingest via `mempalace mine`.

    We first attempt programmatic ingest via mempalace.layers.MemoryStack if it
    exists. If that class is absent or raises, we fall back to the CLI path.

    Returns (turn_count, elapsed_seconds).
    """
    conversation = conv.get("conversation", conv)
    lines: list[str] = []
    for session_key, dt in _session_keys(conversation):
        for turn in conversation.get(session_key) or []:
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "")
            lines.append(_build_turn_text(dt, speaker, text))

    if not lines:
        return 0, 0.0

    t0 = time.time()

    # --- Attempt 1: programmatic API via mempalace.layers.MemoryStack ----------
    try:
        from mempalace.layers import MemoryStack  # type: ignore
        stack = MemoryStack(palace_path=palace)
        for line in lines:
            stack.add(line)
        elapsed = time.time() - t0
        return len(lines), elapsed
    except (ImportError, AttributeError):
        pass  # Fall through to CLI path.
    except Exception:
        pass  # Also fall through — CLI is the canonical ingest path.

    # --- Attempt 2: CLI `mempalace mine <path>` --------------------------------
    # Write turns to a temp .txt file inside the palace directory so the crawler
    # picks them up.
    Path(palace).mkdir(parents=True, exist_ok=True)
    stage_file = os.path.join(palace, "_locomo_stage.txt")
    with open(stage_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    result = subprocess.run(
        ["mempalace", "mine", palace],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"mempalace mine failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return len(lines), elapsed


def _mempalace_search(palace: str, question: str, top_k: int) -> list[str]:
    """Synchronous wrapper around search_memories(); returns list of text strings."""
    from mempalace.searcher import search_memories  # type: ignore
    out = search_memories(
        query=question,
        palace_path=palace,
        n_results=top_k,
    )
    results = out.get("results", []) if isinstance(out, dict) else out
    return [r["text"] for r in results if r.get("text")]


# ---------------------------------------------------------------------------
# Per-QA processing
# ---------------------------------------------------------------------------

async def _process_qa(
    qa: dict,
    conv_id: str,
    palace: str,
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    top_k: int,
    loop: asyncio.AbstractEventLoop,
) -> dict | None:
    if "answer" not in qa:
        return None
    question = qa["question"]
    reference = str(qa["answer"])
    category = int(qa.get("category", 0))

    t0 = time.time()
    try:
        context_chunks = await loop.run_in_executor(
            None, _mempalace_search, palace, question, top_k
        )
    except Exception as exc:
        return {
            "conversation_id": conv_id,
            "question": question,
            "reference": reference,
            "predicted": "",
            "category": category,
            "f1": None,
            "bleu1": None,
            "judge": None,
            "retrieval_ms": 0.0,
            "gen_ms": 0.0,
            **_EVIDENCE_UNAVAILABLE,
            "error": f"{type(exc).__name__}: {exc}",
        }
    retrieval_ms = (time.time() - t0) * 1000.0

    context = "\n---\n".join(context_chunks) if context_chunks else ""

    t1 = time.time()
    generation_error: str | None = None
    try:
        predicted = await _ollama_generate(
            client, ollama_url, model,
            ANSWER_PROMPT.format(context=context, question=question),
        )
    except Exception as exc:
        predicted = ""
        generation_error = f"{type(exc).__name__}: {exc}"
    gen_ms = (time.time() - t1) * 1000.0

    if generation_error is None:
        judge = await _judge(client, ollama_url, model, question, reference, predicted)
        f1_val = round(_f1(predicted, reference), 4)
        bleu_val = round(_bleu1(predicted, reference), 4)
        judge_val = round(judge, 4) if judge is not None else None
    else:
        f1_val = None
        bleu_val = None
        judge_val = None

    row: dict = {
        "conversation_id": conv_id,
        "question": question,
        "reference": reference,
        "predicted": predicted,
        "category": category,
        "f1": f1_val,
        "bleu1": bleu_val,
        "judge": judge_val,
        "retrieval_ms": round(retrieval_ms, 2),
        "gen_ms": round(gen_ms, 2),
        **_EVIDENCE_UNAVAILABLE,
    }
    if generation_error is not None:
        row["error"] = generation_error
    return row


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    try:
        from mempalace.searcher import search_memories  # noqa: F401
    except ImportError:
        print(
            "ERROR: mempalace is not installed. Run: pip install mempalace",
            file=sys.stderr,
        )
        return 1

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

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    results: list[dict] = []
    total_seen = 0
    failed_qa = 0
    concurrency = max(1, int(args.concurrency))
    sem = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    loop = asyncio.get_event_loop()

    async def _guarded(qa: dict, conv_id: str, palace: str) -> dict | None:
        nonlocal total_seen, failed_qa
        async with sem:
            try:
                outcome = await _process_qa(
                    qa, conv_id, palace, client, args.ollama_url,
                    args.model, args.top_k, loop,
                )
            except Exception as e:
                async with progress_lock:
                    failed_qa += 1
                    print(f"  qa failed: {e!r}", flush=True)
                return None
        if outcome is not None:
            async with progress_lock:
                total_seen += 1
                if total_seen % 25 == 0:
                    print(f"  progress: {total_seen} QAs processed", flush=True)
        return outcome

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for conv in conversations:
            conv_id = conv.get("sample_id", "unknown")
            palace = _palace_path(run_id, conv_id)

            # Ingest phase: synchronous; offload to thread pool.
            try:
                added, ingest_s = await loop.run_in_executor(
                    None, _ingest_conversation_mempalace, conv, conv_id, palace
                )
            except Exception as exc:
                print(f"[{conv_id}] ingest failed: {exc!r} — skipping conversation", flush=True)
                continue
            print(
                f"[{conv_id}] ingested {added} turns to mempalace in {ingest_s:.1f}s",
                flush=True,
            )

            eligible: list[dict] = []
            for qa in conv.get("qa", []) or []:
                cat = int(qa.get("category", 0))
                if cat not in include_cats:
                    continue
                if args.per_conv_limit and len(eligible) >= args.per_conv_limit:
                    break
                if args.limit and (total_seen + len(eligible)) >= args.limit:
                    break
                eligible.append(qa)

            if eligible:
                tasks = [_guarded(qa, conv_id, palace) for qa in eligible]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                for outcome in gathered:
                    if isinstance(outcome, Exception) or outcome is None:
                        continue
                    results.append(outcome)

            if args.limit and total_seen >= args.limit:
                break

    by_category, overall = _aggregate(results)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    meta = {
        "run_id": args.run_id or "",
        "timestamp": timestamp,
        "model": args.model,
        "memory_backend": "mempalace",
        "top_k": args.top_k,
        "total_qa": len(results),
        "categories_included": sorted(include_cats),
        "conversations": min(args.conversations, len(conversations)),
        "git_sha": _git_sha(),
        "dataset": str(dataset_path),
        "concurrency": concurrency,
        "failed_qa": failed_qa,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        tag = f"_{args.run_id}" if args.run_id else ""
        out_path = (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "results"
            / f"locomo_{timestamp}{tag}_full_mempalace_e2b.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"meta": meta, "by_category": by_category, "overall": overall, "results": results},
        indent=2,
    ))

    _print_summary(meta, by_category, overall)
    print(f"\nwrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATASET = os.environ.get(
    "LOCOMO_DATASET",
    os.path.join(_REPO_ROOT, "data", "locomo", "data", "locomo10.json"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoCoMo benchmark runner using MemPalace as the memory layer"
    )
    p.add_argument("--dataset", default=_DEFAULT_DATASET,
                   help=f"LoCoMo dataset JSON. Env: LOCOMO_DATASET (default: {_DEFAULT_DATASET})")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap total QAs across all conversations (0=all)")
    p.add_argument("--per-conv-limit", type=int, default=0,
                   help="Cap QAs per conversation for balanced sampling (0=all eligible)")
    p.add_argument("--conversations", type=int, default=10)
    p.add_argument("--category", default="all")
    p.add_argument("--include-adversarial", action="store_true")
    p.add_argument(
        "--ollama-url",
        default=os.environ.get("TAOSMD_OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL. Env: TAOSMD_OLLAMA_URL (default http://localhost:11434)",
    )
    p.add_argument("--model", default="gemma4:e2b")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--concurrency", type=int, default=3,
        help=("Max parallel QAs per conversation. Requires OLLAMA_NUM_PARALLEL "
              ">= N on the Ollama server. Default 3; use 1 for sequential."),
    )
    p.add_argument(
        "--timeout", type=float, default=120.0,
        help="HTTP timeout for Ollama calls in seconds. Default 120.",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))
