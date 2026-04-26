#!/usr/bin/env python3
"""LoCoMo benchmark runner for taosmd.

Ingests each LoCoMo conversation into a fresh VectorMemory, runs retrieval
against every QA, generates answers via Ollama, and scores with F1, BLEU-1,
and an LLM judge. Output mirrors the Mem0 paper convention.

Per-conversation QAs run concurrently (see ``--concurrency``) to cut wall
time; conversations themselves remain sequential so vmem instances do not
fight over the tempdir / embedding model load. Concurrency requires the
Ollama server to be configured with ``OLLAMA_NUM_PARALLEL`` at least as
large as ``--concurrency`` or requests will serialise on the server side.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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

MULTIHOP_PROMPT = """Split this question into 2 or 3 shorter, focused sub-queries that together cover everything needed to answer it. Respond with one sub-query per line. No numbering, no explanation.

Question: {question}"""

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
                           prompt: str, temperature: float = 0.2,
                           thinking_mode: bool = False) -> str:
    # Default: think=false disables reasoning mode on Qwen3/3.5/3.6 generators
    # (10-20x faster; hidden reasoning adds no visible content but bills full
    # generation time). Set thinking_mode=True (via --thinking-mode on the CLI)
    # to measure whether reasoning-on improves answer quality. Slow — expect
    # ~150s per call vs ~5s with thinking off.
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature}}
    if not thinking_mode:
        payload["think"] = False
    resp = await client.post(f"{url}/api/generate", json=payload)
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


async def _ingest_conversation(
    vmem: VectorMemory, conv: dict, multi_level: bool = False,
) -> tuple[int, float, dict[str, dict]]:
    """Ingest conversation turns into vmem.

    Returns (added_count, elapsed_s, turn_index) where turn_index maps a
    sequential turn index (across all sessions) to its metadata dict, for use
    with --adjacent-turns.

    When multi_level=True, also ingests two additional retrieval levels:
      - session_summary entries (Episode-level, ~300-char narrative per session)
      - event_summary entries (Fact-level, structured per-speaker events)
    Both come pre-computed from the LoCoMo dataset — no LLM cost. The vmem's
    RRF fusion at retrieval naturally surfaces whichever level is most
    relevant for a given question. Equivalent to HyperMem's coarse-to-fine
    Topic/Episode/Fact hierarchy without the LLM-driven construction.
    """
    conversation = conv.get("conversation", conv)
    t0 = time.time()
    added = 0
    turn_index: dict[str, dict] = {}  # str(global_idx) -> {datetime, text, speaker, dia_id}
    global_idx = 0
    for session_key, dt in _session_keys(conversation):
        for turn in conversation.get(session_key) or []:
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "")
            dia_id = turn.get("dia_id", "")
            await vmem.add(
                f"[{speaker}] {text}",
                metadata={
                    "dia_id": dia_id,
                    "session": session_key,
                    "datetime": dt,
                    "speaker": speaker,
                    "turn_idx": global_idx,
                    "level": "turn",
                },
            )
            turn_index[str(global_idx)] = {
                "datetime": dt,
                "text": f"[{speaker}] {text}",
                "speaker": speaker,
                "dia_id": dia_id,
            }
            global_idx += 1
            added += 1

    if multi_level:
        # Episode level — session_summary entries.
        sess_summaries = conv.get("session_summary") or {}
        for key, summary_text in sess_summaries.items():
            if not summary_text:
                continue
            session_id = key.replace("_summary", "")  # e.g. session_1
            dt = conversation.get(f"{session_id}_date_time", "")
            await vmem.add(
                f"[session-summary {session_id}] {summary_text}",
                metadata={
                    "session": session_id,
                    "datetime": dt,
                    "level": "session_summary",
                },
            )
            added += 1

        # Fact level — event_summary entries (per-speaker events).
        event_summaries = conv.get("event_summary") or {}
        for key, events_dict in event_summaries.items():
            if not isinstance(events_dict, dict):
                continue
            session_id = key.replace("events_", "")  # events_session_1 -> session_1
            event_date = events_dict.get("date", "")
            for speaker, events in events_dict.items():
                if speaker == "date" or not events:
                    continue
                if isinstance(events, str):
                    events = [events]
                for event_text in events:
                    if not event_text:
                        continue
                    await vmem.add(
                        f"[event {session_id}] {speaker}: {event_text}",
                        metadata={
                            "session": session_id,
                            "datetime": event_date,
                            "speaker": speaker,
                            "level": "event",
                        },
                    )
                    added += 1

    return added, time.time() - t0, turn_index


def _build_adjacent_map(
    hits: list[dict], turn_index: dict[str, dict], adjacent_turns: int
) -> dict[str, list[str]]:
    """For each hit, collect text of ±adjacent_turns neighbours.

    Returns a dict mapping turn_idx (as str) to list of neighbour text strings,
    deduplicated across hits.
    """
    if adjacent_turns <= 0 or not turn_index:
        return {}

    max_idx = max(int(k) for k in turn_index)
    seen_indices: set[int] = set()

    # Collect the primary turn indices from hits first so we can exclude them
    # from the neighbour lists (they'll appear in the main context).
    primary_indices: set[int] = set()
    for hit in hits:
        meta = hit.get("metadata", {}) or {}
        idx = meta.get("turn_idx")
        if idx is not None:
            primary_indices.add(int(idx))

    adj_map: dict[str, list[str]] = {}
    for hit in hits:
        meta = hit.get("metadata", {}) or {}
        idx = meta.get("turn_idx")
        if idx is None:
            continue
        idx = int(idx)
        neighbours = []
        for offset in range(-adjacent_turns, adjacent_turns + 1):
            if offset == 0:
                continue
            ni = idx + offset
            if ni < 0 or ni > max_idx:
                continue
            if ni in seen_indices or ni in primary_indices:
                continue
            if str(ni) not in turn_index:
                continue
            neighbours.append(turn_index[str(ni)]["text"])
            seen_indices.add(ni)
        if neighbours:
            adj_map[str(idx)] = neighbours

    return adj_map


def _build_context(
    hits: list[dict],
    context_format: str = "plain",
    adjacent_turns_map: dict[str, list[str]] | None = None,
) -> str:
    lines = []
    for hit in hits:
        meta = hit.get("metadata", {}) or {}
        dt = meta.get("datetime", "")
        turn_idx = str(meta.get("turn_idx", ""))

        if context_format == "session_date":
            prefix = f"[Session date: {dt}] " if dt else ""
        elif context_format == "both":
            prefix = f"[Session date: {dt}] [{dt}] " if dt else ""
        else:  # plain
            prefix = f"[{dt}] " if dt else ""

        lines.append(f"{prefix}{hit.get('text', '')}")

        # Inject adjacent turns immediately after each hit
        if adjacent_turns_map and turn_idx in adjacent_turns_map:
            for neighbour_text in adjacent_turns_map[turn_idx]:
                lines.append(neighbour_text)

    return "\n".join(lines)


def _evidence_hits(hits: list[dict], evidence: list[str]) -> int:
    if not evidence:
        return 0
    retrieved = {(h.get("metadata", {}) or {}).get("dia_id") for h in hits}
    retrieved.discard(None)
    return sum(1 for e in evidence if e in retrieved)


def _load_reranker(reranker_choice: str) -> object | None:
    """Return a reranker instance or None, based on --reranker choice."""
    if reranker_choice == "off":
        return None

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if reranker_choice == "ms-marco":
        from taosmd.cross_encoder import CrossEncoderReranker
        onnx_path = os.path.join(_REPO_ROOT, "models", "cross-encoder-onnx")
        return CrossEncoderReranker(onnx_path=onnx_path)

    if reranker_choice == "bge-v2-m3":
        # CrossEncoderReranker is generic — it loads any HF cross-encoder via
        # ONNX + AutoTokenizer from the model dir. BGE-v2-m3 is a 568M XLM-R
        # cross-encoder; same input/output shape as ms-marco-MiniLM, just
        # multilingual and stronger. Drop-in.
        from taosmd.cross_encoder import CrossEncoderReranker
        bge_path = os.path.join(_REPO_ROOT, "models", "bge-reranker-v2-m3-onnx")
        if not Path(bge_path).exists():
            raise FileNotFoundError(
                f"BGE reranker model not found at {bge_path}.\n"
                "Pull it with:\n"
                "  hf download BAAI/bge-reranker-v2-m3 "
                f"--local-dir {bge_path}"
            )
        return CrossEncoderReranker(onnx_path=bge_path)

    raise ValueError(f"Unknown reranker choice: {reranker_choice!r}")


def _apply_temporal_recency_boost(
    hits: list[dict], turn_index: dict[str, dict], decay_lambda: float = 0.02
) -> list[dict]:
    """Multiply each hit's score by an exponential recency factor.

    The boost biases retrieval toward later turns within a conversation, on the
    hypothesis that LoCoMo questions disproportionately reference the most
    recent state of an entity (e.g. "what does Alice think about X *now*").

    decay_lambda controls how aggressive the bias is. With turn indices
    spanning ~600 turns and lambda=0.02, the oldest turn's score is multiplied
    by exp(-12) ≈ 6e-6 — effectively dropped — while turns within ~50 of the
    most recent are scored at >0.37 of their original.

    No-op when turn_index is empty (e.g. ingest didn't populate it).
    """
    if not turn_index or not hits:
        return hits
    max_idx = max(int(k) for k in turn_index)
    boosted = []
    for h in hits:
        meta = h.get("metadata", {}) or {}
        idx = meta.get("turn_idx")
        if idx is None:
            boosted.append(h)
            continue
        try:
            idx_int = int(idx)
        except (ValueError, TypeError):
            boosted.append(h)
            continue
        age = max_idx - idx_int
        factor = math.exp(-decay_lambda * age)
        new_h = dict(h)
        new_h["score"] = h.get("score", 0.0) * factor
        new_h["recency_factor"] = round(factor, 4)
        boosted.append(new_h)
    boosted.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return boosted


async def _decompose_query(
    client: httpx.AsyncClient,
    ollama_url: str,
    question: str,
) -> list[str]:
    """Ask the LLM to split question into 2-3 sub-queries.

    Returns a list of stripped lines. Falls back to [question] on failure or
    if fewer than 2 lines are returned.
    """
    try:
        raw = await _ollama_generate(
            client, ollama_url, "gemma4:e2b",
            MULTIHOP_PROMPT.format(question=question),
            temperature=0.0,
        )
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if len(lines) >= 2:
            return lines
    except Exception:
        pass
    return [question]


async def _retrieve(
    strategy: str,
    query: str,
    vmem: VectorMemory,
    top_k: int,
    reranker: object | None = None,
    fusion: str = "boost",
) -> list[dict]:
    if strategy == "vector-only":
        # fusion: "boost" (default, legacy additive keyword boost) | "rrf"
        # (Reciprocal Rank Fusion across semantic + keyword) | "none" (pure
        # semantic cosine). The fusion logic lives in VectorMemory.search.
        raw = await vmem.search(query, limit=top_k, fusion=fusion)
        hits = [{"text": r["text"], "metadata": r.get("metadata", {}),
                 "score": r.get("similarity", 0.0)} for r in raw]
        if reranker is not None and getattr(reranker, "available", False):
            hits = reranker.rerank(query, hits, top_k)
        return hits
    # TODO: wire cross-encoder reranker for strategy="full"; for now reuse retrieve() with vector source only.
    hits = await retrieve(query, strategy="thorough", sources={"vector": vmem},
                          limit=top_k, reranker=reranker, agent_name="locomo_eval")
    return [{"text": h.get("text", ""),
             "metadata": h.get("metadata", {}).get("metadata", h.get("metadata", {})),
             "score": h.get("rrf_score", h.get("source_score", 0.0))} for h in hits]


async def _process_qa(
    qa: dict,
    conv_id: str,
    vmem: VectorMemory,
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    top_k: int,
    strategy: str,
    retrieval_top_k: int,
    context_format: str,
    adjacent_turns: int,
    llm_query_expansion: bool,
    reranker: object | None,
    multihop_decompose: bool,
    full_context: bool,
    thinking_mode: bool,
    temporal_boost: float,
    fusion: str,
    turn_index: dict[str, dict],
) -> dict | None:
    if "answer" not in qa:
        return None
    question = qa["question"]
    reference = str(qa["answer"])
    category = int(qa.get("category", 0))
    evidence = qa.get("evidence", []) or []

    if full_context:
        # Bypass retrieval entirely; pass every turn of the current conversation
        # as context in turn-order. Intended for large-context generators
        # (Qwen3.6-MoE) to measure retrieval vs raw-context.
        hits = [
            {
                "text": meta.get("text", ""),
                "metadata": {
                    "turn_idx": int(idx_str),
                    "datetime": meta.get("datetime", ""),
                    "speaker": meta.get("speaker", ""),
                    "dia_id": meta.get("dia_id", ""),
                },
            }
            for idx_str, meta in sorted(turn_index.items(), key=lambda kv: int(kv[0]))
        ]
        retrieval_ms = 0.0
        adj_map: dict[str, list[str]] = {}
    else:
        # --- optional LLM query expansion ---
        retrieval_query = question
        if llm_query_expansion:
            try:
                from taosmd.query_expansion import expand_query_llm
                expansion = await expand_query_llm(question, llm_url=ollama_url, model=model)
                reformulations = expansion.get("reformulations", [])
                if reformulations:
                    retrieval_query = reformulations[0]
            except Exception:
                pass  # fall through to original question

        t0 = time.time()

        if multihop_decompose:
            sub_queries = await _decompose_query(client, ollama_url, retrieval_query)
            seen_texts: set[str] = set()
            all_hits: list[dict] = []
            for sq in sub_queries:
                sq_hits = await _retrieve(strategy, sq, vmem, retrieval_top_k, reranker, fusion=fusion)
                for h in sq_hits:
                    t = h.get("text", "")
                    if t not in seen_texts:
                        seen_texts.add(t)
                        all_hits.append(h)
            # Cap at retrieval_top_k, preserving order
            hits = all_hits[:retrieval_top_k]
        else:
            hits = await _retrieve(strategy, retrieval_query, vmem, retrieval_top_k, reranker, fusion=fusion)

        # Optional temporal-recency boost — re-rank hits with an exponential
        # decay over turn-index age. Applied AFTER retrieval/rerank so the
        # cross-encoder's relevance signal stays the primary axis but
        # break-ties favour later turns.
        if temporal_boost and temporal_boost > 0.0:
            hits = _apply_temporal_recency_boost(hits, turn_index, temporal_boost)

        retrieval_ms = (time.time() - t0) * 1000.0

        adj_map = _build_adjacent_map(hits, turn_index, adjacent_turns)

    context = _build_context(hits, context_format=context_format,
                             adjacent_turns_map=adj_map if adj_map else None)

    t1 = time.time()
    try:
        predicted = await _ollama_generate(
            client, ollama_url, model,
            ANSWER_PROMPT.format(context=context, question=question),
            thinking_mode=thinking_mode,
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

    # Resolve retrieval_top_k: falls back to top_k if not explicitly set.
    retrieval_top_k = args.retrieval_top_k if args.retrieval_top_k is not None else args.top_k

    # Initialise reranker once (shared across all QAs).
    try:
        reranker = _load_reranker(args.reranker)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results: list[dict] = []
    total_seen = 0
    failed_qa = 0
    concurrency = max(1, int(args.concurrency))
    sem = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()

    async def _guarded(
        qa: dict,
        conv_id: str,
        vmem: VectorMemory,
        client: httpx.AsyncClient,
        turn_index: dict[str, dict],
    ) -> dict | None:
        nonlocal total_seen, failed_qa
        async with sem:
            try:
                outcome = await _process_qa(
                    qa, conv_id, vmem, client,
                    args.ollama_url, args.model, args.top_k, args.strategy,
                    retrieval_top_k=retrieval_top_k,
                    context_format=args.context_format,
                    adjacent_turns=args.adjacent_turns,
                    llm_query_expansion=args.llm_query_expansion,
                    reranker=reranker,
                    multihop_decompose=args.multihop_decompose,
                    full_context=args.full_context,
                    thinking_mode=args.thinking_mode,
                    temporal_boost=args.temporal_boost,
                    fusion=args.fusion,
                    turn_index=turn_index,
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
            tmp = tempfile.mkdtemp(prefix=f"locomo_{conv_id}_")
            vmem = VectorMemory(
                db_path=os.path.join(tmp, "vmem.db"),
                qmd_url=args.qmd_url,
                embed_mode=args.embed_mode,
                onnx_path=args.onnx_path,
            )
            await vmem.init(http_client=client)
            added, ingest_s, turn_index = await _ingest_conversation(
                vmem, conv, multi_level=args.multi_level_retrieval
            )
            print(f"[{conv_id}] ingested {added} turns in {ingest_s:.1f}s", flush=True)

            # Pick eligible QAs up-front. --per-conv-limit caps QAs per
            # conversation (for balanced sampling); --limit is the global cap.
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
                tasks = [_guarded(qa, conv_id, vmem, client, turn_index) for qa in eligible]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                for outcome in gathered:
                    if isinstance(outcome, Exception) or outcome is None:
                        continue
                    results.append(outcome)
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
        "retrieval_top_k": retrieval_top_k,
        "context_format": args.context_format,
        "adjacent_turns": args.adjacent_turns,
        "llm_query_expansion": args.llm_query_expansion,
        "reranker": args.reranker,
        "multihop_decompose": args.multihop_decompose,
        "full_context": args.full_context,
        "thinking_mode": args.thinking_mode,
        "temporal_boost": args.temporal_boost,
        "fusion": args.fusion,
        "multi_level_retrieval": args.multi_level_retrieval,
        "strategy": args.strategy,
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
        out_path = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / f"locomo_{timestamp}{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"meta": meta, "by_category": by_category, "overall": overall, "results": results},
        indent=2,
    ))

    _print_summary(meta, by_category, overall)
    print(f"\nwrote {out_path}")
    return 0


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATASET = os.environ.get(
    "LOCOMO_DATASET",
    os.path.join(_REPO_ROOT, "data", "locomo", "data", "locomo10.json"),
)
_DEFAULT_ONNX = os.environ.get(
    "TAOSMD_ONNX_PATH",
    os.path.join(_REPO_ROOT, "models", "minilm-onnx"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo benchmark runner for taosmd")
    p.add_argument("--dataset", default=_DEFAULT_DATASET,
                   help=f"LoCoMo dataset JSON. Env: LOCOMO_DATASET (default: {_DEFAULT_DATASET})")
    p.add_argument("--limit", type=int, default=0, help="Cap total QAs across all conversations (0=all)")
    p.add_argument("--per-conv-limit", type=int, default=0,
                   help="Cap QAs per conversation for balanced sampling (0=all eligible)")
    p.add_argument("--conversations", type=int, default=10)
    p.add_argument("--category", default="all")
    p.add_argument("--include-adversarial", action="store_true")
    p.add_argument("--qmd-url", default="http://localhost:7832")
    p.add_argument("--embed-mode", choices=["qmd", "local", "onnx"], default="onnx",
                   help="Embedding backend. Default onnx (Fedora reference stack).")
    p.add_argument("--onnx-path", default=_DEFAULT_ONNX,
                   help=f"Path to MiniLM ONNX model directory. Env: TAOSMD_ONNX_PATH (default: {_DEFAULT_ONNX})")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--model", default="qwen3:4b")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--retrieval-top-k", type=int, default=None,
                   help="Override top-K used in retrieval independently from --top-k. "
                        "If not set, falls back to --top-k. Useful for widening the "
                        "retrieval candidate pool while keeping the scoring metric fixed.")
    p.add_argument("--context-format", choices=["plain", "session_date", "both"],
                   default="plain",
                   help="How each retrieved hit is prefixed in the answer context. "
                        "plain: '[{date}] {text}' (default). "
                        "session_date: '[Session date: {date}] {text}'. "
                        "both: '[Session date: {date}] [{date}] {text}'.")
    p.add_argument("--adjacent-turns", type=int, default=0,
                   help="Include ±N neighbouring turns from the same conversation "
                        "alongside each retrieval hit. Neighbours are deduplicated "
                        "across hits. Default 0 (disabled).")
    p.add_argument("--llm-query-expansion", action="store_true",
                   help="Before retrieval, expand the query via taosmd.query_expansion."
                        "expand_query_llm and use the first reformulation as the "
                        "retrieval query. Default off (uses original question).")
    p.add_argument("--reranker", choices=["ms-marco", "bge-v2-m3", "off"],
                   default="ms-marco",
                   help="Cross-encoder reranker applied after vector retrieval. "
                        "ms-marco: ms-marco-MiniLM-L-6-v2 ONNX (default). "
                        "bge-v2-m3: BAAI/bge-reranker-v2-m3 ONNX (model must be "
                        "present under models/bge-reranker-v2-m3-onnx/). "
                        "off: skip reranking (pure vector).")
    p.add_argument("--multihop-decompose", action="store_true",
                   help="Before retrieval, ask an LLM (gemma4:e2b) to split the "
                        "question into 2-3 sub-queries, run retrieval for each, "
                        "then union and dedupe the results up to --retrieval-top-k. "
                        "Falls back to single-query retrieval if decomposition fails "
                        "or returns fewer than 2 lines. Default off.")
    p.add_argument("--full-context", action="store_true",
                   help="Skip retrieval entirely. Feed the full conversation "
                        "(every turn, in order) as context to the generator. "
                        "Use on large-context-window generators (Qwen3.6-MoE, "
                        "long-context builds) to measure retrieval vs. raw-context "
                        "performance. --retrieval-top-k, --adjacent-turns, "
                        "--llm-query-expansion, and --multihop-decompose are "
                        "ignored when this flag is set.")
    p.add_argument("--thinking-mode", action="store_true",
                   help="Enable reasoning-mode on the generator (do not pass "
                        "think=false). For Qwen3/3.5/3.6 this lets the model "
                        "emit hidden reasoning tokens before the answer. "
                        "Slow — expect ~150s per call vs ~5s with thinking "
                        "off — but useful to test whether chain-of-thought "
                        "improves answer quality. Default off.")
    p.add_argument("--temporal-boost", type=float, default=0.0,
                   metavar="LAMBDA",
                   help="Apply exponential recency decay to retrieval scores "
                        "after vector + cross-encoder ranking. LAMBDA is the "
                        "decay rate per turn-index age (0.0 disables, 0.02 "
                        "down-weights the oldest turn in a 600-turn convo by "
                        "~6e-6, 0.005 by ~0.05). Default 0.0.")
    p.add_argument("--fusion", choices=["boost", "rrf", "none"], default="boost",
                   help="Vector + keyword fusion mode (vector-only strategy):\n"
                        "  boost: legacy additive keyword boost (0.3 * overlap, our "
                        "current default in the matrix runs).\n"
                        "  rrf: Reciprocal Rank Fusion across semantic + keyword "
                        "ranked lists (the 2026 LoCoMo recipe — already implemented "
                        "in VectorMemory.search but never invoked from the bench).\n"
                        "  none: pure semantic cosine (MemPalace-equivalent).")
    p.add_argument("--multi-level-retrieval", action="store_true",
                   help="Ingest three levels into vmem instead of one: raw turns "
                        "(default), session_summary entries (Episode-level, "
                        "narrative summaries pre-computed in the LoCoMo dataset), "
                        "and event_summary entries (Fact-level, per-speaker events). "
                        "RRF fusion at retrieval naturally surfaces the most "
                        "relevant level. HyperMem-inspired (arxiv:2604.08256) "
                        "but reuses LoCoMo's pre-computed summaries — zero LLM "
                        "ingest cost.")
    p.add_argument("--strategy", choices=["vector-only", "full"], default="vector-only")
    p.add_argument("--out", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--concurrency", type=int, default=3,
        help=("Max parallel QAs per conversation. Requires OLLAMA_NUM_PARALLEL "
              ">= N on the Ollama server. Default 3; use 1 for sequential."),
    )
    p.add_argument(
        "--timeout", type=float, default=120.0,
        help="HTTP timeout for Ollama calls in seconds. Default 120 (was 60 — "
             "bumped after observing p95 latency of 52s under concurrency=3).",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))
