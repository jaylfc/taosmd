#!/usr/bin/env python3
"""Judge-free evidence-COVERAGE probe for taosmd on LoCoMo.

Question under test: is the weakest LoCoMo category (MultiHop) limited by
RETRIEVAL COVERAGE — gold evidence spans never reaching the generator's
context — rather than by generation? An audit of existing runner results
showed MultiHop averages only ~0.43 of its gold evidence spans retrieved
(mean of per-question ``evidence_hits / evidence_total``), a ceiling no
prompt can fix. This probe measures whether WIDER retrieval (a larger
candidate pool and/or more adjacent turns) raises that coverage.

RETRIEVAL ONLY. It never loads a generator LLM and never calls a judge, so
it is CPU-only (ONNX embedder + ONNX cross-encoder reranker) and does not
touch the GPU or Ollama. If you see a generator or judge model load, that is
a wiring bug — the probe must never generate.

It reuses locomo_runner's machinery wholesale (no reimplementation of the
gold-evidence matching):

  - dataset load + conversation slice   -> same defaults / env vars
  - per-conversation ingest             -> locomo_runner._ingest_conversation
  - retrieval                           -> locomo_runner._retrieve
    (through VectorMemory.search; --fusion / --reranker behave identically)
  - gold-evidence matching              -> locomo_runner._evidence_hits
  - adjacency neighbour selection       -> locomo_runner._build_adjacent_map

Two coverage numbers per question (both judge-free):

  coverage_hits    = _evidence_hits(hits) / evidence_total
      Gold spans present in the RETRIEVED hits. This is exactly the runner's
      ``evidence_hits`` field, so it reproduces the audited 0.43 MultiHop /
      0.94 Temporal baseline. With a reranker ON, hits narrow to --top-k, so
      only candidate_top_k moves this number; adjacent_turns cannot.

  coverage_context = |gold spans in (hits UNION adjacency neighbours)| / evidence_total
      Gold spans that actually reach the GENERATOR CONTEXT — either as a
      top-k hit or as an adjacency-injected neighbour turn. This is the true
      coverage ceiling the generator faces, and the number adjacent_turns can
      move. Neighbour dia_ids are recovered from the runner's own
      _build_adjacent_map output (text -> dia_id via turn_index), so the
      adjacency logic is the runner's, unchanged.

Per the corrected LoCoMo category mapping used across taosmd docs:
  1 = MultiHop   2 = Temporal   3 = OpenDomain   4 = SingleHop   (5 = Adversarial, excluded)
(Note: locomo_runner.CATEGORY_NAMES carries the OLD/wrong labels; this probe
prints the corrected ones. Category NUMBERS are unchanged.)

Coverage denominators only count QAs that carry evidence annotations
(evidence_total > 0), matching the runner's R@K convention.

Failure posture: per-QA exceptions are counted; if more than --max-error-rate
of attempted QAs error, or an empty corpus is detected (canary embed), the run
aborts loudly with a non-zero exit rather than reporting a hollow 0.000.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _BENCH_DIR)

from taosmd.vector_memory import VectorMemory  # noqa: E402

from locomo_runner import (  # noqa: E402
    _DEFAULT_DATASET,
    _DEFAULT_ONNX,
    _build_adjacent_map,
    _evidence_hits,
    _git_sha,
    _ingest_conversation,
    _load_reranker,
    _retrieve,
)

# Same category set the runner scores by default (adversarial excluded).
INCLUDE_CATS = {1, 2, 3, 4}

# Corrected LoCoMo category labels (the runner's CATEGORY_NAMES are wrong).
CORRECTED_CATEGORY_NAMES = {
    1: "MultiHop",
    2: "Temporal",
    3: "OpenDomain",
    4: "SingleHop",
    5: "Adversarial",
}


def _adjacency_dia_ids(
    hits: list[dict], turn_index: dict[str, dict], adjacent_turns: int
) -> set:
    """dia_ids of the adjacency neighbour turns the runner would inject.

    Reuses locomo_runner._build_adjacent_map (the runner's exact ±N neighbour
    selection, returning neighbour TEXT) and resolves each neighbour text back
    to its dia_id via turn_index. No reimplementation of neighbour selection.
    """
    if adjacent_turns <= 0 or not turn_index:
        return set()
    text_to_dia: dict[str, str] = {}
    for meta in turn_index.values():
        t = meta.get("text")
        d = meta.get("dia_id")
        if t and d:
            text_to_dia.setdefault(t, d)
    adj_map = _build_adjacent_map(hits, turn_index, adjacent_turns)
    dia_ids: set = set()
    for texts in adj_map.values():
        for t in texts:
            d = text_to_dia.get(t)
            if d:
                dia_ids.add(d)
    return dia_ids


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _summarize(rows: list[dict]) -> dict:
    """Per-category + overall coverage. Denominator = evidence-bearing QAs."""
    by_cat: dict[str, dict] = {}
    cats: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        cats[int(r["category"])].append(r)

    def cat_block(c_rows: list[dict]) -> dict:
        ev = [r for r in c_rows if r["evidence_total"] > 0]
        hits_frac = [r["evidence_hits"] / r["evidence_total"] for r in ev]
        ctx_frac = [r["evidence_context"] / r["evidence_total"] for r in ev]
        pool_h = sum(r["evidence_hits"] for r in ev)
        pool_c = sum(r["evidence_context"] for r in ev)
        pool_t = sum(r["evidence_total"] for r in ev)
        atk_hits = [1 if r["evidence_hits"] > 0 else 0 for r in ev]
        atk_ctx = [1 if r["evidence_context"] > 0 else 0 for r in ev]
        return {
            "count": len(c_rows),
            "n_with_evidence": len(ev),
            "coverage_hits": round(_mean(hits_frac), 4),
            "coverage_context": round(_mean(ctx_frac), 4),
            "pooled_coverage_hits": round(pool_h / pool_t, 4) if pool_t else 0.0,
            "pooled_coverage_context": round(pool_c / pool_t, 4) if pool_t else 0.0,
            "r_at_k_hits": round(_mean(atk_hits), 4),
            "r_at_k_context": round(_mean(atk_ctx), 4),
        }

    for c in sorted(cats):
        by_cat[str(c)] = {"name": CORRECTED_CATEGORY_NAMES.get(c, f"cat-{c}"),
                          **cat_block(cats[c])}
    overall = {"name": "Overall", **cat_block(rows)}
    return {"by_category": by_cat, "overall": overall}


def _print_summary(config: dict, summary: dict) -> None:
    sep, dash = "=" * 78, "-" * 78
    print(sep)
    print(f"Coverage probe (judge-free): taosmd (commit {config['git_sha']})")
    print(sep)
    print(f"candidate_top_k={config['candidate_top_k']} adj={config['adjacent_turns']} "
          f"top_k={config['top_k']} fusion={config['fusion']} "
          f"reranker={config['reranker']} embed={config['embed_mode']} "
          f"n={config['limit']}")
    print(dash)
    print(f"{'Category':<12} {'n':>4} {'n_ev':>5} "
          f"{'cov_hits':>9} {'cov_ctx':>9} {'R@K_hit':>8} {'R@K_ctx':>8}")
    print(dash)
    order = sorted(summary["by_category"].items(), key=lambda kv: int(kv[0]))
    for cat, s in order:
        print(f"{s['name']:<12} {s['count']:>4} {s['n_with_evidence']:>5} "
              f"{s['coverage_hits']:>9.3f} {s['coverage_context']:>9.3f} "
              f"{s['r_at_k_hits']:>8.3f} {s['r_at_k_context']:>8.3f}")
    print(dash)
    o = summary["overall"]
    print(f"{'Overall':<12} {o['count']:>4} {o['n_with_evidence']:>5} "
          f"{o['coverage_hits']:>9.3f} {o['coverage_context']:>9.3f} "
          f"{o['r_at_k_hits']:>8.3f} {o['r_at_k_context']:>8.3f}")
    print(sep)


async def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}\n"
              f"Set --dataset or the LOCOMO_DATASET env var.", file=sys.stderr)
        return 2
    conversations = json.loads(dataset_path.read_text())[: args.conversations]

    try:
        reranker = _load_reranker(args.reranker)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    reranker_available = bool(getattr(reranker, "available", False)) if reranker else False
    if reranker is not None and not reranker_available:
        # A silently-degraded reranker (missing model.onnx -> passthrough) would
        # change what this probe measures. Refuse, as the latency probe does.
        print(f"ERROR: --reranker {args.reranker} requested but the model is not "
              f"loadable (model.onnx missing?). Pass --reranker off, or install "
              f"the model.", file=sys.stderr)
        return 2

    results: list[dict] = []
    errors: list[dict] = []
    attempted = 0

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for conv in conversations:
            conv_id = conv.get("sample_id", "unknown")
            tmp = tempfile.mkdtemp(prefix=f"cov_probe_{conv_id}_")
            vmem = VectorMemory(
                db_path=os.path.join(tmp, "vmem.db"),
                qmd_url=args.qmd_url,
                embed_mode=args.embed_mode,
                onnx_path=args.onnx_path,
            )
            await vmem.init(http_client=client)
            # Canary: a missing ONNX model silently embeds nothing and would
            # "succeed" at 0.000 coverage. Refuse to measure an empty corpus.
            canary = await vmem.embed("canary embedding check", task="search_document")
            if not canary:
                print(f"ERROR: embed() returned empty for mode={args.embed_mode!r} "
                      f"(onnx_path={args.onnx_path!r}, qmd_url={args.qmd_url!r}). "
                      "The embedding backend is not available.", file=sys.stderr)
                return 2

            added, ingest_s, turn_index = await _ingest_conversation(vmem, conv)
            print(f"[{conv_id}] ingested {added} turns in {ingest_s:.1f}s", flush=True)

            for qa in conv.get("qa", []) or []:
                if "answer" not in qa:
                    continue
                category = int(qa.get("category", 0))
                if category not in INCLUDE_CATS:
                    continue
                if args.limit and attempted >= args.limit:
                    break
                attempted += 1
                question = qa["question"]
                evidence = qa.get("evidence", []) or []
                try:
                    t0 = time.time()
                    hits = await _retrieve(
                        "vector-only", question, vmem, args.candidate_top_k,
                        reranker, fusion=args.fusion, rerank_top_k=args.top_k,
                    )
                    retrieval_ms = (time.time() - t0) * 1000.0

                    ev_hits = _evidence_hits(hits, evidence)
                    # Context coverage = evidence in hits OR adjacency neighbours.
                    # Recover neighbour dia_ids from the runner's adjacency map
                    # and present them to _evidence_hits as pseudo-hits so the
                    # gold-matching logic itself is never reimplemented.
                    adj_dia_ids = _adjacency_dia_ids(hits, turn_index, args.adjacent_turns)
                    pseudo = [{"metadata": {"dia_id": d}} for d in adj_dia_ids]
                    ev_context = _evidence_hits(hits + pseudo, evidence)

                    results.append({
                        "conversation_id": conv_id,
                        "question": question,
                        "category": category,
                        "n_hits": len(hits),
                        "n_adjacent_dia_ids": len(adj_dia_ids),
                        "evidence_total": len(evidence),
                        "evidence_hits": ev_hits,
                        "evidence_context": ev_context,
                        "retrieval_ms": round(retrieval_ms, 2),
                    })
                except Exception as exc:
                    errors.append({
                        "conversation_id": conv_id, "question": question,
                        "category": category,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"  qa ERROR [{conv_id}] {type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)

                if attempted >= 10 and len(errors) / attempted > args.max_error_rate:
                    print(f"\nABORT: {len(errors)}/{attempted} QAs errored "
                          f"(> {args.max_error_rate:.0%}). Config is broken. "
                          f"First error: {errors[0]['error']}", file=sys.stderr)
                    await vmem.close()
                    return 1
                if attempted % 25 == 0:
                    print(f"  progress: {attempted} QAs "
                          f"({len(results)} ok, {len(errors)} errors)", flush=True)

            await vmem.close()
            if args.limit and attempted >= args.limit:
                break

    if not attempted:
        print("ERROR: no eligible QAs found (empty slice after category filter).",
              file=sys.stderr)
        return 1
    if not results:
        print(f"ERROR: ALL {attempted} retrievals errored, zero results. "
              f"First error: {errors[0]['error'] if errors else 'unknown'}",
              file=sys.stderr)
        return 1
    if len(errors) / attempted > args.max_error_rate:
        print(f"\nABORT: {len(errors)}/{attempted} QAs errored "
              f"(> {args.max_error_rate:.0%}). Not writing a summary for a "
              f"broken run. First error: {errors[0]['error']}", file=sys.stderr)
        return 1

    config = {
        "dataset": str(dataset_path),
        "limit": args.limit,
        "conversations": min(args.conversations, len(conversations)),
        "candidate_top_k": args.candidate_top_k,
        "top_k": args.top_k,
        "adjacent_turns": args.adjacent_turns,
        "fusion": args.fusion,
        "reranker": args.reranker,
        "embed_mode": args.embed_mode,
        "onnx_path": args.onnx_path,
        "git_sha": _git_sha(),
        "timestamp": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "n_errors": len(errors),
    }
    summary = _summarize(results)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (Path(_BENCH_DIR) / "results"
                    / f"coverage_probe_{config['timestamp']}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": config, "results": results, "errors": errors,
         "summary": summary}, indent=2))

    _print_summary(config, summary)
    print(f"\nwrote {out_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Judge-free LoCoMo evidence-coverage probe (retrieval only; "
                    "no generation, no judge). Reports coverage by category.")
    p.add_argument("--dataset", default=_DEFAULT_DATASET,
                   help=f"LoCoMo dataset JSON. Env: LOCOMO_DATASET "
                        f"(default: {_DEFAULT_DATASET})")
    p.add_argument("--limit", type=int, default=200,
                   help="Cap total QAs across all conversations (0=all). "
                        "Default 200 to match the subset-200 probes.")
    p.add_argument("--conversations", type=int, default=10)
    p.add_argument("--candidate-top-k", type=int, default=50, dest="candidate_top_k",
                   help="Candidate pool fetched from vector search before rerank "
                        "narrowing (the runner's retrieval_top_k). Default 50 "
                        "(the shipped maxsim-rerank-9b leader recipe pool).")
    p.add_argument("--top-k", type=int, default=10,
                   help="Final hit count after reranking (rerank_top_k). "
                        "Default 10.")
    p.add_argument("--adjacent-turns", type=int, default=2,
                   help="Include ±N neighbouring turns per hit when computing "
                        "coverage_context. Default 2.")
    p.add_argument("--fusion",
                   choices=["boost", "rrf", "bm25_rrf", "bm25_lemma_rrf",
                            "mem0_additive", "none"],
                   default="mem0_additive",
                   help="Vector + keyword fusion mode. Default mem0_additive "
                        "(matches the audited leader config).")
    p.add_argument("--reranker", choices=["ms-marco", "bge-v2-m3", "off"],
                   default="bge-v2-m3",
                   help="Cross-encoder reranker after vector retrieval. Default "
                        "bge-v2-m3 (the audited leader config). A "
                        "requested-but-unloadable reranker is a hard error.")
    p.add_argument("--embed-mode", choices=["qmd", "local", "onnx"],
                   default="onnx", help="Embedding backend. Default onnx.")
    p.add_argument("--onnx-path", default=_DEFAULT_ONNX,
                   help=f"MiniLM ONNX model dir. Env: TAOSMD_ONNX_PATH "
                        f"(default: {_DEFAULT_ONNX})")
    p.add_argument("--qmd-url", default="http://localhost:7832",
                   help="qmd server URL (only used with --embed-mode qmd).")
    p.add_argument("--max-error-rate", type=float, default=0.10,
                   help="Abort (exit 1) when more than this share of attempted "
                        "QAs raise during retrieval. Default 0.10.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="HTTP timeout in seconds (qmd embed mode only).")
    p.add_argument("--out", default=None,
                   help="Results JSON path (default: "
                        "benchmarks/results/coverage_probe_<ts>.json)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))
