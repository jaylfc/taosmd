#!/usr/bin/env python3
"""Retrieval-only LoCoMo probe for taosmd.

Measures retrieval quality (evidence recall, R@K) and per-query wall-clock
latency with NO LLM generation and NO Ollama dependency, so retrieval levers
(fusion modes, late interaction / MaxSim, ColBERT models, binary quant,
cross-encoder rerank) can be compared for viability on CPU-only boxes.

Reuses locomo_runner's machinery wholesale rather than reimplementing it:

  - dataset loading + conversation slicing (same defaults, same env vars)
  - per-conversation corpus build/ingest  -> locomo_runner._ingest_conversation
  - the retrieval call                    -> locomo_runner._retrieve
    (goes through VectorMemory.search, so --late-interaction,
    --colbert-model, --binary-quant and --fusion behave identically)
  - evidence judging                      -> locomo_runner._evidence_hits

R@K convention matches locomo_runner._summary: share of QAs that have
evidence annotations (evidence_total > 0) where at least one evidence
dia_id was retrieved. Category 5 (adversarial) is excluded, as in the
runner's default category set, so numbers line up with the subset-200
probes.

Failure posture (this probe exists because a previous probe "completed"
with zero results): per-QA exceptions are counted and reported; if more
than --max-error-rate of attempted QAs error the run aborts loudly with a
non-zero exit; a run where every retrieval errored also exits non-zero.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _BENCH_DIR)

from taosmd.vector_memory import VectorMemory  # noqa: E402

from locomo_runner import (  # noqa: E402
    CATEGORY_NAMES,
    _DEFAULT_DATASET,
    _DEFAULT_ONNX,
    _build_adjacent_map,
    _evidence_hits,
    _git_sha,
    _ingest_conversation,
    _load_reranker,
    _retrieve,
)

# Same category set the runner uses by default (adversarial excluded).
INCLUDE_CATS = {1, 2, 3, 4}


def _pctl(values: list[float], q: float) -> float:
    """Nearest-rank percentile. q in [0, 1]."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]


def _latency_stats(values: list[float]) -> dict:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": round(_pctl(values, 0.50), 2),
        "p95_ms": round(_pctl(values, 0.95), 2),
        "mean_ms": round(sum(values) / len(values), 2),
    }


def _recall_at_k(rows: list[dict]) -> tuple[float, int]:
    """(R@K, n_with_evidence) over rows. runner convention: only QAs that
    carry evidence annotations count toward the denominator."""
    ev_rows = [r for r in rows if r.get("evidence_total", 0) > 0]
    if not ev_rows:
        return 0.0, 0
    hits = sum(1 for r in ev_rows if r["evidence_hits"] > 0)
    return hits / len(ev_rows), len(ev_rows)


def _summarize(rows: list[dict], errors: list[dict],
               ingest_stats: list[dict]) -> dict:
    recall, n_ev = _recall_at_k(rows)
    by_cat: dict[str, dict] = {}
    cats: dict[int, list[dict]] = {}
    for r in rows:
        cats.setdefault(int(r["category"]), []).append(r)
    for cat in sorted(cats):
        c_rows = cats[cat]
        c_recall, c_nev = _recall_at_k(c_rows)
        by_cat[str(cat)] = {
            "name": CATEGORY_NAMES.get(cat, f"cat-{cat}"),
            "count": len(c_rows),
            "n_with_evidence": c_nev,
            "r_at_k": round(c_recall, 4),
            **_latency_stats([r["retrieval_ms"] for r in c_rows]),
        }
    return {
        "n": len(rows),
        "n_errors": len(errors),
        "n_with_evidence": n_ev,
        "r_at_k": round(recall, 4),
        "retrieval_latency": _latency_stats([r["retrieval_ms"] for r in rows]),
        "by_category": by_cat,
        "ingest": {
            "conversations": len(ingest_stats),
            "total_items": sum(s["added"] for s in ingest_stats),
            "total_ingest_s": round(sum(s["ingest_s"] for s in ingest_stats), 2),
            "per_conversation": ingest_stats,
        },
    }


def _print_summary(config: dict, summary: dict) -> None:
    sep, dash = "=" * 64, "-" * 64
    li = "colbert" if config["colbert_model"] else (
        "maxsim" if config["late_interaction"] else "dense")
    print(sep)
    print(f"Retrieval-only LoCoMo probe: taosmd (commit {config['git_sha']})")
    print(sep)
    print(f"mode={li} embed={config['embed_mode']} fusion={config['fusion']} "
          f"reranker={config['reranker']} top_k={config['top_k']} "
          f"retrieval_top_k={config['retrieval_top_k']} "
          f"adj={config['adjacent_turns']} binq={config['binary_quant']}")
    if config["colbert_model"]:
        print(f"colbert_model={config['colbert_model']}")
    print(dash)
    print(f"{'Category':<18} {'n':>4} {'R@K':>6} {'p50ms':>9} {'p95ms':>9} {'mean':>9}")
    print(dash)
    for cat, s in sorted(summary["by_category"].items(), key=lambda kv: int(kv[0])):
        print(f"{s['name']:<18} {s['count']:>4} {s['r_at_k']:>6.3f} "
              f"{s['p50_ms']:>9.1f} {s['p95_ms']:>9.1f} {s['mean_ms']:>9.1f}")
    print(dash)
    lat = summary["retrieval_latency"]
    print(f"{'Overall':<18} {summary['n']:>4} {summary['r_at_k']:>6.3f} "
          f"{lat['p50_ms']:>9.1f} {lat['p95_ms']:>9.1f} {lat['mean_ms']:>9.1f}")
    ing = summary["ingest"]
    print(dash)
    print(f"ingest+index: {ing['total_items']} items across "
          f"{ing['conversations']} conversations in {ing['total_ingest_s']:.1f}s")
    if summary["n_errors"]:
        print(f"errors: {summary['n_errors']} QA(s) raised during retrieval "
              f"(see 'errors' in the output JSON)")
    print(sep)


async def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}\n"
              f"Set --dataset or the LOCOMO_DATASET env var.", file=sys.stderr)
        return 2
    conversations = json.loads(dataset_path.read_text())[: args.conversations]

    retrieval_top_k = args.retrieval_top_k
    try:
        reranker = _load_reranker(args.reranker)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    reranker_available = bool(getattr(reranker, "available", False)) if reranker else False
    if reranker is not None and not reranker_available:
        # CrossEncoderReranker degrades to passthrough when model.onnx is
        # missing; that would silently change what this probe measures.
        print(f"ERROR: --reranker {args.reranker} requested but the model is "
              f"not loadable (model.onnx missing?). Pass --reranker off to "
              f"probe without reranking, or install the model.", file=sys.stderr)
        return 2

    results: list[dict] = []
    errors: list[dict] = []
    ingest_stats: list[dict] = []
    attempted = 0

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for conv in conversations:
            conv_id = conv.get("sample_id", "unknown")
            tmp = tempfile.mkdtemp(prefix=f"locomo_probe_{conv_id}_")
            vmem = VectorMemory(
                db_path=os.path.join(tmp, "vmem.db"),
                qmd_url=args.qmd_url,
                embed_mode=args.embed_mode,
                onnx_path=args.onnx_path,
                binary_quant=args.binary_quant,
                late_interaction=args.late_interaction,
                colbert_model=args.colbert_model,
            )
            await vmem.init(http_client=client)
            added, ingest_s, turn_index = await _ingest_conversation(vmem, conv)
            ingest_stats.append({
                "conversation_id": conv_id,
                "added": added,
                "ingest_s": round(ingest_s, 2),
            })
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
                        "vector-only", question, vmem, retrieval_top_k,
                        reranker, fusion=args.fusion, rerank_top_k=args.top_k,
                    )
                    retrieval_ms = (time.time() - t0) * 1000.0
                    # Adjacency injection is a context-builder step in the
                    # runner (doesn't change which dia_ids count as hits),
                    # but its cost matters on CPU, so time it separately.
                    t1 = time.time()
                    adj_map = _build_adjacent_map(hits, turn_index, args.adjacent_turns)
                    adjacency_ms = (time.time() - t1) * 1000.0
                    results.append({
                        "conversation_id": conv_id,
                        "question": question,
                        "category": category,
                        "n_hits": len(hits),
                        "n_adjacent": sum(len(v) for v in adj_map.values()),
                        "evidence_hits": _evidence_hits(hits, evidence),
                        "evidence_total": len(evidence),
                        "retrieval_ms": round(retrieval_ms, 2),
                        "adjacency_ms": round(adjacency_ms, 2),
                    })
                except Exception as exc:
                    errors.append({
                        "conversation_id": conv_id,
                        "question": question,
                        "category": category,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"  qa ERROR [{conv_id}] {type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)

                # Abort early if the error rate blows past the threshold;
                # never let a broken config "finish" quietly.
                if attempted >= 10 and len(errors) / attempted > args.max_error_rate:
                    print(f"\nABORT: {len(errors)}/{attempted} QAs errored "
                          f"(> {args.max_error_rate:.0%} threshold). The "
                          f"retrieval config is broken; fix it instead of "
                          f"trusting partial numbers. First error: "
                          f"{errors[0]['error']}", file=sys.stderr)
                    await vmem.close()
                    return 1

                if attempted % 25 == 0:
                    print(f"  progress: {attempted} QAs "
                          f"({len(results)} ok, {len(errors)} errors)", flush=True)

            await vmem.close()
            if args.limit and attempted >= args.limit:
                break

    if not attempted:
        print("ERROR: no eligible QAs found (dataset slice empty after "
              "category filtering). Nothing was measured.", file=sys.stderr)
        return 1
    if not results:
        print(f"ERROR: ALL {attempted} retrievals errored, zero results. "
              f"This run measured nothing. First error: "
              f"{errors[0]['error'] if errors else 'unknown'}", file=sys.stderr)
        return 1
    if len(errors) / attempted > args.max_error_rate:
        print(f"\nABORT: {len(errors)}/{attempted} QAs errored "
              f"(> {args.max_error_rate:.0%} threshold). Not writing a "
              f"summary for a broken run. First error: {errors[0]['error']}",
              file=sys.stderr)
        return 1

    config = {
        "dataset": str(dataset_path),
        "limit": args.limit,
        "conversations": min(args.conversations, len(conversations)),
        "top_k": args.top_k,
        "retrieval_top_k": retrieval_top_k,
        "adjacent_turns": args.adjacent_turns,
        "fusion": args.fusion,
        "embed_mode": args.embed_mode,
        "onnx_path": args.onnx_path,
        "qmd_url": args.qmd_url,
        "reranker": args.reranker,
        "binary_quant": args.binary_quant,
        "late_interaction": args.late_interaction or bool(args.colbert_model),
        "colbert_model": args.colbert_model,
        "git_sha": _git_sha(),
        "timestamp": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    }
    summary = _summarize(results, errors, ingest_stats)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (Path(_BENCH_DIR) / "results"
                    / f"retrieval_probe_{config['timestamp']}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": config, "results": results, "errors": errors,
         "summary": summary},
        indent=2,
    ))

    _print_summary(config, summary)
    print(f"\nwrote {out_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval-only LoCoMo probe (no LLM generation, no "
                    "Ollama). Reports R@K + per-query retrieval latency.")
    p.add_argument("--dataset", default=_DEFAULT_DATASET,
                   help=f"LoCoMo dataset JSON. Env: LOCOMO_DATASET "
                        f"(default: {_DEFAULT_DATASET})")
    p.add_argument("--limit", type=int, default=200,
                   help="Cap total QAs across all conversations (0=all). "
                        "Default 200 to match the subset-200 probes.")
    p.add_argument("--conversations", type=int, default=10)
    p.add_argument("--top-k", type=int, default=10,
                   help="Final hit count after reranking (rerank_top_k in the "
                        "runner's coarse-to-fine path). Default 10.")
    p.add_argument("--retrieval-top-k", type=int, default=20,
                   help="Candidate pool fetched from vector search before any "
                        "rerank narrowing. Default 20.")
    p.add_argument("--adjacent-turns", type=int, default=2,
                   help="Build the ±N adjacency map per hit (timed as "
                        "adjacency_ms; does not change R@K, mirroring the "
                        "runner where adjacency only widens generator "
                        "context). Default 2.")
    p.add_argument("--fusion",
                   choices=["boost", "rrf", "bm25_rrf", "bm25_lemma_rrf",
                            "mem0_additive", "none"],
                   default="mem0_additive",
                   help="Vector + keyword fusion mode (see locomo_runner). "
                        "Default mem0_additive to match the subset-200 probes.")
    p.add_argument("--embed-mode", choices=["qmd", "local", "onnx"],
                   default="onnx",
                   help="Embedding backend. Default onnx.")
    p.add_argument("--onnx-path", default=_DEFAULT_ONNX,
                   help=f"MiniLM ONNX model dir. Env: TAOSMD_ONNX_PATH "
                        f"(default: {_DEFAULT_ONNX})")
    p.add_argument("--qmd-url", default="http://localhost:7832",
                   help="qmd server URL (only used with --embed-mode qmd).")
    p.add_argument("--reranker", choices=["ms-marco", "bge-v2-m3", "off"],
                   default="ms-marco",
                   help="Cross-encoder reranker after vector retrieval "
                        "(runner default ms-marco). Unlike the runner, a "
                        "requested-but-unloadable reranker is a hard error "
                        "here, not a silent passthrough.")
    p.add_argument("--binary-quant", action="store_true",
                   help="Sign-bit Hamming first-stage scoring "
                        "(VectorMemory.binary_quant).")
    p.add_argument("--late-interaction", action="store_true",
                   help="MiniLM token-level MaxSim (ColBERT-style late "
                        "interaction) instead of pooled cosine.")
    p.add_argument("--colbert-model", default="", dest="colbert_model",
                   help="HF name or local path of a proper ColBERT model via "
                        "sentence-transformers. Implies --late-interaction "
                        "and --embed-mode local (handled inside VectorMemory)."
                        " Example: answerdotai/answerai-colbert-small-v1")
    p.add_argument("--max-error-rate", type=float, default=0.10,
                   help="Abort (exit 1) when more than this share of "
                        "attempted QAs raise during retrieval. Default 0.10.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="HTTP timeout in seconds (qmd embed mode only).")
    p.add_argument("--out", default=None,
                   help="Results JSON path (default: "
                        "benchmarks/results/retrieval_probe_<ts>.json)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))
