#!/usr/bin/env python3
"""LongMemEval Recall@5 — Enhanced retrieval experiments.

Tests retrieval improvements targeting the 17 misses from hybrid baseline (96.6%):
  - 8 misses in temporal-reasoning (94.0%)
  - 3 misses in single-session-preference (90.0%)
  - 2 misses in single-session-assistant (96.4%)
  - 2 misses in multi-session (98.5%)
  - 2 misses in single-session-user (97.1%)

Configs:
  A) hybrid_baseline — reproduce 96.6% baseline
  B) query_expand    — add entity keywords from query expansion
  C) temporal_boost  — rerank results with temporal awareness
  D) combined_v2     — query expansion + temporal boost + hybrid
  E) wider_retrieval — top-10 retrieve, rerank to top-5
"""

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd import VectorMemory
from taosmd.query_expansion import expand_query_fast
from taosmd.temporal_boost import temporal_rerank, classify_temporal_query
from taosmd.preference_extractor import extract_preferences

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_full.json")
ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx")


async def run_question(item, top_k, config):
    question = item["question"]
    answer_session_ids = item.get("answer_session_ids", [])
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])

    tmp = tempfile.mkdtemp()
    vmem = VectorMemory(
        db_path=os.path.join(tmp, "v.db"),
        embed_mode="onnx",
        onnx_path=ONNX_PATH,
    )
    await vmem.init()

    # Ingest: user-turns only (proven best strategy)
    session_map = {}
    for si, session in enumerate(sessions):
        parts = [t["content"] for t in session
                 if t.get("role") == "user" and t.get("content")]
        text = "\n".join(parts)
        if not text:
            parts = [f"[{t.get('role','user')}]: {t.get('content','')}"
                     for t in session if t.get("content")]
            text = "\n".join(parts)
        if not text:
            continue

        sid = session_ids[si] if si < len(session_ids) else f"s{si}"
        vmem_id = await vmem.add(text[:512], metadata={"session_id": sid})
        if vmem_id > 0:
            session_map[vmem_id] = sid

    # Build search query based on config
    search_query = question
    retrieve_k = top_k

    if config in ("query_expand", "combined_v2"):
        expanded = expand_query_fast(question)
        # Add extracted entities to query for better keyword matching
        entities = expanded["entities"][:3]
        if entities:
            search_query = f"{question} {' '.join(entities)}"

        # For preference queries, add preference-related terms
        prefs = extract_preferences(question)
        if prefs:
            pref_terms = [p["signal"][:30] for p in prefs[:2]]
            search_query = f"{search_query} {' '.join(pref_terms)}"

    if config == "wider_retrieval":
        retrieve_k = top_k * 2  # Retrieve 10, rerank to 5

    # Retrieve
    results = await vmem.search(search_query, limit=retrieve_k, hybrid=True)

    # Temporal reranking for temporal queries
    if config in ("temporal_boost", "combined_v2", "wider_retrieval"):
        tclass = classify_temporal_query(question)
        if tclass["is_temporal"]:
            results = temporal_rerank(results, question, boost_factor=0.25)

    # Trim to top_k after reranking
    results = results[:top_k]

    # Score
    retrieved_session_ids = set()
    for r in results:
        meta = r.get("metadata", {})
        sid = meta.get("session_id", "")
        if sid:
            retrieved_session_ids.add(sid)
        vmem_id = r.get("id")
        if vmem_id in session_map:
            retrieved_session_ids.add(session_map[vmem_id])

    recall_hit = any(aid in retrieved_session_ids for aid in answer_session_ids)
    await vmem.close()
    return recall_hit


async def run_benchmark(limit: int = 500, top_k: int = 5):
    print("=" * 74)
    print(f"LongMemEval-S Enhanced Retrieval Experiments — Recall@{top_k}")
    print(f"Model: all-MiniLM-L6-v2 (ONNX) — same as MemPalace")
    print(f"Target: beat 96.6% (MemPalace) on {limit} questions")
    print("=" * 74)

    with open(DATA_PATH) as f:
        dataset = json.load(f)
    dataset = dataset[:limit]

    configs = {
        "hybrid_baseline": "Hybrid baseline (reproduce 96.6%)",
        "query_expand": "Hybrid + query expansion (entity keywords)",
        "temporal_boost": "Hybrid + temporal reranking",
        "combined_v2": "Hybrid + query expansion + temporal boost",
        "wider_retrieval": "Retrieve top-10, temporal rerank to top-5",
    }

    all_results = {}

    for config_name, config_desc in configs.items():
        print(f"\n{'─'*74}")
        print(f"Config: {config_desc}")
        print(f"{'─'*74}")

        results_by_type = {}
        total_recall = 0
        total_questions = 0
        t_start = time.time()

        for i, item in enumerate(dataset):
            qtype = item["question_type"]
            recall_hit = await run_question(item, top_k, config_name)

            total_questions += 1
            if recall_hit:
                total_recall += 1

            if qtype not in results_by_type:
                results_by_type[qtype] = {"hits": 0, "total": 0}
            results_by_type[qtype]["total"] += 1
            if recall_hit:
                results_by_type[qtype]["hits"] += 1

            if (i + 1) % 50 == 0 or i == len(dataset) - 1:
                running_pct = total_recall / total_questions * 100
                elapsed = time.time() - t_start
                print(f"  [{i+1:3d}/{len(dataset)}] {total_recall}/{total_questions} ({running_pct:.1f}%) — {elapsed:.0f}s")

        total_time = time.time() - t_start
        overall = total_recall / total_questions * 100

        print(f"\n  Result: {total_recall}/{total_questions} ({overall:.1f}%) in {total_time:.0f}s")
        print(f"  By category:")
        for qtype, data in sorted(results_by_type.items()):
            pct = data["hits"] / data["total"] * 100
            print(f"    {qtype:35s} {data['hits']:3d}/{data['total']:<3d} ({pct:.1f}%)")

        all_results[config_name] = {
            "overall": overall, "total": total_recall,
            "questions": total_questions, "time": total_time,
            "by_type": results_by_type,
        }

    # Summary
    print(f"\n{'='*74}")
    print(f"SUMMARY — Recall@{top_k}")
    print(f"{'='*74}")
    print(f"\n  {'Config':<45s} {'Score':>8s}  {'Delta':>6s}")
    print(f"  {'─'*62}")
    baseline = all_results.get("hybrid_baseline", {}).get("overall", 0)
    for name, desc in configs.items():
        r = all_results[name]
        delta = r["overall"] - baseline
        delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}" if delta != 0 else "  —"
        print(f"  {desc:<45s} {r['overall']:>5.1f}%  {delta_str:>6s}")
    print(f"  {'─'*62}")
    print(f"  MemPalace (published){'':>25s}  96.6%")
    print(f"  agentmemory (published){'':>22s}  95.2%")
    print(f"{'='*74}")

    output_path = os.path.join(os.path.dirname(__file__), "results",
                               f"enhanced_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"benchmark": "enhanced", "top_k": top_k, "results": all_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_benchmark(limit=args.limit, top_k=args.top_k))
