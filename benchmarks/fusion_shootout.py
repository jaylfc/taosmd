#!/usr/bin/env python3
"""Fusion Strategy Shootout — Find the best retrieval approach.

Tests: raw semantic, additive boost, RRF, and RRF + temporal rerank + wider retrieval.
All on LongMemEval-S full dataset with all-MiniLM-L6-v2 ONNX.
"""

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd import VectorMemory
from taosmd.temporal_boost import temporal_rerank, classify_temporal_query
from taosmd.query_expansion import expand_query_fast

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_full.json")
ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx")


async def run_question(item, top_k, config):
    question = item["question"]
    answer_session_ids = item.get("answer_session_ids", [])
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])

    tmp = tempfile.mkdtemp()
    vmem = VectorMemory(db_path=os.path.join(tmp, "v.db"), embed_mode="onnx", onnx_path=ONNX_PATH)
    await vmem.init()

    # Ingest: user-turns only
    session_map = {}
    for si, session in enumerate(sessions):
        parts = [t["content"] for t in session if t.get("role") == "user" and t.get("content")]
        text = "\n".join(parts)
        if not text:
            text = "\n".join(f"[{t.get('role','user')}]: {t.get('content','')}" for t in session if t.get("content"))
        if not text:
            continue
        sid = session_ids[si] if si < len(session_ids) else f"s{si}"
        vmem_id = await vmem.add(text[:512], metadata={"session_id": sid})
        if vmem_id > 0:
            session_map[vmem_id] = sid

    # Search with config-specific parameters
    search_query = question
    retrieve_k = top_k

    if config == "raw_semantic":
        results = await vmem.search(question, limit=top_k, hybrid=False)
    elif config == "additive_boost":
        results = await vmem.search(question, limit=top_k, hybrid=True, fusion="boost")
    elif config == "rrf":
        results = await vmem.search(question, limit=top_k, hybrid=True, fusion="rrf")
    elif config == "rrf_temporal":
        results = await vmem.search(question, limit=top_k, hybrid=True, fusion="rrf")
        tclass = classify_temporal_query(question)
        if tclass["is_temporal"]:
            results = temporal_rerank(results, question, boost_factor=0.25)
    elif config == "rrf_wide_temporal":
        # Retrieve wider, rerank, trim
        results = await vmem.search(question, limit=top_k * 2, hybrid=True, fusion="rrf")
        tclass = classify_temporal_query(question)
        if tclass["is_temporal"]:
            results = temporal_rerank(results, question, boost_factor=0.25)
        results = results[:top_k]
    elif config == "rrf_expand":
        expanded = expand_query_fast(question)
        entities = expanded["entities"][:3]
        if entities:
            search_query = f"{question} {' '.join(entities)}"
        results = await vmem.search(search_query, limit=top_k, hybrid=True, fusion="rrf")
    elif config == "rrf_expand_temporal":
        expanded = expand_query_fast(question)
        entities = expanded["entities"][:3]
        if entities:
            search_query = f"{question} {' '.join(entities)}"
        results = await vmem.search(search_query, limit=top_k * 2, hybrid=True, fusion="rrf")
        tclass = classify_temporal_query(question)
        if tclass["is_temporal"]:
            results = temporal_rerank(results, question, boost_factor=0.25)
        results = results[:top_k]
    else:
        results = await vmem.search(question, limit=top_k, hybrid=True, fusion="rrf")

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
    print(f"Fusion Strategy Shootout — Recall@{top_k} on LongMemEval-S")
    print(f"Model: all-MiniLM-L6-v2 (ONNX) | {limit} questions")
    print("=" * 74)

    with open(DATA_PATH) as f:
        dataset = json.load(f)
    dataset = dataset[:limit]

    configs = [
        ("raw_semantic", "Raw cosine (MemPalace-equivalent)"),
        ("additive_boost", "Additive keyword boost (old hybrid)"),
        ("rrf", "Reciprocal Rank Fusion"),
        ("rrf_temporal", "RRF + temporal rerank"),
        ("rrf_wide_temporal", "RRF wide (top-10→5) + temporal"),
        ("rrf_expand", "RRF + query expansion"),
        ("rrf_expand_temporal", "RRF + expand + wide + temporal (kitchen sink)"),
    ]

    all_results = {}

    for config_name, config_desc in configs:
        print(f"\n{'─'*74}")
        print(f"{config_desc}")
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
                pct = total_recall / total_questions * 100
                print(f"  [{i+1:3d}/{len(dataset)}] {total_recall}/{total_questions} ({pct:.1f}%) — {time.time()-t_start:.0f}s")

        overall = total_recall / total_questions * 100
        print(f"\n  Result: {total_recall}/{total_questions} ({overall:.1f}%)")
        for qtype, data in sorted(results_by_type.items()):
            pct = data["hits"] / data["total"] * 100
            print(f"    {qtype:35s} {data['hits']:3d}/{data['total']:<3d} ({pct:.1f}%)")

        all_results[config_name] = {"overall": overall, "total": total_recall,
                                     "questions": total_questions, "by_type": results_by_type}

    # Final table
    print(f"\n{'='*74}")
    print(f"SHOOTOUT RESULTS — Recall@{top_k}")
    print(f"{'='*74}")
    baseline = all_results.get("raw_semantic", {}).get("overall", 0)
    for name, desc in configs:
        r = all_results[name]
        delta = r["overall"] - baseline
        delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}" if delta != 0 else "  —"
        print(f"  {desc:<50s} {r['overall']:>5.1f}% {delta_str:>6s}")
    print(f"  {'─'*60}")
    print(f"  MemPalace (published){'':>30s}  96.6%")
    print(f"{'='*74}")

    output_path = os.path.join(os.path.dirname(__file__), "results",
                               f"fusion_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": all_results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_benchmark(limit=args.limit, top_k=args.top_k))
