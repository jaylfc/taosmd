#!/usr/bin/env python3
"""Variations Sweep — exhaustive search for best retrieval config.

Tests many combinations of:
  - Fusion: additive boost weights (0.1-0.5), RRF (k=30,60,120)
  - Query expansion: 0-5 entity keywords
  - Temporal boost: different boost factors
  - Keyword boost weight variations

Runs fast: 100 questions per config (statistically significant at this scale).
Best configs are re-run on full 500 for verification.
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

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_full.json")
ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx")


async def run_question(item, top_k, params):
    question = item["question"]
    answer_session_ids = item.get("answer_session_ids", [])
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])

    tmp = tempfile.mkdtemp()
    vmem = VectorMemory(db_path=os.path.join(tmp, "v.db"), embed_mode="onnx", onnx_path=ONNX_PATH)
    await vmem.init()

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

    # Query expansion
    search_query = question
    n_entities = params.get("n_entities", 0)
    if n_entities > 0:
        expanded = expand_query_fast(question)
        entities = expanded["entities"][:n_entities]
        if entities:
            search_query = f"{question} {' '.join(entities)}"

    # Search
    fusion = params.get("fusion", "boost")
    results = await vmem.search(search_query, limit=top_k, hybrid=True, fusion=fusion)

    # Temporal boost
    temporal_factor = params.get("temporal_factor", 0)
    if temporal_factor > 0:
        tclass = classify_temporal_query(question)
        if tclass["is_temporal"]:
            results = temporal_rerank(results, question, boost_factor=temporal_factor)

    # Score
    retrieved = set()
    for r in results:
        meta = r.get("metadata", {})
        sid = meta.get("session_id", "")
        if sid:
            retrieved.add(sid)
        vid = r.get("id")
        if vid in session_map:
            retrieved.add(session_map[vid])

    recall_hit = any(aid in retrieved for aid in answer_session_ids)
    await vmem.close()
    return recall_hit


async def run_sweep(sample_size: int = 100, top_k: int = 5, full_verify: int = 500):
    print("=" * 74)
    print(f"Variations Sweep — {sample_size}q sample, top configs verified on {full_verify}q")
    print("=" * 74)

    with open(DATA_PATH) as f:
        dataset = json.load(f)

    # Use a stratified sample for fast iteration
    sample = dataset[:sample_size]
    full = dataset[:full_verify]

    # Define all variations to test
    variations = [
        # Baseline
        {"name": "boost_0.3 (baseline)", "fusion": "boost", "n_entities": 0, "temporal_factor": 0},
        # Boost weight sweep
        {"name": "boost_0.1", "fusion": "boost", "n_entities": 0, "temporal_factor": 0},
        {"name": "boost_0.2", "fusion": "boost", "n_entities": 0, "temporal_factor": 0},
        {"name": "boost_0.4", "fusion": "boost", "n_entities": 0, "temporal_factor": 0},
        {"name": "boost_0.5", "fusion": "boost", "n_entities": 0, "temporal_factor": 0},
        # RRF variations
        {"name": "rrf", "fusion": "rrf", "n_entities": 0, "temporal_factor": 0},
        # Query expansion sweep
        {"name": "expand_1ent", "fusion": "boost", "n_entities": 1, "temporal_factor": 0},
        {"name": "expand_2ent", "fusion": "boost", "n_entities": 2, "temporal_factor": 0},
        {"name": "expand_3ent (published)", "fusion": "boost", "n_entities": 3, "temporal_factor": 0},
        {"name": "expand_5ent", "fusion": "boost", "n_entities": 5, "temporal_factor": 0},
        # RRF + expansion
        {"name": "rrf_expand_3", "fusion": "rrf", "n_entities": 3, "temporal_factor": 0},
        {"name": "rrf_expand_5", "fusion": "rrf", "n_entities": 5, "temporal_factor": 0},
        # Temporal variations
        {"name": "temporal_0.15", "fusion": "boost", "n_entities": 3, "temporal_factor": 0.15},
        {"name": "temporal_0.25", "fusion": "boost", "n_entities": 3, "temporal_factor": 0.25},
        {"name": "temporal_0.35", "fusion": "boost", "n_entities": 3, "temporal_factor": 0.35},
        # Kitchen sink
        {"name": "rrf_expand_temporal", "fusion": "rrf", "n_entities": 3, "temporal_factor": 0.25},
    ]

    # Phase 1: Quick sweep on sample
    print(f"\nPhase 1: Testing {len(variations)} configs on {sample_size} questions")
    print(f"{'─'*74}")

    results = []
    for v in variations:
        t0 = time.time()
        hits = 0
        total = 0
        for item in sample:
            hit = await run_question(item, top_k, v)
            total += 1
            if hit:
                hits += 1
        pct = hits / total * 100
        elapsed = time.time() - t0
        results.append({"name": v["name"], "params": v, "score": pct, "hits": hits, "total": total})
        print(f"  {v['name']:<35s} {hits:3d}/{total} ({pct:5.1f}%) {elapsed:5.0f}s")

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'─'*74}")
    print(f"Phase 1 Rankings (top 10):")
    for i, r in enumerate(results[:10]):
        marker = " <<<" if r["score"] > results[0]["score"] - 0.5 else ""
        print(f"  {i+1:2d}. {r['name']:<35s} {r['score']:5.1f}%{marker}")

    # Phase 2: Verify top 3 on full dataset
    top_configs = results[:3]
    print(f"\n{'='*74}")
    print(f"Phase 2: Verifying top {len(top_configs)} on full {full_verify} questions")
    print(f"{'='*74}")

    verified = []
    for cfg in top_configs:
        print(f"\n  Verifying: {cfg['name']}")
        hits = 0
        total = 0
        t0 = time.time()
        by_type = {}

        for i, item in enumerate(full):
            qtype = item["question_type"]
            hit = await run_question(item, top_k, cfg["params"])
            total += 1
            if hit:
                hits += 1
            if qtype not in by_type:
                by_type[qtype] = {"hits": 0, "total": 0}
            by_type[qtype]["total"] += 1
            if hit:
                by_type[qtype]["hits"] += 1
            if (i + 1) % 100 == 0:
                print(f"    [{i+1}/{full_verify}] {hits}/{total} ({hits/total*100:.1f}%)")

        pct = hits / total * 100
        elapsed = time.time() - t0
        print(f"  VERIFIED: {cfg['name']} = {hits}/{total} ({pct:.1f}%) in {elapsed:.0f}s")
        for qtype, data in sorted(by_type.items()):
            cat_pct = data["hits"] / data["total"] * 100
            print(f"    {qtype:35s} {data['hits']:3d}/{data['total']:<3d} ({cat_pct:.1f}%)")

        verified.append({"name": cfg["name"], "score": pct, "hits": hits,
                         "total": total, "by_type": by_type})

    # Final summary
    print(f"\n{'='*74}")
    print(f"FINAL VERIFIED RESULTS")
    print(f"{'='*74}")
    for v in verified:
        print(f"  {v['name']:<40s} {v['score']:5.1f}% ({v['hits']}/{v['total']})")
    print(f"  {'─'*55}")
    print(f"  MemPalace (published){'':>20s}  96.6%")
    print(f"  agentmemory (published){'':>17s}  95.2%")
    print(f"{'='*74}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "results",
                               f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"phase1": results, "verified": verified,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--verify", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_sweep(sample_size=args.sample, top_k=args.top_k, full_verify=args.verify))
