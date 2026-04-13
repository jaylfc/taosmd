#!/usr/bin/env python3
"""Embedding Model Comparison — MiniLM vs Nomic on LongMemEval-S.

Same methodology as MemPalace (raw semantic, user-turns, fresh index per question).
Tests if switching embedding models improves baseline recall.
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

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_full.json")
MINILM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx")
NOMIC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nomic-onnx")


def _user_turns_text(session):
    parts = [t["content"] for t in session if t.get("role") == "user" and t.get("content")]
    return "\n".join(parts) or "\n".join(
        f"[{t.get('role','user')}]: {t.get('content','')}" for t in session if t.get("content")
    )


async def run_question(item, top_k, onnx_path, use_hybrid, use_expand):
    question = item["question"]
    answer_session_ids = item.get("answer_session_ids", [])
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])

    tmp = tempfile.mkdtemp()
    vmem = VectorMemory(db_path=os.path.join(tmp, "v.db"), embed_mode="onnx", onnx_path=onnx_path)
    await vmem.init()

    session_map = {}
    for si, session in enumerate(sessions):
        text = _user_turns_text(session)
        if not text:
            continue
        sid = session_ids[si] if si < len(session_ids) else f"s{si}"
        vmem_id = await vmem.add(text[:512], metadata={"session_id": sid})
        if vmem_id > 0:
            session_map[vmem_id] = sid

    search_query = question
    if use_expand:
        expanded = expand_query_fast(question)
        entities = expanded["entities"][:3]
        if entities:
            search_query = f"{question} {' '.join(entities)}"

    results = await vmem.search(search_query, limit=top_k, hybrid=use_hybrid)

    retrieved = set()
    for r in results:
        meta = r.get("metadata", {})
        sid = meta.get("session_id", "")
        if sid:
            retrieved.add(sid)
        vid = r.get("id")
        if vid in session_map:
            retrieved.add(session_map[vid])

    hit = any(aid in retrieved for aid in answer_session_ids)
    await vmem.close()
    return hit


async def run_benchmark(limit: int = 500, top_k: int = 5):
    print("=" * 78)
    print(f"Embedding Model Comparison — Recall@{top_k} on {limit} questions")
    print("=" * 78)

    with open(DATA_PATH) as f:
        dataset = json.load(f)
    dataset = dataset[:limit]

    # Check model availability
    nomic_available = os.path.exists(os.path.join(NOMIC_PATH, "onnx", "model.onnx"))
    print(f"MiniLM: {MINILM_PATH} {'✓' if os.path.exists(os.path.join(MINILM_PATH, 'model.onnx')) else '✗'}")
    print(f"Nomic:  {NOMIC_PATH} {'✓' if nomic_available else '✗'}")

    configs = [
        ("minilm_raw", MINILM_PATH, False, False, "MiniLM raw semantic (MemPalace method)"),
        ("minilm_hybrid", MINILM_PATH, True, False, "MiniLM hybrid boost"),
        ("minilm_hybrid_expand", MINILM_PATH, True, True, "MiniLM hybrid + expand (published 97.0%)"),
    ]
    if nomic_available:
        configs.extend([
            ("nomic_raw", NOMIC_PATH + "/onnx", False, False, "Nomic raw semantic"),
            ("nomic_hybrid", NOMIC_PATH + "/onnx", True, False, "Nomic hybrid boost"),
            ("nomic_hybrid_expand", NOMIC_PATH + "/onnx", True, True, "Nomic hybrid + expand"),
        ])

    all_results = {}
    for name, onnx_path, hybrid, expand, desc in configs:
        # Check ONNX model exists at path
        model_file = os.path.join(onnx_path, "model.onnx")
        if not os.path.exists(model_file):
            print(f"\n  SKIP: {desc} — model not found at {model_file}")
            continue

        print(f"\n{'─'*78}")
        print(f"{desc}")
        print(f"{'─'*78}")

        results_by_type = {}
        total_recall = 0
        total_questions = 0
        t_start = time.time()

        for i, item in enumerate(dataset):
            qtype = item["question_type"]
            hit = await run_question(item, top_k, onnx_path, hybrid, expand)
            total_questions += 1
            if hit:
                total_recall += 1
            if qtype not in results_by_type:
                results_by_type[qtype] = {"hits": 0, "total": 0}
            results_by_type[qtype]["total"] += 1
            if hit:
                results_by_type[qtype]["hits"] += 1
            if (i + 1) % 50 == 0 or i == len(dataset) - 1:
                pct = total_recall / total_questions * 100
                print(f"  [{i+1:3d}/{len(dataset)}] {total_recall}/{total_questions} ({pct:.1f}%) — {time.time()-t_start:.0f}s")

        overall = total_recall / total_questions * 100
        total_time = time.time() - t_start
        print(f"\n  RESULT: {total_recall}/{total_questions} ({overall:.1f}%)")
        for qtype, data in sorted(results_by_type.items()):
            pct = data["hits"] / data["total"] * 100
            print(f"    {qtype:35s} {data['hits']:3d}/{data['total']:<3d} ({pct:.1f}%)")

        all_results[name] = {"overall": overall, "total": total_recall, "time": total_time,
                             "by_type": results_by_type, "description": desc}

    # Summary
    print(f"\n{'='*78}")
    print(f"EMBEDDING COMPARISON — Recall@{top_k}")
    print(f"{'='*78}")
    for name, _, _, _, desc in configs:
        if name in all_results:
            r = all_results[name]
            print(f"  {desc:<50s} {r['overall']:>5.1f}%")
    print(f"  {'─'*58}")
    print(f"  MemPalace (MiniLM, raw semantic){'':>19s}  96.6%")
    print(f"{'='*78}")

    output_path = os.path.join(os.path.dirname(__file__), "results",
                               f"embedding_{time.strftime('%Y%m%d_%H%M%S')}.json")
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
