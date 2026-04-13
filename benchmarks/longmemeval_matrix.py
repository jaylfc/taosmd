#!/usr/bin/env python3
"""LongMemEval Recall@5 — Full Comparison Matrix.

Runs 4 configurations to produce honest, defensible numbers:

  A) MemPalace-match: user-turns-only, raw semantic (no hybrid) — fair comparison to MemPalace 96.6%
  B) taOSmd v0.1:     user-turns-only, hybrid search — our published approach
  C) taOSmd v0.2:     user-turns-only, hybrid + graph expansion (retrieval-only, no scan cheat)
  D) All-turns:        full session text, hybrid — to show ingestion strategy impact

All use: all-MiniLM-L6-v2 via ONNX, same model as MemPalace.
All use: fresh index per question, top-5 retrieval, recall_any@5.
Dataset: LongMemEval-S, 500 questions.

This is the benchmark we publish. Every number must be reproducible.
"""

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd import VectorMemory, KnowledgeGraph
from taosmd.graph_expansion import expand_from_results
from taosmd.memory_extractor import extract_facts_from_text

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_full.json")
ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx")


def ingest_session_text(session: list[dict], mode: str) -> str:
    """Build ingestible text from a session.

    Modes:
      user_turns — only user messages, raw content (MemPalace methodology)
      all_turns  — all messages with [role]: prefix
    """
    if mode == "user_turns":
        parts = [t["content"] for t in session
                 if t.get("role") == "user" and t.get("content")]
        text = "\n".join(parts)
        # Fallback to all turns if no user turns
        if not text:
            parts = [f"[{t.get('role','user')}]: {t.get('content','')}"
                     for t in session if t.get("content")]
            text = "\n".join(parts)
        return text
    else:  # all_turns
        parts = [f"[{t.get('role','user')}]: {t.get('content','')}"
                 for t in session if t.get("content")]
        return "\n".join(parts)


async def run_question(item, top_k, config):
    """Run a single question through a specific configuration.

    Configs:
      mempalace_match — user_turns, raw semantic only (hybrid=False)
      v01_hybrid      — user_turns, hybrid search
      v02_graph       — user_turns, hybrid + KG graph expansion (retrieval-valid)
      all_turns       — all_turns, hybrid search
    """
    question = item["question"]
    answer_session_ids = item.get("answer_session_ids", [])
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])

    # Determine ingestion mode
    ingest_mode = "all_turns" if config == "all_turns" else "user_turns"
    use_hybrid = config != "mempalace_match"
    use_graph = config == "v02_graph"

    tmp = tempfile.mkdtemp()
    vmem = VectorMemory(
        db_path=os.path.join(tmp, "v.db"),
        embed_mode="onnx",
        onnx_path=ONNX_PATH,
    )
    await vmem.init()

    kg = None
    if use_graph:
        kg = KnowledgeGraph(os.path.join(tmp, "kg.db"))
        await kg.init()

    # Ingest sessions
    session_map = {}  # vmem_id -> session_id
    for si, session in enumerate(sessions):
        text = ingest_session_text(session, ingest_mode)
        if not text:
            continue

        sid = session_ids[si] if si < len(session_ids) else f"s{si}"
        vmem_id = await vmem.add(text[:512], metadata={"session_id": sid})
        if vmem_id > 0:
            session_map[vmem_id] = sid

        # Extract facts for graph expansion (cap to avoid slowdown)
        if kg:
            facts = extract_facts_from_text(text)
            for fact in facts[:5]:
                try:
                    await kg.add_triple(
                        fact["subject"], fact["predicate"], fact["object"],
                        source=f"session:{sid}",
                    )
                except Exception:
                    pass

    # Retrieve
    results = await vmem.search(question, limit=top_k, hybrid=use_hybrid)

    # Collect retrieved session IDs (ONLY from top-k results)
    retrieved_session_ids = set()
    for r in results:
        meta = r.get("metadata", {})
        sid = meta.get("session_id", "")
        if sid:
            retrieved_session_ids.add(sid)
        vmem_id = r.get("id")
        if vmem_id in session_map:
            retrieved_session_ids.add(session_map[vmem_id])

    # Graph expansion: use KG to rerank/boost, but ONLY within top-k results
    # This is the valid version — we don't scan all sessions, we only use
    # the KG to check if expanded entities give us confidence in existing
    # top-k results. We do NOT add new session IDs from outside top-k.
    if kg and results:
        expanded = await expand_from_results(kg, results, max_hops=2, max_expanded=5)
        # Graph expansion provides additional context but does NOT expand
        # the retrieved set beyond top-k. It's useful for QA accuracy (L2/L3
        # context assembly) but doesn't change Recall@k by design.

    recall_hit = any(aid in retrieved_session_ids for aid in answer_session_ids)

    await vmem.close()
    if kg:
        await kg.close()

    return recall_hit


async def run_matrix(limit: int = 500, top_k: int = 5):
    print("=" * 74)
    print(f"LongMemEval-S Recall@{top_k} — Full Comparison Matrix")
    print(f"Model: all-MiniLM-L6-v2 (ONNX, 384-dim) — same as MemPalace")
    print(f"Dataset: {limit} questions, fresh index per question")
    print("=" * 74)

    with open(DATA_PATH) as f:
        dataset = json.load(f)
    dataset = dataset[:limit]

    configs = {
        "mempalace_match": "User-turns, raw semantic (MemPalace method)",
        "v01_hybrid": "User-turns, hybrid semantic+keyword (taOSmd v0.1)",
        "v02_graph": "User-turns, hybrid + KG graph expansion (taOSmd v0.2)",
        "all_turns": "All-turns, hybrid (harder test)",
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
                print(f"  [{i+1:3d}/{len(dataset)}] {total_recall}/{total_questions} ({running_pct:.1f}%) — {elapsed:.0f}s total")

        total_time = time.time() - t_start
        overall = total_recall / total_questions * 100 if total_questions > 0 else 0

        print(f"\n  Result: {total_recall}/{total_questions} ({overall:.1f}%) in {total_time:.0f}s")
        print(f"  By category:")
        for qtype, data in sorted(results_by_type.items()):
            pct = data["hits"] / data["total"] * 100 if data["total"] > 0 else 0
            print(f"    {qtype:35s} {data['hits']:3d}/{data['total']:<3d} ({pct:.1f}%)")

        all_results[config_name] = {
            "overall": overall,
            "total": total_recall,
            "questions": total_questions,
            "time": total_time,
            "by_type": results_by_type,
        }

    # Final comparison table
    print(f"\n{'='*74}")
    print(f"FINAL RESULTS — Recall@{top_k} on LongMemEval-S ({limit} questions)")
    print(f"{'='*74}")
    print(f"\n{'System':<50s} {'Score':>8s}")
    print(f"{'─'*58}")

    for config_name, config_desc in configs.items():
        r = all_results[config_name]
        label = config_desc.split("(")[1].rstrip(")") if "(" in config_desc else config_name
        print(f"  taOSmd {label:<44s} {r['overall']:>5.1f}%")

    print(f"  {'─'*56}")
    print(f"  MemPalace (published, same model){'':>18s}  96.6%")
    print(f"  agentmemory (published, same model){'':>16s}  95.2%")
    print(f"  SuperMemory{'':>40s}  81.6%")

    print(f"\n  Notes:")
    print(f"  - All taOSmd runs use all-MiniLM-L6-v2 ONNX (same model as MemPalace)")
    print(f"  - 'MemPalace method' = user-turns only, pure cosine similarity, no hybrid")
    print(f"  - 'hybrid' = cosine + 30% keyword overlap boost")
    print(f"  - 'graph expansion' provides richer context but doesn't change Recall@k")
    print(f"  - 'all-turns' includes assistant responses (harder, more noise)")
    print(f"{'='*74}")

    # Save results to JSON for reproducibility
    output_path = os.path.join(os.path.dirname(__file__), "results",
                               f"matrix_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "benchmark": "LongMemEval-S",
            "top_k": top_k,
            "questions": limit,
            "model": "all-MiniLM-L6-v2 (ONNX)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_matrix(limit=args.limit, top_k=args.top_k))
