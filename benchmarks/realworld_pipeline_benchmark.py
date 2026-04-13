#!/usr/bin/env python3
"""Real-World Pipeline Benchmark — Full taOSmd stack vs vector-only.

Unlike our other benchmarks which only test vector search, this one
ingests data through the FULL pipeline (vector + KG + archive + catalog)
and then retrieves using the parallel fan-out retriever.

This is the honest test of whether multi-layer memory actually improves
recall over vector search alone.

Configs:
  A) vector_only     — vector search with hybrid keyword boost (baseline, 97.0%)
  B) vector_kg       — vector + KG facts extracted from sessions
  C) full_pipeline   — vector + KG + archive FTS + catalog (all layers)
  D) full_fanout     — full pipeline + retrieve() with strategy="thorough"

All use: all-MiniLM-L6-v2 ONNX, same model as MemPalace.
Dataset: LongMemEval-S, 500 questions, ~48 sessions each.
"""

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd import (
    VectorMemory, KnowledgeGraph, Archive, SessionCatalog,
    CrystalStore, retrieve,
)
from taosmd.memory_extractor import extract_facts_from_text
from taosmd.query_expansion import expand_query_fast

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_full.json")
ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx")


def _user_turns_text(session: list[dict]) -> str:
    """Extract user-turns-only text from a session (our best strategy)."""
    parts = [t["content"] for t in session if t.get("role") == "user" and t.get("content")]
    text = "\n".join(parts)
    if not text:
        text = "\n".join(
            f"[{t.get('role','user')}]: {t.get('content','')}"
            for t in session if t.get("content")
        )
    return text


async def ingest_session(
    text: str,
    sid: str,
    vmem: VectorMemory,
    kg: KnowledgeGraph | None = None,
    archive: Archive | None = None,
    catalog: SessionCatalog | None = None,
) -> dict:
    """Ingest a session through available pipeline layers."""
    results = {"vector": False, "kg": 0, "archive": False}

    # Vector memory (always)
    vmem_id = await vmem.add(text[:512], metadata={"session_id": sid})
    results["vector"] = vmem_id > 0

    # KG fact extraction
    if kg:
        facts = extract_facts_from_text(text)
        for fact in facts[:10]:  # Cap per session
            try:
                await kg.add_triple(
                    fact["subject"], fact["predicate"], fact["object"],
                    source=f"session:{sid}",
                )
                results["kg"] += 1
            except Exception:
                pass

    # Archive
    if archive:
        try:
            await archive.record(
                "conversation",
                {"content": text[:1000], "session_id": sid},
                summary=text[:100],
            )
            results["archive"] = True
        except Exception:
            pass

    return results


async def run_question_vector_only(item, top_k, use_expand=True):
    """Baseline: vector search only (reproduces 97.0%)."""
    question = item["question"]
    answer_session_ids = item.get("answer_session_ids", [])
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])

    tmp = tempfile.mkdtemp()
    vmem = VectorMemory(db_path=os.path.join(tmp, "v.db"), embed_mode="onnx", onnx_path=ONNX_PATH)
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

    results = await vmem.search(search_query, limit=top_k, hybrid=True)

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


async def run_question_full_pipeline(item, top_k, use_fanout=False):
    """Full pipeline: ingest through all layers, retrieve via fan-out."""
    question = item["question"]
    answer_session_ids = item.get("answer_session_ids", [])
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])

    tmp = tempfile.mkdtemp()

    # Init all stores
    vmem = VectorMemory(db_path=os.path.join(tmp, "v.db"), embed_mode="onnx", onnx_path=ONNX_PATH)
    kg = KnowledgeGraph(os.path.join(tmp, "kg.db"))
    archive = Archive(archive_dir=os.path.join(tmp, "archive"), index_path=os.path.join(tmp, "archive-idx.db"))
    catalog = SessionCatalog(
        db_path=os.path.join(tmp, "catalog.db"),
        archive_dir=os.path.join(tmp, "archive"),
        sessions_dir=os.path.join(tmp, "sessions"),
    )

    await vmem.init()
    await kg.init()
    await archive.init()
    await catalog.init()

    # Ingest all sessions through full pipeline
    session_map = {}
    for si, session in enumerate(sessions):
        text = _user_turns_text(session)
        if not text:
            continue
        sid = session_ids[si] if si < len(session_ids) else f"s{si}"

        ingest_result = await ingest_session(text, sid, vmem, kg, archive)
        vmem_id_row = vmem._conn.execute(
            "SELECT id FROM vector_memory ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if vmem_id_row:
            session_map[vmem_id_row["id"]] = sid

    if use_fanout:
        # Use the retrieval orchestrator with all sources
        search_query = question
        expanded = expand_query_fast(question)
        entities = expanded["entities"][:3]
        if entities:
            search_query = f"{question} {' '.join(entities)}"

        sources = {
            "vector": vmem,
            "kg": kg,
            "archive": archive,
            "catalog": catalog,
        }
        results = await retrieve(
            search_query,
            strategy="thorough",
            sources=sources,
            limit=top_k,
        )

        # Extract session IDs from retrieval results
        retrieved = set()
        for r in results:
            meta = r.get("metadata", {})
            # Vector results have session_id in metadata
            if "session_id" in meta:
                retrieved.add(meta["session_id"])
            # Check nested metadata from adapters
            orig = meta.get("metadata", {})
            if isinstance(orig, dict) and "session_id" in orig:
                retrieved.add(orig["session_id"])
            # Check source_id patterns
            source_id = r.get("source_id", "")
            if source_id in session_ids:
                retrieved.add(source_id)

    else:
        # Vector + KG but search via vector only (with query expansion)
        search_query = question
        expanded = expand_query_fast(question)
        entities = expanded["entities"][:3]
        if entities:
            search_query = f"{question} {' '.join(entities)}"

        results = await vmem.search(search_query, limit=top_k, hybrid=True)

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
    await kg.close()
    await archive.close()
    await catalog.close()
    return hit


async def run_benchmark(limit: int = 500, top_k: int = 5):
    print("=" * 74)
    print(f"Real-World Pipeline Benchmark — Recall@{top_k}")
    print(f"Model: all-MiniLM-L6-v2 (ONNX) | {limit} questions")
    print(f"Tests the FULL taOSmd stack, not just vector search")
    print("=" * 74)

    with open(DATA_PATH) as f:
        dataset = json.load(f)
    dataset = dataset[:limit]

    configs = [
        ("vector_only", "Vector search + hybrid + query expansion (published 97.0%)"),
        ("full_pipeline", "Full pipeline (vector + KG + archive) — vector retrieval"),
        ("full_fanout", "Full pipeline + retrieve(strategy='thorough') fan-out"),
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

            if config_name == "vector_only":
                hit = await run_question_vector_only(item, top_k)
            elif config_name == "full_pipeline":
                hit = await run_question_full_pipeline(item, top_k, use_fanout=False)
            elif config_name == "full_fanout":
                hit = await run_question_full_pipeline(item, top_k, use_fanout=True)
            else:
                hit = False

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
                elapsed = time.time() - t_start
                print(f"  [{i+1:3d}/{len(dataset)}] {total_recall}/{total_questions} ({pct:.1f}%) — {elapsed:.0f}s")

        total_time = time.time() - t_start
        overall = total_recall / total_questions * 100

        print(f"\n  Result: {total_recall}/{total_questions} ({overall:.1f}%) in {total_time:.0f}s")
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
    print(f"REAL-WORLD PIPELINE RESULTS — Recall@{top_k}")
    print(f"{'='*74}")
    baseline = all_results.get("vector_only", {}).get("overall", 0)
    for name, desc in configs:
        r = all_results[name]
        delta = r["overall"] - baseline
        delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}" if delta != 0 else "  —"
        print(f"  {desc[:55]:<55s} {r['overall']:>5.1f}% {delta_str:>6s}")
    print(f"  {'─'*65}")
    print(f"  MemPalace (published){'':>35s}  96.6%")
    print(f"{'='*74}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "results",
                               f"realworld_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "benchmark": "realworld_pipeline",
            "description": "Full taOSmd stack benchmark — ingests through all layers, retrieves via fan-out",
            "methodology": {
                "dataset": "LongMemEval-S (500 questions, ~48 sessions/question with distractors)",
                "embedding_model": "all-MiniLM-L6-v2 (ONNX, 384-dim)",
                "ingestion": "user-turns-only, truncated to 512 chars",
                "metric": "Recall@5 — does any correct session appear in top-5 retrieved?",
            },
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_benchmark(limit=args.limit, top_k=args.top_k))
