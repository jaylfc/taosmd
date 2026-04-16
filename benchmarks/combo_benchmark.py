#!/usr/bin/env python3
"""Combo Benchmark — Exhaustive pipeline path combinations.

Tests every combination of retrieval techniques to find the optimal config.
Each technique can be toggled independently.

Techniques:
  - Embedding: minilm (baseline) / nomic (if available)
  - Query: raw / expanded (entity keywords) / hyde (hypothetical doc)
  - Search: semantic-only / hybrid-boost / hybrid-rrf
  - Retrieval: vector-only / fanout (vector+kg+archive+catalog)
  - Reranking: none / temporal-boost / cross-encoder (if available)
  - Session dedup: off / max-2-per-session
  - Chunk enrichment: off / prepend-context (LLM at index time)

Runs on LongMemEval-S with the Pi-equivalent model (qwen3:4b).
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from taosmd import VectorMemory, KnowledgeGraph, Archive, SessionCatalog
from taosmd.memory_extractor import extract_facts_from_text
from taosmd.query_expansion import expand_query_fast
from taosmd.temporal_boost import temporal_rerank, classify_temporal_query
from taosmd.retrieval import retrieve, _rrf_merge, _deduplicate, _adapt_vector

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_full.json")
ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx")

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:4b"


def _user_turns_text(session):
    parts = [t["content"] for t in session if t.get("role") == "user" and t.get("content")]
    text = "\n".join(parts)
    if not text:
        text = "\n".join(f"[{t.get('role','user')}]: {t.get('content','')}" for t in session if t.get("content"))
    return text


async def generate_hyde(query: str, client: httpx.AsyncClient) -> str:
    """Generate a hypothetical document that answers the query."""
    prompt = f"""Write a short paragraph (2-3 sentences) that directly answers this question as if you knew the answer from a conversation history:

Question: {query}

Write the answer as if recalling from memory. Be specific and factual."""

    try:
        resp = await client.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 150},
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            return content
    except Exception:
        pass
    return query  # Fallback to original query


async def enrich_chunk(text: str, client: httpx.AsyncClient) -> str:
    """Prepend contextual description to a chunk before embedding (Anthropic's technique)."""
    prompt = f"""Write a single sentence describing what this conversation excerpt is about. Be specific about the topic, people, and context.

Excerpt: {text[:500]}

Description:"""

    try:
        resp = await client.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 60},
            timeout=15,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            return f"{content}\n\n{text}"
    except Exception:
        pass
    return text


async def run_question(item, top_k, config, http_client=None):
    question = item["question"]
    answer_session_ids = item.get("answer_session_ids", [])
    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])

    tmp = tempfile.mkdtemp()
    vmem = VectorMemory(db_path=os.path.join(tmp, "v.db"), embed_mode="onnx", onnx_path=ONNX_PATH)
    await vmem.init()

    kg = None
    archive = None
    if config.get("fanout"):
        kg = KnowledgeGraph(os.path.join(tmp, "kg.db"))
        archive = Archive(archive_dir=os.path.join(tmp, "archive"), index_path=os.path.join(tmp, "aidx.db"))
        await kg.init()
        await archive.init()

    # Ingest
    session_map = {}
    for si, session in enumerate(sessions):
        text = _user_turns_text(session)
        if not text:
            continue
        sid = session_ids[si] if si < len(session_ids) else f"s{si}"

        # Chunk enrichment
        ingest_text = text[:512]
        if config.get("enrich") and http_client:
            ingest_text = await enrich_chunk(text, http_client)
            ingest_text = ingest_text[:512]

        vmem_id = await vmem.add(ingest_text, metadata={"session_id": sid})
        if vmem_id > 0:
            session_map[vmem_id] = sid

        # KG extraction for fanout
        if kg:
            facts = extract_facts_from_text(text)
            for f in facts[:5]:
                try:
                    await kg.add_triple(f["subject"], f["predicate"], f["object"], source=f"s:{sid}")
                except Exception:
                    pass

        # Archive for fanout
        if archive:
            await archive.record("conversation", {"content": text[:500], "session_id": sid}, summary=text[:80])

    # Build search query
    search_query = question
    if config.get("expand"):
        expanded = expand_query_fast(question)
        entities = expanded["entities"][:3]
        if entities:
            search_query = f"{question} {' '.join(entities)}"

    if config.get("hyde") and http_client:
        hyde_doc = await generate_hyde(question, http_client)
        search_query = hyde_doc

    if config.get("multi_query") and http_client:
        # Use both original + expanded + hyde, retrieve for each, union
        queries = [question]
        expanded = expand_query_fast(question)
        if expanded["entities"]:
            queries.append(f"{question} {' '.join(expanded['entities'][:3])}")
        hyde_doc = await generate_hyde(question, http_client)
        if hyde_doc != question:
            queries.append(hyde_doc)

        all_results = []
        for q in queries:
            fusion = config.get("fusion", "boost")
            r = await vmem.search(q, limit=top_k * 2, hybrid=config.get("hybrid", True), fusion=fusion)
            all_results.extend(r)

        # Deduplicate by session_id, keep highest similarity
        seen = {}
        for r in all_results:
            sid = r.get("metadata", {}).get("session_id", "")
            vid = r.get("id")
            key = sid or str(vid)
            if key not in seen or r.get("similarity", 0) > seen[key].get("similarity", 0):
                seen[key] = r
        results = sorted(seen.values(), key=lambda x: x.get("similarity", 0), reverse=True)[:top_k * 2]
    elif config.get("fanout"):
        sources = {"vector": vmem}
        if kg:
            sources["kg"] = kg
        if archive:
            sources["archive"] = archive
        results = await retrieve(search_query, strategy="thorough", sources=sources, limit=top_k * 2)
    else:
        fusion = config.get("fusion", "boost")
        results = await vmem.search(search_query, limit=top_k * 2, hybrid=config.get("hybrid", True), fusion=fusion)

    # Temporal reranking
    if config.get("temporal"):
        tclass = classify_temporal_query(question)
        if tclass["is_temporal"]:
            results = temporal_rerank(results, question, boost_factor=0.25)

    # Session dedup — max N results per session
    if config.get("session_dedup"):
        session_counts = {}
        deduped = []
        for r in results:
            sid = r.get("metadata", {}).get("session_id", "")
            vid = r.get("id")
            key = sid or str(vid)
            session_counts[key] = session_counts.get(key, 0) + 1
            if session_counts[key] <= 2:
                deduped.append(r)
        results = deduped

    results = results[:top_k]

    # Extract session IDs
    retrieved = set()
    for r in results:
        meta = r.get("metadata", {})
        if isinstance(meta, dict):
            sid = meta.get("session_id", "")
            if sid:
                retrieved.add(sid)
            orig = meta.get("metadata", {})
            if isinstance(orig, dict) and "session_id" in orig:
                retrieved.add(orig["session_id"])
        vid = r.get("id")
        if vid and vid in session_map:
            retrieved.add(session_map[vid])

    hit = any(aid in retrieved for aid in answer_session_ids)

    await vmem.close()
    if kg:
        await kg.close()
    if archive:
        await archive.close()
    return hit


async def run_benchmark(limit: int = 500, top_k: int = 5):
    print("=" * 78)
    print(f"Combo Benchmark — Exhaustive pipeline path combinations")
    print(f"LLM: {LLM_MODEL} | Embed: MiniLM ONNX | {limit} questions | Top-{top_k}")
    print("=" * 78)

    # Check Ollama
    has_ollama = False
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                has_ollama = any(LLM_MODEL in m for m in models)
    except Exception:
        pass
    print(f"Ollama: {'available' if has_ollama else 'unavailable'} ({LLM_MODEL})")

    with open(DATA_PATH) as f:
        dataset = json.load(f)
    dataset = dataset[:limit]

    # Define configs — from simple to complex
    configs = [
        # Baselines
        {"name": "semantic_only", "hybrid": False},
        {"name": "hybrid_boost", "hybrid": True, "fusion": "boost"},
        {"name": "hybrid_boost+expand", "hybrid": True, "fusion": "boost", "expand": True},
        # RRF variants
        {"name": "hybrid_rrf", "hybrid": True, "fusion": "rrf"},
        {"name": "hybrid_rrf+expand", "hybrid": True, "fusion": "rrf", "expand": True},
        # Temporal
        {"name": "boost+expand+temporal", "hybrid": True, "fusion": "boost", "expand": True, "temporal": True},
        # Session dedup
        {"name": "boost+expand+dedup", "hybrid": True, "fusion": "boost", "expand": True, "session_dedup": True},
        # Fan-out
        {"name": "fanout+boost+expand", "hybrid": True, "fusion": "boost", "expand": True, "fanout": True},
    ]

    # LLM-dependent configs (only if Ollama available)
    if has_ollama:
        configs.extend([
            {"name": "hyde", "hybrid": True, "fusion": "boost", "hyde": True},
            {"name": "multi_query", "hybrid": True, "fusion": "boost", "multi_query": True},
            {"name": "hyde+expand", "hybrid": True, "fusion": "boost", "hyde": True, "expand": True},
            {"name": "enrich+boost+expand", "hybrid": True, "fusion": "boost", "expand": True, "enrich": True},
            {"name": "KITCHEN_SINK", "hybrid": True, "fusion": "boost", "expand": True, "temporal": True,
             "session_dedup": True, "multi_query": True, "fanout": True},
        ])

    all_results = {}

    async with httpx.AsyncClient(timeout=120) as client:
        for cfg in configs:
            name = cfg.pop("name")
            print(f"\n{'─'*78}")
            print(f"{name}")
            print(f"  Config: {cfg}")
            print(f"{'─'*78}")

            results_by_type = {}
            total_recall = 0
            total_questions = 0
            t_start = time.time()

            for i, item in enumerate(dataset):
                qtype = item["question_type"]
                hit = await run_question(item, top_k, cfg, client if has_ollama else None)
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
                    per_q = elapsed / total_questions
                    print(f"  [{i+1:3d}/{len(dataset)}] {total_recall}/{total_questions} ({pct:.1f}%) — {elapsed:.0f}s ({per_q:.1f}s/q)")

            total_time = time.time() - t_start
            overall = total_recall / total_questions * 100
            print(f"\n  RESULT: {total_recall}/{total_questions} ({overall:.1f}%) in {total_time:.0f}s")
            for qtype, data in sorted(results_by_type.items()):
                pct = data["hits"] / data["total"] * 100
                print(f"    {qtype:35s} {data['hits']:3d}/{data['total']:<3d} ({pct:.1f}%)")

            all_results[name] = {
                "overall": overall, "total": total_recall, "questions": total_questions,
                "time": round(total_time, 1), "per_question": round(total_time / total_questions, 2),
                "config": cfg, "by_type": results_by_type,
            }
            cfg["name"] = name  # Restore for logging

    # Leaderboard
    print(f"\n{'='*78}")
    print(f"LEADERBOARD — Recall@{top_k} on {limit} questions")
    print(f"{'='*78}")
    ranked = sorted(all_results.items(), key=lambda x: x[1]["overall"], reverse=True)
    for i, (name, r) in enumerate(ranked):
        marker = " ***" if r["overall"] > 97.0 else ""
        print(f"  {i+1:2d}. {name:<40s} {r['overall']:>5.1f}% ({r['total']}/{r['questions']}) {r['per_question']:.1f}s/q{marker}")
    print(f"  {'─'*70}")
    print(f"      MemPalace (published){'':>22s}  96.6%")
    print(f"      taOSmd published{'':>27s}  97.0%")
    print(f"{'='*78}")

    output_path = os.path.join(os.path.dirname(__file__), "results",
                               f"combo_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "benchmark": "combo_exhaustive",
            "model": LLM_MODEL, "embedding": "all-MiniLM-L6-v2 ONNX",
            "questions": limit, "top_k": top_k,
            "results": all_results,
            "leaderboard": [(name, r["overall"]) for name, r in ranked],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=str, default=LLM_MODEL)
    args = parser.parse_args()
    LLM_MODEL = args.model
    asyncio.run(run_benchmark(limit=args.limit, top_k=args.top_k))
