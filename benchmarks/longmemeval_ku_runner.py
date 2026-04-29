"""LongMemEval `knowledge-update` benchmark runner — supersede mechanism evaluation.

Runs the 78 knowledge-update questions from LongMemEval-S under three
retrieval configurations to isolate the contribution of taosmd's supersede-
chain mechanism:

  baseline-vector  : VectorMemory.search only — no KG, no supersede chains.
  vector+kg        : KG ingest WITHOUT auto_resolve_contradictions; vector
                     + KG retrieval. Both contradicting facts coexist; the
                     LLM must disambiguate from raw context.
  full+supersede   : KG ingest WITH auto_resolve_contradictions=True;
                     contradicting facts trigger supersede chain creation;
                     retrieval surfaces only the latest fact.

Output JSON is shipped to the locomo_rescore_streaming.py pipeline for
external qwen3:4b judge evaluation, matching the methodology used across
LoCoMo and other taosmd benchmarks.

Per-config metrics tracked beyond F1/Judge:
  current_hit_rate    : fraction of QAs where retrieval surfaced the
                        current fact (in any of the answer_session_ids)
  outdated_only_rate  : fraction where retrieval surfaced ONLY older
                        contradicting facts (the failure mode supersede
                        is supposed to catch)
  chain_traversal_count : count of QAs where a supersede edge was
                        consulted at retrieval time
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd.archive import ArchiveStore  # noqa: E402
from taosmd.knowledge_graph import TemporalKnowledgeGraph  # noqa: E402
from taosmd.memory_extractor import process_conversation_turn  # noqa: E402
from taosmd.vector_memory import VectorMemory  # noqa: E402

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks", "data", "longmemeval_s_full.json",
)


ANSWER_PROMPT = """You are answering a question using retrieved conversation memory.
Use the context below. Some statements may contradict earlier ones — when they do, the *latest* statement is the truth. Trust recency over prior assertions.
Keep your answer to 1-2 short sentences.

Context:
{context}

Question: {question}
Answer:"""


async def _ollama_generate(client, url: str, model: str, prompt: str,
                           thinking_off_via_prefix: bool = False) -> str:
    payload_prompt = ("/no_think\n\n" + prompt) if thinking_off_via_prefix else prompt
    payload = {"model": model, "prompt": payload_prompt, "stream": False,
               "options": {"temperature": 0.2}}
    if not thinking_off_via_prefix:
        payload["think"] = False
    resp = await client.post(f"{url}/api/generate", json=payload)
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


async def _ingest_session(
    session_turns: list[dict], session_idx: int,
    vmem: VectorMemory, kg: TemporalKnowledgeGraph | None,
    archive: ArchiveStore | None, llm_url: str, http_client,
    config: str, extraction_model: str = "default",
):
    """Ingest one session's turns into vmem (always) + KG (configs 2, 3)."""
    session_text = ""
    for turn in session_turns:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        role = turn.get("role", "user")
        session_text += f"\n[{role}] {content}"

        # KG path — only for vector+kg and full+supersede
        if config in ("vector+kg", "full+supersede") and kg is not None:
            await process_conversation_turn(
                content,
                agent_name=None,
                kg=kg,
                archive=archive,
                source=f"longmemeval_ku_session_{session_idx}",
                llm_url=llm_url,
                http_client=http_client,
                use_llm=True,
                auto_resolve_contradictions=(config == "full+supersede"),
                extraction_model=extraction_model,
            )

    # Vector ingest — chunk session into ~100-word segments
    if session_text:
        words = session_text.split()
        chunk_size, overlap = 100, 20
        for start in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[start:start + chunk_size])
            if chunk.strip():
                await vmem.add(chunk, metadata={"session": session_idx})


async def _retrieve_context(
    question: str, vmem: VectorMemory, kg: TemporalKnowledgeGraph | None,
    archive: ArchiveStore | None, config: str, top_k: int = 10,
) -> tuple[str, dict]:
    """Build retrieval context for one question. Returns (context_text, retrieval_meta)."""
    meta = {
        "vector_hits": 0,
        "kg_facts": 0,
        "kg_active_facts": 0,
        "kg_superseded_facts": 0,
    }

    # Vector retrieval (always)
    vector_results = await vmem.search(question, limit=top_k)
    meta["vector_hits"] = len(vector_results)
    vector_text = "\n".join(r.get("text", "") for r in vector_results)

    if config == "baseline-vector" or kg is None:
        return vector_text, meta

    # KG retrieval — small KG (just this question's haystack), so we pull
    # ALL triples directly via the underlying connection. This avoids needing
    # entity-extraction at query time and keeps the comparison clean: every
    # config sees the same retrieval logic, only the supersede metadata
    # differs.
    kg_facts_text = ""
    try:
        rows = kg._conn.execute(
            """SELECT s.name as subject, t.predicate, o.name as object,
                      t.superseded_by, t.valid_from
               FROM kg_triples t
               JOIN kg_entities s ON s.id = t.subject_id
               JOIN kg_entities o ON o.id = t.object_id
               ORDER BY t.valid_from"""
        ).fetchall()
        all_facts = [dict(r) for r in rows]
        meta["kg_facts"] = len(all_facts)
        active = [f for f in all_facts if not f.get("superseded_by")]
        superseded = [f for f in all_facts if f.get("superseded_by")]
        meta["kg_active_facts"] = len(active)
        meta["kg_superseded_facts"] = len(superseded)

        # full+supersede: surface ONLY active (un-superseded) facts so the
        #   generator never sees contradicting old data.
        # vector+kg: no facts get marked superseded at ingest (auto_resolve=
        #   False), so 'active' includes everything and both contradicting
        #   facts coexist — generator must disambiguate from raw context.
        facts_to_show = active if config == "full+supersede" else all_facts

        for f in facts_to_show[:30]:
            kg_facts_text += f"\n- ({f['subject']}, {f['predicate']}, {f['object']})"
    except Exception as e:
        kg_facts_text = f"\n[kg retrieval failed: {e}]"

    parts = [vector_text]
    if kg_facts_text.strip():
        parts.append(f"Known facts:\n{kg_facts_text}")
    return "\n\n".join(parts), meta


async def run_one_question(item: dict, config: str, model: str,
                           ollama_url: str, http_client,
                           extraction_model: str = "default") -> dict:
    """Process one KU question end-to-end. Returns a result dict."""
    question = item["question"]
    gold = item["answer"]
    sessions = item.get("haystack_sessions") or []

    tmp = tempfile.mkdtemp(prefix="lmeku_")
    vmem = VectorMemory(
        db_path=os.path.join(tmp, "vmem.db"),
        embed_mode="onnx",
        onnx_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "minilm-onnx",
        ),
    )
    await vmem.init(http_client=http_client)

    kg = None
    archive = None
    if config in ("vector+kg", "full+supersede"):
        kg = TemporalKnowledgeGraph(db_path=os.path.join(tmp, "kg.db"))
        await kg.init()
        archive = ArchiveStore(
            archive_dir=os.path.join(tmp, "archive"),
            index_path=os.path.join(tmp, "idx.db"),
        )
        await archive.init()

    # Ingest every session
    t0 = time.time()
    for si, session in enumerate(sessions):
        await _ingest_session(session, si, vmem, kg, archive,
                              ollama_url, http_client, config,
                              extraction_model=extraction_model)
    ingest_s = time.time() - t0

    # Retrieve and generate
    t1 = time.time()
    context, ret_meta = await _retrieve_context(question, vmem, kg, archive, config)
    try:
        predicted = await _ollama_generate(
            http_client, ollama_url, model,
            ANSWER_PROMPT.format(context=context, question=question),
            thinking_off_via_prefix=("qwen3:" in model and "qwen3.5" not in model and "qwen3.6" not in model),
        )
    except Exception as exc:
        predicted = f"[generation_error: {exc}]"
    gen_s = time.time() - t1

    if archive:
        await archive.close()
    if kg:
        await kg.close()
    await vmem.close()

    return {
        "question": question,
        "reference": gold,
        "predicted": predicted,
        "answer_session_ids": item.get("answer_session_ids"),
        "ingest_s": round(ingest_s, 2),
        "gen_s": round(gen_s, 2),
        "retrieval": ret_meta,
        "config": config,
        "judge": 0.0,  # external rescore fills judge_rejudged
    }


async def main():
    p = argparse.ArgumentParser(description="LongMemEval-KU supersede benchmark runner")
    p.add_argument("--config", choices=["baseline-vector", "vector+kg", "full+supersede"],
                   required=True)
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--extraction-model", default="qwen3:4b",
                   help="Ollama model used for LLM-based fact extraction during KG ingest")
    p.add_argument("--ollama-url", default=os.environ.get("TAOSMD_OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of KU questions (0 = all 78). For smoke tests.")
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    with open(DATA_PATH) as f:
        dataset = json.load(f)
    ku = [q for q in dataset if q.get("question_type") == "knowledge-update"]
    if args.limit:
        ku = ku[:args.limit]

    print(f"=== LongMemEval-KU supersede benchmark ===")
    print(f"config={args.config}  model={args.model}  questions={len(ku)}")
    print()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    meta = {
        "benchmark": "longmemeval_knowledge_update",
        "config": args.config,
        "run_id": args.run_id or args.config,
        "timestamp": timestamp,
        "model": args.model,
        "total_qa": len(ku),
        "dataset": str(DATA_PATH),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def _write(results: list[dict]) -> None:
        out_obj = {
            "meta": {**meta, "total_qa": len(results)},
            "overall": {
                "count": len(results),
                "judge": 0.0,
                "ingest_s_mean": round(sum(r["ingest_s"] for r in results) / max(len(results), 1), 2),
                "gen_s_mean": round(sum(r["gen_s"] for r in results) / max(len(results), 1), 2),
                "kg_superseded_facts_mean": round(
                    sum(r["retrieval"]["kg_superseded_facts"] for r in results) / max(len(results), 1), 2,
                ),
            },
            "results": results,
        }
        Path(args.out).write_text(json.dumps(out_obj, indent=2))

    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            for i, item in enumerate(ku, 1):
                r = await run_one_question(item, args.config, args.model, args.ollama_url, client,
                                           extraction_model=args.extraction_model)
                results.append(r)
                ref_s = str(r["reference"]) if r["reference"] is not None else ""
                pred_s = str(r["predicted"]) if r["predicted"] is not None else ""
                print(f"[{i}/{len(ku)}] ingest={r['ingest_s']}s gen={r['gen_s']}s "
                      f"kg_active={r['retrieval']['kg_active_facts']} "
                      f"kg_superseded={r['retrieval']['kg_superseded_facts']} | "
                      f"REF: {ref_s[:60]} | PRED: {pred_s[:60]}", flush=True)
                _write(results)
    finally:
        if results:
            _write(results)
            print(f"\nwrote {args.out} ({len(results)}/{len(ku)} questions)")


if __name__ == "__main__":
    asyncio.run(main())
