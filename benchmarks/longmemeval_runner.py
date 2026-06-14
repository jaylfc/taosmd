#!/usr/bin/env python3
"""Run LongMemEval benchmark against taOSmd.

This runs the official LongMemEval-Oracle benchmark (500 questions).
Each question has conversation sessions as context. We:
1. Ingest sessions into taOSmd (KG + archive via extraction)
2. Query taOSmd for the answer
3. Score using substring matching (same as the official eval)

Usage: .venv/bin/python benchmarks/longmemeval_runner.py [--limit N] [--type TYPE]
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd.knowledge_graph import TemporalKnowledgeGraph
from taosmd.archive import ArchiveStore
from taosmd.memory_extractor import extract_facts_from_text, process_conversation_turn
from taosmd.context_assembler import ContextAssembler
from taosmd.vector_memory import VectorMemory


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "longmemeval_oracle.json")

# Remote LLM for answer generation (override via TAOSMD_OLLAMA_URL env var)
REMOTE_LLM_URL = os.environ.get("TAOSMD_OLLAMA_URL", "http://localhost:11434")
REMOTE_LLM_MODEL = os.environ.get("TAOSMD_OLLAMA_MODEL", "qwen2.5:3b")
# E-012: the judge MUST be external to the generator (same-model judging
# inflates, and a same-model thinking judge emits CoT and never cleanly says
# CORRECT, which zeroed the qwen3.5:9b run). Defaults to the generator for
# back-compat; set TAOSMD_JUDGE_MODEL to a different, non-thinking model.
JUDGE_MODEL = os.environ.get("TAOSMD_JUDGE_MODEL", REMOTE_LLM_MODEL)
# Evidence pipeline depth. E-012 diagnosis: the generator was STARVED of
# evidence (assembler capped at 2000 tok, 5 vector chunks, 3 FTS hits), which
# craters multi-session/temporal questions that need several sessions in view.
# All tunable so we can sweep "get the scores up" configs.
CONTEXT_CHARS = int(os.environ.get("TAOSMD_CONTEXT_CHARS", "16000"))
ASSEMBLE_TOKENS = int(os.environ.get("TAOSMD_ASSEMBLE_TOKENS", "4000"))
RETRIEVE_LIMIT = int(os.environ.get("TAOSMD_RETRIEVE_LIMIT", "12"))
FTS_LIMIT = int(os.environ.get("TAOSMD_FTS_LIMIT", "5"))
# Intelligent context release (Jay's idea): retrieve MORE, then RERANK and keep
# only the clean top-K before generation, so the generator reasons over the
# relevant subset, not a noisy pile. Drops REDUNDANT context (the claims layer
# already drops INCORRECT). bge-v2-m3 cross-encoder, already in the stack.
RERANK = os.environ.get("TAOSMD_RERANK", "0") == "1"
RERANK_PATH = os.environ.get("TAOSMD_RERANK_PATH", "models/bge-reranker-v2-m3-onnx")
RERANK_TOP_K = int(os.environ.get("TAOSMD_RERANK_TOP_K", "8"))
# E-012 lever 2: query decomposition with iterative retrieval. Split the
# question into 2-3 sub-queries, retrieve for each, union deduplicated. Ported
# from locomo_runner._decompose_query. Default off.
DECOMPOSE = os.environ.get("TAOSMD_DECOMPOSE", "0") == "1"
DECOMPOSE_MODEL = os.environ.get("TAOSMD_DECOMPOSE_MODEL", "gemma4:e2b")
# E-012 lever 3: CoVe-style answer self-verification. After the first answer,
# one extra generator pass keeps the draft if it is fully supported by the
# context, else rewrites it from the context. Default off.
SELF_VERIFY = os.environ.get("TAOSMD_SELF_VERIFY", "0") == "1"
_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker once; None if disabled/unavailable."""
    global _reranker
    if _reranker is None and RERANK:
        from taosmd.cross_encoder import CrossEncoderReranker  # noqa: PLC0415
        _reranker = CrossEncoderReranker(onnx_path=RERANK_PATH)
    return _reranker

ANSWER_PROMPT = """Based on the following context from past conversations, answer the question.
If the answer is not in the context, say "I don't know."
Answer concisely in 1-2 sentences. /no_think

Context:
{context}

Question: {question}

Answer:"""


def score_answer_substring(predicted: str, gold: str) -> bool:
    """Score using substring matching (fast, lower bound)."""
    return gold.lower().strip() in predicted.lower()


async def score_answer_llm(client, predicted: str, gold: str, question: str) -> bool:
    """Score using LLM-as-judge (official LongMemEval approach)."""
    prompt = f"""You are a strict answer evaluator. Determine if the predicted answer contains the same factual information as the reference answer.

Rules:
- "I don't know" or similar non-answers are ALWAYS incorrect
- The predicted answer must contain the key facts from the reference answer
- Paraphrasing is fine, but the core information must match
- If the predicted answer is vague or generic while the reference is specific, that is INCORRECT

Reply with exactly one word: CORRECT or INCORRECT

Question: {question}
Reference answer: {gold}
Predicted answer: {predicted}

Verdict: /no_think"""
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 16},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            judgment = resp.json().get("message", {}).get("content", "").strip().upper()
            return "CORRECT" in judgment
    except Exception:
        pass
    return False


def score_answer(predicted: str, gold: str) -> bool:
    """Fast substring check (used when LLM judge not available)."""
    return score_answer_substring(predicted, gold)


async def llm_answer(client, context: str, question: str) -> str:
    """Use remote LLM to generate answer from recalled context."""
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": REMOTE_LLM_MODEL,
                "messages": [{"role": "user", "content": ANSWER_PROMPT.format(context=context[:CONTEXT_CHARS], question=question)}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 100},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "")
    except Exception:
        pass
    return ""


MULTIHOP_PROMPT = """Split this question into 2 or 3 shorter, focused sub-queries that together cover everything needed to answer it. Respond with one sub-query per line. No numbering, no explanation.

Question: {question}"""


async def decompose_query(client, question: str) -> list:
    """Split a question into 2-3 sub-queries via a small utility model.

    Returns a list of >=2 stripped lines, or [question] on failure or if fewer
    than two lines come back (so callers can iterate uniformly).
    """
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": DECOMPOSE_MODEL,
                "messages": [{"role": "user", "content": MULTIHOP_PROMPT.format(question=question) + " /no_think"}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 80},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            raw = resp.json().get("message", {}).get("content", "")
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            if len(lines) >= 2:
                return lines
    except Exception:
        pass
    return [question]


VERIFY_PROMPT = """You are checking a draft answer against the context.
If every part of the draft answer is supported by the context, repeat the draft answer exactly.
If any part is unsupported or contradicted by the context, write a corrected answer using ONLY the context.
Answer concisely in 1-2 sentences, no explanation. /no_think

Context:
{context}

Question: {question}

Draft answer: {answer}

Final answer:"""


async def self_verify_answer(client, context: str, question: str, answer: str) -> str:
    """One CoVe-style verification pass.

    Keeps the draft if it is supported by the context, otherwise returns a
    corrected answer. Falls back to the original draft on empty draft, empty
    revision, or any failure, so it can never make an answer worse by erroring.
    """
    if not answer:
        return answer
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": REMOTE_LLM_MODEL,
                "messages": [{"role": "user", "content": VERIFY_PROMPT.format(context=context[:CONTEXT_CHARS], question=question, answer=answer)}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 100},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            revised = resp.json().get("message", {}).get("content", "").strip()
            if revised:
                return revised
    except Exception:
        pass
    return answer


async def run_benchmark(limit: int = 50, question_type: str | None = None, use_llm: bool = False):
    print("=" * 70)
    print("LongMemEval Benchmark — taOSmd")
    print("=" * 70)

    # Load dataset
    with open(DATA_PATH) as f:
        dataset = json.load(f)

    if question_type:
        dataset = [q for q in dataset if q["question_type"] == question_type]
        print(f"Filtered to type: {question_type} ({len(dataset)} questions)")

    dataset = dataset[:limit]
    print(f"Running {len(dataset)} questions")

    results_by_type = {}
    total_correct = 0
    total_questions = 0
    total_time = 0

    # Create LLM client if needed
    llm_client = None
    if use_llm:
        import httpx as _httpx
        llm_client = _httpx.AsyncClient(timeout=30)

    for i, item in enumerate(dataset):
        qtype = item["question_type"]
        question = item["question"]
        gold_answer = item["answer"]
        sessions = item.get("haystack_sessions", [])

        # Create fresh KG + archive + vector memory per question (isolated test)
        tmp = tempfile.mkdtemp()
        kg = TemporalKnowledgeGraph(db_path=os.path.join(tmp, "kg.db"))
        archive = ArchiveStore(archive_dir=os.path.join(tmp, "archive"), index_path=os.path.join(tmp, "idx.db"))
        vmem = VectorMemory(
            db_path=os.path.join(tmp, "vectors.db"),
            embed_mode=os.environ.get("TAOSMD_EMBED_MODE", "onnx"),
            onnx_path=os.environ.get(
                "TAOSMD_ONNX_PATH",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx"),
            ),
        )
        await kg.init()
        await archive.init()

        import httpx as _httpx
        embed_client = _httpx.AsyncClient(timeout=15)
        await vmem.init(http_client=embed_client)

        # Ingest conversation sessions
        t0 = time.time()
        for si, session in enumerate(sessions):
            # Build session-level text blocks for embedding
            session_text = ""
            for turn in session:
                content = turn.get("content", "")
                role = turn.get("role", "user")
                if content:
                    # Extract facts into KG
                    await process_conversation_turn(
                        content, agent_name="assistant" if role == "assistant" else None,
                        kg=kg, archive=archive, source="longmemeval",
                    )
                    # Archive raw content
                    await archive.record(
                        "conversation",
                        {"role": role, "content": content},
                        summary=content[:80],
                    )
                    session_text += f"\n[{role}]: {content}"

            # Embed the full session as one block (better for multi-turn recall)
            if session_text:
                # Split into ~500 char chunks with overlap for embedding
                chunks = []
                words = session_text.split()
                chunk_size = 100  # words per chunk
                overlap = 20
                for start in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[start:start + chunk_size])
                    if chunk.strip():
                        chunks.append(chunk)
                for chunk in chunks:
                    await vmem.add(chunk, metadata={"session": si})

        ingest_time = time.time() - t0

        # Query taOSmd — both assembled context AND raw archive search
        assembler = ContextAssembler(kg=kg, archive=archive)
        t1 = time.time()
        ctx = await assembler.assemble(
            query=question,
            depth="auto",
            max_total_tokens=ASSEMBLE_TOKENS,
        )

        # Also do FTS search over raw archive (the MemPalace approach — verbatim recall)
        # Search for key words from the question AND the answer
        search_terms = question.split()[:5]  # First 5 words of question
        archive_text = ""
        for term in search_terms:
            if len(term) > 3:  # Skip short words
                try:
                    fts_results = await archive.search_fts(term, limit=FTS_LIMIT)
                    for r in fts_results:
                        archive_text += " " + r.get("data_json", "") + " " + r.get("summary", "")
                except Exception:
                    pass

        # Also do semantic vector search (the MemPalace approach). Retrieve a
        # wider pool, then rerank/prune to the clean top-K (intelligent context
        # release) so the generator is not buried in redundant chunks.
        if DECOMPOSE and llm_client is not None:
            sub_queries = await decompose_query(llm_client, question)
            seen_texts = set()
            vector_results = []
            for sq in sub_queries:
                for r in await vmem.search(sq, limit=RETRIEVE_LIMIT):
                    t = r.get("text", "")
                    if t and t not in seen_texts:
                        seen_texts.add(t)
                        vector_results.append(r)
            vector_results = vector_results[:RETRIEVE_LIMIT]
        else:
            vector_results = await vmem.search(question, limit=RETRIEVE_LIMIT)
        rr = _get_reranker()
        if rr is not None and getattr(rr, "available", False) and vector_results:
            vector_results = rr.rerank(question, vector_results, RERANK_TOP_K)
        vector_text = " ".join(r["text"] for r in vector_results)

        query_time = time.time() - t1

        # Score — combine ALL retrieval methods
        full_context = ctx["context"] + " " + archive_text + " " + vector_text

        if use_llm and llm_client is not None:
            t_llm = time.time()
            # Step 1: LLM generates answer from recalled context
            answer = await llm_answer(llm_client, full_context, question)
            # Step 2: LLM judges whether answer matches gold (official eval method)
            if answer and not any(idk in answer.lower() for idk in ("i don't know", "i do not know", "i'm sorry", "not in the context", "does not contain", "no information")):
                correct = await score_answer_llm(llm_client, answer, gold_answer, question)
            else:
                correct = False
            llm_time = time.time() - t_llm
            # Debug
            if i < 5:
                print(f"      [{llm_time:.1f}s] Answer: {(answer or 'EMPTY')[:80]} → {'✓' if correct else '✗'}")
        else:
            correct = score_answer_substring(full_context, gold_answer)

        total_questions += 1
        if correct:
            total_correct += 1

        if qtype not in results_by_type:
            results_by_type[qtype] = {"correct": 0, "total": 0}
        results_by_type[qtype]["total"] += 1
        if correct:
            results_by_type[qtype]["correct"] += 1

        elapsed = ingest_time + query_time
        total_time += elapsed

        status = "✓" if correct else "✗"
        print(f"  [{i+1:3d}/{len(dataset)}] {status} {qtype:25s} | ingest:{ingest_time:.1f}s query:{query_time:.3f}s | {question[:50]}")

        await archive.close()
        await kg.close()
        await vmem.close()
        await embed_client.aclose()

    # Results
    overall = total_correct / total_questions * 100 if total_questions > 0 else 0

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\n  Overall: {total_correct}/{total_questions} ({overall:.1f}%)")
    print(f"  Total time: {total_time:.1f}s ({total_time/total_questions:.1f}s per question)")

    print(f"\n  By question type:")
    for qtype, data in sorted(results_by_type.items()):
        pct = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"    {qtype:30s} {data['correct']:3d}/{data['total']:<3d} ({pct:.1f}%)")

    print(f"\n  Comparison:")
    print(f"    MemPalace (raw verbatim):     96.6%")
    print(f"    SuperMemory:                  81.6%")
    print(f"    GPT-4o (full context):        ~70%")
    print(f"    taOSmd (Pi NPU, no cloud):    {overall:.1f}%")
    print(f"{'='*70}")

    if llm_client:
        await llm_client.aclose()

    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of questions to run")
    parser.add_argument("--type", type=str, default=None, help="Filter by question type")
    parser.add_argument("--llm", action="store_true", help="Use remote LLM for answer generation")
    args = parser.parse_args()

    asyncio.run(run_benchmark(limit=args.limit, question_type=args.type, use_llm=args.llm))
