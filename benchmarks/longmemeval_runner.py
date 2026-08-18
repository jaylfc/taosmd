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
import inspect
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
from taosmd.retrieval import retrieve as _retrieve
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
# Representative sampling. The oracle set is ordered by question type, so a head
# slice dataset[:limit] is single-type (the first ~133 questions are all
# temporal). Set TAOSMD_SAMPLE_SEED to shuffle deterministically before slicing
# so a screen at limit<500 covers all six types. Unset = head slice (back-compat).
SAMPLE_SEED = os.environ.get("TAOSMD_SAMPLE_SEED")
# Ollama context window for generation. 0 (the default) means "do not send
# num_ctx", which leaves Ollama on its own default and keeps every historical
# LongMemEval number reproducible byte for byte. It is recorded in the result
# JSON so a run's window is never guessed after the fact. Note that a 16000-char
# context is roughly 4500-5000 tokens, so Ollama's 4096 default already
# truncates some prompts; raising this changes what is measured, which is why it
# is opt-in rather than silently bumped here.
NUM_CTX = int(os.environ.get("TAOSMD_LME_NUM_CTX", "0"))
_reranker = None


def _gen_options(**extra):
    """Ollama options dict, carrying num_ctx only when it was explicitly set."""
    opts = dict(extra)
    if NUM_CTX:
        opts["num_ctx"] = NUM_CTX
    return opts


def sample_dataset(dataset, limit, seed=None):
    """Return `limit` items from the dataset.

    With a seed, shuffle representatively first (the oracle set is type-ordered,
    so a head slice would be a single question type); without a seed, take the
    head slice for back-compat.
    """
    if seed is not None:
        import random  # noqa: PLC0415
        ds = list(dataset)
        random.Random(int(seed)).shuffle(ds)
        return ds[:limit]
    return dataset[:limit]


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


def _parse_verdict(content: str) -> bool:
    """Parse a judge's one-word CORRECT / INCORRECT verdict into a pass/fail.

    The negative verdict is checked FIRST on purpose: the string "INCORRECT"
    contains the substring "CORRECT", so a naive ``"CORRECT" in judgment`` scores
    every INCORRECT reply as a pass. An empty or unrecognised reply is treated as
    incorrect (fail-closed).
    """
    j = (content or "").strip().upper()
    if "INCORRECT" in j:
        return False
    return "CORRECT" in j


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
            content = resp.json().get("message", {}).get("content", "")
            return _parse_verdict(content)
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
                "options": _gen_options(temperature=0, num_predict=100),
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
                "options": _gen_options(temperature=0, num_predict=100),
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


def retrieve_supports_graph_expansion() -> bool:
    """True when the installed ``taosmd.retrieval.retrieve`` accepts graph_expansion.

    The control is added by the bi-temporal fact-readback work (PR #191). On a
    checkout without it the runner still works, but the E-030 arms cannot be
    distinguished, so ``--graph-expansion N>0`` is rejected up front rather
    than silently producing two identical arms (the exact failure mode this
    wiring exists to remove).
    """
    return "graph_expansion" in inspect.signature(_retrieve).parameters


async def retrieve_vector_hits(
    question: str,
    kg,
    vmem,
    graph_expansion: int = 0,
    reranker=None,
    limit: int | None = None,
) -> list[dict]:
    """Fetch the vector-layer hits through the real retrieval path.

    Goes through ``taosmd.retrieval.retrieve`` rather than calling
    ``vmem.search`` directly, so runtime retrieval controls -- notably
    ``graph_expansion`` -- actually take effect on this benchmark.

    Anchor preservation: the call is deliberately parameterised to reproduce
    the hand-rolled vector stage it replaces.

    * ``strategy="custom"`` with ``memory_layers=["vector"]`` queries the
      vector source only. ``kg`` is still passed in ``sources`` because
      ``_append_graph_expansion`` reads ``sources["kg"]`` regardless of which
      layers were searched, so the fact-readback block can fire without the KG
      competing for the ``limit`` slots.
    * ``candidate_top_k=RETRIEVE_LIMIT`` pins the vector fetch to exactly the
      ``limit`` the old code used (retrieve() would otherwise fetch limit * 3).
    * ``limit`` is ``RERANK_TOP_K`` when reranking is on and ``RETRIEVE_LIMIT``
      otherwise, matching the old narrow-after-rerank behaviour.
    * ``verify`` is left at its default False: the old stage never gated on
      claims, so turning verification on here would quietly change what the
      benchmark measures.

    The one residual difference from the old code is retrieve()'s near-duplicate
    filter (Jaccard >= 0.8 over the hit texts) where the old path deduplicated
    on exact string equality only. LongMemEval chunks are ~100-word windows with
    a 20-word overlap, so unlike EventQA this filter CAN fire; that is exactly
    what ``--report-retrieval-delta`` measures rather than assumes.

    Returns the raw normalised hit dicts (each with a ``text`` field). When
    ``graph_expansion`` > 0 a trailing derived block (``source ==
    "kg_expansion"``) is present.
    """
    if limit is None:
        limit = RERANK_TOP_K if reranker is not None else RETRIEVE_LIMIT

    kwargs = dict(
        strategy="custom",
        memory_layers=["vector"],
        sources={"vector": vmem, "kg": kg},
        limit=limit,
        candidate_top_k=RETRIEVE_LIMIT,
        reranker=reranker,
        agent_name="longmemeval_eval",
    )
    if graph_expansion:
        kwargs["graph_expansion"] = graph_expansion
    return await _retrieve(question, **kwargs)


async def retrieve_vector_results(
    question: str,
    kg,
    vmem,
    llm_client=None,
    graph_expansion: int = 0,
    retrieval_path: str = "retrieve",
) -> list[dict]:
    """The runner's vector stage, on either the wired or the legacy path.

    Reproduces the pre-wiring shape in both modes: optional query decomposition
    with a deduplicated union capped at ``RETRIEVE_LIMIT``, then an optional
    cross-encoder rerank down to ``RERANK_TOP_K``.
    """
    rr = _get_reranker()
    reranker = rr if (rr is not None and getattr(rr, "available", False)) else None
    decomposing = DECOMPOSE and llm_client is not None

    if graph_expansion and retrieval_path == "legacy":
        raise ValueError(
            "graph_expansion requires --retrieval-path retrieve; the legacy "
            "path assembles context by hand and cannot honour the control."
        )
    if graph_expansion and decomposing:
        raise ValueError(
            "graph_expansion cannot be combined with TAOSMD_DECOMPOSE=1: the "
            "sub-query union is truncated to RETRIEVE_LIMIT, which can discard "
            "the derived fact-readback block and make the arm unmeasurable."
        )

    async def _search(query: str, want_rerank: bool) -> list[dict]:
        """One query's hits. want_rerank folds the rerank into retrieve()."""
        if retrieval_path == "legacy":
            return await vmem.search(query, limit=RETRIEVE_LIMIT)
        return await retrieve_vector_hits(
            question=query,
            kg=kg,
            vmem=vmem,
            graph_expansion=graph_expansion,
            reranker=reranker if want_rerank else None,
            limit=None if want_rerank else RETRIEVE_LIMIT,
        )

    if decomposing:
        # The union is reranked once at the end, so the per-sub-query fetch must
        # NOT rerank; otherwise the union would be assembled from pre-narrowed
        # pools and would not match the pre-wiring behaviour.
        sub_queries = await decompose_query(llm_client, question)
        seen_texts = set()
        vector_results = []
        for sq in sub_queries:
            for r in await _search(sq, want_rerank=False):
                t = r.get("text", "")
                if t and t not in seen_texts:
                    seen_texts.add(t)
                    vector_results.append(r)
        vector_results = vector_results[:RETRIEVE_LIMIT]
        if reranker is not None and vector_results:
            vector_results = reranker.rerank(question, vector_results, RERANK_TOP_K)
        return vector_results

    vector_results = await _search(question, want_rerank=retrieval_path != "legacy")
    if retrieval_path == "legacy" and reranker is not None and vector_results:
        vector_results = reranker.rerank(question, vector_results, RERANK_TOP_K)
    return vector_results


async def retrieve_context(
    question: str,
    kg,
    archive,
    vmem,
    llm_client=None,
    graph_expansion: int = 0,
    retrieval_path: str = "retrieve",
) -> str:
    """Assemble the generator context for one question.

    Combines the three retrieval methods the runner has always combined:
    ContextAssembler, archive FTS over the question's leading terms, and the
    vector stage. Only the vector stage moved onto ``retrieve()``; the other
    two are untouched.

    Args:
        graph_expansion: Token budget for the bi-temporal fact-readback block
            (0 = off). Requires ``retrieval_path="retrieve"``; the manual
            legacy path cannot honour it, which is what blocked E-030.
        retrieval_path: ``"retrieve"`` (default) routes the vector stage
            through ``taosmd.retrieval.retrieve`` so runtime controls apply.
            ``"legacy"`` reproduces the pre-wiring hand-rolled stage so the
            published anchors can be re-measured side by side.
    """
    assembler = ContextAssembler(kg=kg, archive=archive)
    ctx = await assembler.assemble(
        query=question,
        depth="auto",
        max_total_tokens=ASSEMBLE_TOKENS,
    )

    # FTS search over raw archive (the MemPalace approach -- verbatim recall).
    # Search for key words from the question.
    archive_text = ""
    for term in question.split()[:5]:
        if len(term) > 3:  # Skip short words
            try:
                fts_results = await archive.search_fts(term, limit=FTS_LIMIT)
                for r in fts_results:
                    archive_text += " " + r.get("data_json", "") + " " + r.get("summary", "")
            except Exception:
                pass

    vector_results = await retrieve_vector_results(
        question, kg, vmem,
        llm_client=llm_client,
        graph_expansion=graph_expansion,
        retrieval_path=retrieval_path,
    )
    vector_text = " ".join(r["text"] for r in vector_results if r.get("text"))

    return ctx["context"] + " " + archive_text + " " + vector_text


def summarize_retrieval_delta(results: list[dict]) -> dict | None:
    """Summarise the wired-vs-legacy context comparison, or None if not measured.

    Only populated when the runner is given ``--report-retrieval-delta``. This
    is the evidence for whether routing through retrieve() preserved the
    published LongMemEval anchors: 100% byte-identical means they carry over
    untouched, and anything less quantifies exactly how much has to be
    re-anchored before the arms can be trusted.
    """
    deltas = [r["retrieval_delta"] for r in results if r.get("retrieval_delta")]
    if not deltas:
        return None
    n = len(deltas)
    identical = sum(1 for d in deltas if d["identical"])
    return {
        "n": n,
        "identical": identical,
        "identical_pct": round(identical / n * 100, 2),
        "mean_legacy_chars": round(sum(d["legacy_chars"] for d in deltas) / n, 1),
        "mean_wired_chars": round(sum(d["wired_chars"] for d in deltas) / n, 1),
    }


def load_dataset() -> list:
    """Load the LongMemEval oracle set from disk."""
    with open(DATA_PATH) as f:
        return json.load(f)


async def run_benchmark(
    limit: int = 50,
    question_type: str | None = None,
    use_llm: bool = False,
    args: argparse.Namespace | None = None,
):
    graph_expansion = 0
    retrieval_path = "retrieve"
    report_retrieval_delta = False
    out_path = ""
    if args is not None:
        limit = args.limit
        question_type = args.type
        use_llm = args.llm
        graph_expansion = args.graph_expansion
        retrieval_path = args.retrieval_path
        report_retrieval_delta = args.report_retrieval_delta
        out_path = args.out

    print("=" * 70)
    print("LongMemEval Benchmark — taOSmd")
    print("=" * 70)
    print(
        f"  retrieval_path={retrieval_path}  graph_expansion={graph_expansion}  "
        f"num_ctx={NUM_CTX or 'ollama-default'}"
    )

    # Refuse rather than silently run two identical arms. Exits non-zero on
    # purpose so an automated chain checking $? reads a refusal as a failure.
    if graph_expansion and not retrieve_supports_graph_expansion():
        print(
            "  ERROR: the installed taosmd.retrieval.retrieve() has no "
            "graph_expansion parameter, so both E-030 arms would be identical. "
            "Run this on a checkout that includes the bi-temporal fact-readback "
            "work (PR #191).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load dataset
    dataset = load_dataset()

    if question_type:
        dataset = [q for q in dataset if q["question_type"] == question_type]
        print(f"Filtered to type: {question_type} ({len(dataset)} questions)")

    dataset = sample_dataset(dataset, limit, SAMPLE_SEED)
    print(f"Running {len(dataset)} questions" + (f" (sampled, seed={SAMPLE_SEED})" if SAMPLE_SEED else ""))

    results_by_type = {}
    all_results: list[dict] = []
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

        # Query taOSmd: assembled context + raw archive FTS + the vector stage,
        # the last of which now runs through taosmd.retrieval.retrieve().
        t1 = time.time()
        full_context = await retrieve_context(
            question, kg, archive, vmem,
            llm_client=llm_client,
            graph_expansion=graph_expansion,
            retrieval_path=retrieval_path,
        )
        query_time = time.time() - t1

        # Anchor evidence: with the flag on, also build the pre-wiring context
        # and record how far the two diverge, so "does the default path still
        # reproduce the published numbers?" is answered with a measurement.
        delta = None
        if report_retrieval_delta:
            legacy_ctx = await retrieve_context(
                question, kg, archive, vmem,
                llm_client=llm_client,
                retrieval_path="legacy",
            )
            delta = {
                "legacy_chars": len(legacy_ctx),
                "wired_chars": len(full_context),
                "identical": legacy_ctx == full_context,
            }
            print(
                f"    [DELTA] q={i} legacy={delta['legacy_chars']} "
                f"wired={delta['wired_chars']} identical={delta['identical']}"
            )

        if use_llm and llm_client is not None:
            t_llm = time.time()
            # Step 1: LLM generates answer from recalled context
            answer = await llm_answer(llm_client, full_context, question)
            if SELF_VERIFY:
                answer = await self_verify_answer(llm_client, full_context, question, answer)
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
        all_results.append({
            "idx": i,
            "question_type": qtype,
            "question": question,
            "correct": bool(correct),
            "retrieved_chars": len(full_context),
            "retrieval_delta": delta,
        })

        status = "✓" if correct else "✗"
        print(f"  [{i+1:3d}/{len(dataset)}] {status} {qtype:25s} | ingest:{ingest_time:.1f}s query:{query_time:.3f}s | {question[:50]}")

        await archive.close()
        await kg.close()
        await vmem.close()
        await embed_client.aclose()

    # Results
    overall = total_correct / total_questions * 100 if total_questions > 0 else 0
    per_q = total_time / total_questions if total_questions > 0 else 0.0

    delta_summary = summarize_retrieval_delta(all_results)
    if delta_summary:
        print(f"\n{'='*70}")
        print("RETRIEVAL DELTA (wired retrieve() path vs pre-wiring legacy path)")
        print(f"  questions compared:      {delta_summary['n']}")
        print(f"  byte-identical contexts: {delta_summary['identical']}"
              f" ({delta_summary['identical_pct']:.1f}%)")
        print(f"  mean chars legacy/wired: {delta_summary['mean_legacy_chars']:.0f}"
              f" / {delta_summary['mean_wired_chars']:.0f}")

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\n  Overall: {total_correct}/{total_questions} ({overall:.1f}%)")
    print(f"  Total time: {total_time:.1f}s ({per_q:.1f}s per question)")

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

    if total_questions:
        if not out_path:
            out_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"longmemeval_{int(time.time())}.json")
        result_doc = {
            "question_type": question_type,
            "limit": limit,
            "generator": REMOTE_LLM_MODEL,
            "judge": JUDGE_MODEL,
            "rerank": RERANK,
            "decompose": DECOMPOSE,
            "self_verify": SELF_VERIFY,
            "assemble_tokens": ASSEMBLE_TOKENS,
            "retrieve_limit": RETRIEVE_LIMIT,
            "fts_limit": FTS_LIMIT,
            "context_chars": CONTEXT_CHARS,
            "num_ctx": NUM_CTX,
            "retrieval_path": retrieval_path,
            "graph_expansion": graph_expansion,
            "retrieval_delta": delta_summary,
            "metrics": {
                "n": total_questions,
                "correct": total_correct,
                "accuracy": overall,
                "by_type": results_by_type,
            },
            "results": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(result_doc, f, indent=2)
        print(f"  results -> {out_path}")

    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate taOSmd on LongMemEval")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions to run")
    parser.add_argument("--type", type=str, default=None, help="Filter by question type")
    parser.add_argument("--llm", action="store_true", help="Use remote LLM for answer generation")
    parser.add_argument(
        "--graph-expansion",
        type=int,
        default=int(os.environ.get("TAOSMD_GRAPH_EXPANSION", "0")),
        metavar="N",
        help="Bi-temporal fact-readback token budget (0 = off, the default). "
             "This is the E-030 lever; needs --retrieval-path retrieve.",
    )
    parser.add_argument(
        "--retrieval-path",
        default="retrieve",
        choices=("retrieve", "legacy"),
        help="'retrieve' (default) routes the vector stage through "
             "taosmd.retrieval.retrieve() so runtime controls apply. 'legacy' "
             "reproduces the pre-wiring hand-rolled stage, for re-anchoring only.",
    )
    parser.add_argument(
        "--report-retrieval-delta",
        action="store_true",
        help="Also build the legacy context per question and report how far the "
             "wired path diverges (anchor evidence). Doubles retrieval cost.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Result JSON path (default: benchmarks/results/longmemeval_<ts>.json)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(args=args))


if __name__ == "__main__":
    main()
