#!/usr/bin/env python3
"""EventQA benchmark runner for taOSmd (MemoryAgentBench, Accurate_Retrieval split).

Evaluates the taOSmd retrieval-and-generation stack on MemoryAgentBench EventQA,
reporting Accuracy (substring exact-match) so that TAOSMD_SELF_VERIFY on vs off
can be compared as the primary lever under test.

Dataset: ai-hyz/MemoryAgentBench, split "Accurate_Retrieval".
Filter: rows where str(row["metadata"]["source"]) contains "eventqa".
Tiers:  eventqa_65536 (default), eventqa_131072, eventqa_full.

Metric: substring exact-match per the MemoryAgentBench EventQA methodology
(SQuAD/DrQA normalize_answer + substring containment), averaged over all
questions -> Accuracy %.

Usage:
    # dry wiring check (no LLM, no GPU):
    python3 benchmarks/eventqa_runner.py --tier eventqa_65536 --contexts 3

    # full run with generation (needs a live Ollama / GPU):
    TAOSMD_OLLAMA_MODEL=qwen2.5:3b TAOSMD_ONNX_PATH=models/arctic-embed-s-onnx \\
    python3 benchmarks/eventqa_runner.py --tier eventqa_65536 --contexts 50 --llm

Retrieval path:
    The vector stage goes through taosmd.retrieval.retrieve() so that runtime
    retrieval controls take effect here. --graph-expansion N selects the
    bi-temporal fact-readback lever (E-025); 0 is off and is the default.
    --retrieval-path legacy restores the pre-wiring hand-rolled stage, and
    --report-retrieval-delta measures how far the two diverge per question.

Env levers (mirror beam_runner):
    TAOSMD_OLLAMA_URL         Ollama base URL (default http://localhost:11434)
    TAOSMD_OLLAMA_MODEL       Generator model (default qwen2.5:3b)
    TAOSMD_ONNX_PATH          ONNX embedder path
    TAOSMD_ONNX_POOLING       Embedding pooling strategy
    TAOSMD_RETRIEVE_LIMIT     Vector search top-K (default 12)
    TAOSMD_FTS_LIMIT          FTS snippets per keyword (default 5)
    TAOSMD_RERANK             "1" to enable cross-encoder rerank
    TAOSMD_RERANK_PATH        Cross-encoder model path
    TAOSMD_RERANK_TOP_K       Post-rerank candidate count (default 8)
    TAOSMD_ASSEMBLE_TOKENS    ContextAssembler token budget (default 4000)
    TAOSMD_CONTEXT_CHARS      Max context chars passed to generator (default 16000)
    TAOSMD_CHUNK_WORDS        Chunk size in words (default 512)
    TAOSMD_SELF_VERIFY        "1" to enable self-verification pass (the lever)
    TAOSMD_EVENTQA_NUM_CTX    Ollama context window for generation (default 8192;
                              the default 4096 truncates EventQA prompts)
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import re
import string
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd.knowledge_graph import TemporalKnowledgeGraph
from taosmd.archive import ArchiveStore
from taosmd.memory_extractor import process_conversation_turn
from taosmd.context_assembler import ContextAssembler
from taosmd.retrieval import retrieve as _retrieve
from taosmd.vector_memory import VectorMemory

# ── env levers (mirror beam_runner) ──────────────────────────────────────────
REMOTE_LLM_URL = os.environ.get("TAOSMD_OLLAMA_URL", "http://localhost:11434")
REMOTE_LLM_MODEL = os.environ.get("TAOSMD_OLLAMA_MODEL", "qwen2.5:3b")
CONTEXT_CHARS = int(os.environ.get("TAOSMD_CONTEXT_CHARS", "16000"))
ASSEMBLE_TOKENS = int(os.environ.get("TAOSMD_ASSEMBLE_TOKENS", "4000"))
RETRIEVE_LIMIT = int(os.environ.get("TAOSMD_RETRIEVE_LIMIT", "12"))
FTS_LIMIT = int(os.environ.get("TAOSMD_FTS_LIMIT", "5"))
RERANK = os.environ.get("TAOSMD_RERANK", "0") == "1"
RERANK_PATH = os.environ.get("TAOSMD_RERANK_PATH", "models/bge-reranker-v2-m3-onnx")
RERANK_TOP_K = int(os.environ.get("TAOSMD_RERANK_TOP_K", "8"))
SELF_VERIFY = os.environ.get("TAOSMD_SELF_VERIFY", "0") == "1"

# Chunking: EventQA contexts are long novel excerpts (~65K-full tokens).
# Default is 512 words per chunk (larger than BEAM's 100-word default, since
# novel narrative benefits from wider windows for coherence retrieval). The
# caller can sweep TAOSMD_CHUNK_WORDS to find the optimal window.
CHUNK_WORDS = int(os.environ.get("TAOSMD_CHUNK_WORDS", "3000"))
# MemoryAgentBench EventQA ingests at chunk_size=4096 tiktoken tokens; match that
# protocol exactly when tiktoken is available so retrieval sees the same chunks.
CHUNK_TOKENS = int(os.environ.get("TAOSMD_EVENTQA_CHUNK_TOKENS", "4096"))
# Ollama context window for generation. The default num_ctx (4096) is too small
# for EventQA prompts: a 16000-char retrieved context of dense novel prose is
# about 4500-5000 tokens, which overflows 4096 and makes Ollama truncate the
# prompt, so the model emits a 1-2 token stub instead of a real answer. 8192
# holds the full prompt plus the answer with headroom; raise it for longer tiers.
NUM_CTX = int(os.environ.get("TAOSMD_EVENTQA_NUM_CTX", "8192"))

ONNX_PATH = os.environ.get(
    "TAOSMD_ONNX_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx"),
)

HF_DATASET = "ai-hyz/MemoryAgentBench"
HF_SPLIT = "Accurate_Retrieval"
VALID_TIERS = ("eventqa_65536", "eventqa_131072", "eventqa_full")

_reranker = None


# ── scoring (MemoryAgentBench EventQA metric, published methodology) ──────────


def normalize_answer(s: str) -> str:
    """Normalize a string: lowercase, strip punctuation, strip articles, collapse whitespace.

    Replicates MemoryAgentBench EventQA normalize_answer (utils/eval_other_utils.py)
    EXACTLY, including the order: lowercase, then remove punctuation, then remove
    articles, then collapse whitespace. The order matters for edge cases.
    """
    text = s.lower()
    # Remove all punctuation characters (BEFORE articles, matching their order).
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles: a, an, the (whole-word match).
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace and strip.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def substring_exact_match(pred: str, gold_list: list[str]) -> int:
    """Return 1 if any normalized gold string is a non-empty substring of normalized pred.

    Guards:
    - An empty or whitespace prediction scores 0 (not retrievable signal).
    - An empty gold string cannot match (prevents trivial 1 from bad data).
    """
    norm_pred = normalize_answer(pred)
    if not norm_pred:
        return 0
    for g in gold_list:
        norm_g = normalize_answer(g)
        if norm_g and norm_g in norm_pred:
            return 1
    return 0


# ── dataset loading ───────────────────────────────────────────────────────────


def load_eventqa_rows(tier: str, max_contexts: int, logger=print) -> list[dict]:
    """Load EventQA rows from MemoryAgentBench.

    Uses datasets.load_dataset at runtime (no download in this file; the real
    run on the bench host verifies the schema). Filters to rows whose
    metadata["source"] contains the requested tier string. Caps at max_contexts.

    Row schema (trust; do not try to download in this environment):
        context:   str  -- long novel excerpt
        questions: list[str]  -- ~100 questions per row
        answers:   list[list[str]]  -- parallel; each gold is a single-element list
        metadata:  dict  -- includes "source" key identifying the tier
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for EventQA loading. "
            "Install with: pip install datasets"
        ) from exc

    logger(f"  loading {HF_DATASET} split={HF_SPLIT} tier={tier} ...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)

    rows: list[dict] = []
    for item in ds:
        source = str((item.get("metadata") or {}).get("source", ""))
        if tier not in source:
            continue
        rows.append(dict(item))
        if len(rows) >= max_contexts:
            break

    logger(f"  loaded {len(rows)} row(s) matching tier={tier!r}")
    return rows


def parse_eventqa_row(row: dict) -> tuple[list[str], list[list[str]], str]:
    """Extract questions, answers, and source from an EventQA row.

    Returns:
        questions: list[str]
        answers:   list[list[str]]  -- each element is a list of gold strings
        source:    str
    """
    questions: list[str] = list(row.get("questions") or [])
    raw_answers = row.get("answers") or []
    answers: list[list[str]] = []
    for a in raw_answers:
        if isinstance(a, list):
            answers.append([str(x) for x in a])
        elif a is not None:
            answers.append([str(a)])
        else:
            answers.append([])
    source = str((row.get("metadata") or {}).get("source", ""))
    return questions, answers, source


# ── chunking ──────────────────────────────────────────────────────────────────

# Tiktoken is attempted first for accurate token-based splitting. Falls back
# to word-based chunking when not installed (same behaviour as beam_runner).
# The chunk size is controlled by TAOSMD_CHUNK_WORDS (word mode) or the
# equivalent number of tokens (tiktoken mode). We use CHUNK_WORDS for the
# word-count target in both paths to keep env-var semantics identical.

def _chunk_by_words(text: str, words: int) -> list[str]:
    """Split text into non-overlapping windows of up to `words` words each."""
    toks = text.split()
    if not toks:
        return []
    chunks: list[str] = []
    for start in range(0, len(toks), words):
        chunk = " ".join(toks[start : start + words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_context(text: str) -> list[str]:
    """Chunk an EventQA context into embeddable segments.

    Attempts tiktoken cl100k_base for token-accurate splitting; falls back to
    word-based chunks. The target size is CHUNK_WORDS tokens/words per chunk
    (default 512). No overlap -- EventQA passages are sequential novel excerpts
    and the question already contains full context, so overlap adds little and
    doubles ingest cost on long contexts.
    """
    if not text or not text.strip():
        return []

    try:
        import tiktoken  # noqa: PLC0415
        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(text)
        target = CHUNK_TOKENS  # match MemoryAgentBench EventQA chunk_size (4096 tiktoken tokens)
        chunks: list[str] = []
        for start in range(0, len(token_ids), target):
            segment_ids = token_ids[start : start + target]
            segment = enc.decode(segment_ids).strip()
            if segment:
                chunks.append(segment)
        return chunks
    except ImportError:
        pass

    return _chunk_by_words(text, CHUNK_WORDS)


# ── prompt construction ───────────────────────────────────────────────────────

# The question text already contains the events-so-far, a list of 6 candidate
# next-events, and the instruction to output only the answer. We add a terse
# framing that anchors the model to the retrieved context and reinforces the
# output format (one of the listed option strings verbatim, no preamble).
_ANSWER_PROMPT = """\
You are answering a multiple-choice question about a story. Use ONLY the provided context.
Output ONLY the exact text of the correct option -- no preamble, no punctuation change, no paraphrase.

Context:
{context}

{question}

Answer (copy one option exactly):"""

_VERIFY_PROMPT = """\
You are verifying a multiple-choice answer against the story context.
If the answer is one of the listed options and supported by the context, output it exactly.
If it is wrong or not one of the listed options, output the correct option exactly instead.
Do NOT add preamble, punctuation, or explanation. /no_think

Context:
{context}

{question}

Draft answer: {answer}

Final answer (exact option text):"""


def build_answer_prompt(context: str, question: str) -> str:
    return _ANSWER_PROMPT.format(context=context[:CONTEXT_CHARS], question=question)


def build_verify_prompt(context: str, question: str, answer: str) -> str:
    return _VERIFY_PROMPT.format(
        context=context[:CONTEXT_CHARS], question=question, answer=answer
    )


# ── LLM calls ─────────────────────────────────────────────────────────────────


async def llm_answer(client, context: str, question: str) -> str:
    """Generate an answer via Ollama. Returns empty string on failure."""
    prompt = build_answer_prompt(context, question)
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/generate",
            json={
                "model": REMOTE_LLM_MODEL,
                "prompt": "/no_think\n\n" + prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 64, "num_ctx": NUM_CTX},
            },
        )
        if resp.status_code == 200:
            return (resp.json().get("response") or "").strip()
    except Exception:
        pass
    return ""


async def self_verify_answer(client, context: str, question: str, answer: str) -> str:
    """Self-verification pass (TAOSMD_SELF_VERIFY lever). Never worsens on error.

    Mirrors beam_runner.self_verify_answer: one extra LLM call that checks the
    draft answer against context and returns a corrected answer if needed. Falls
    back to the original draft on any error or empty response.
    """
    if not answer:
        return answer
    prompt = build_verify_prompt(context, question, answer)
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/generate",
            json={
                "model": REMOTE_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 64, "num_ctx": NUM_CTX},
            },
        )
        if resp.status_code == 200:
            revised = (resp.json().get("response") or "").strip()
            if revised:
                return revised
    except Exception:
        pass
    return answer


# ── taOSmd retrieval ──────────────────────────────────────────────────────────


def _get_reranker():
    global _reranker
    if _reranker is None and RERANK:
        from taosmd.cross_encoder import CrossEncoderReranker  # noqa: PLC0415
        _reranker = CrossEncoderReranker(onnx_path=RERANK_PATH)
    return _reranker


def retrieve_supports_graph_expansion() -> bool:
    """True when the installed ``taosmd.retrieval.retrieve`` accepts graph_expansion.

    The control is added by the bi-temporal fact-readback work (PR #191). On a
    checkout without it the runner still works, but the E-025 arms cannot be
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

    The one residual difference from the old code is retrieve()'s near-duplicate
    filter (Jaccard >= 0.8 over the hit texts) where the old path deduplicated
    on exact string equality only. EventQA chunks are ~4096-token distinct novel
    passages, so this is not expected to fire; ``--report-retrieval-delta``
    measures it directly rather than assuming.

    Returns the raw normalised hit dicts (each with a ``text`` field). When
    ``graph_expansion`` > 0 a trailing derived block (``source ==
    "kg_expansion"``) is present.
    """
    rr = _get_reranker()
    reranker = rr if (rr is not None and getattr(rr, "available", False)) else None
    limit = RERANK_TOP_K if reranker is not None else RETRIEVE_LIMIT

    kwargs = dict(
        strategy="custom",
        memory_layers=["vector"],
        sources={"vector": vmem, "kg": kg},
        limit=limit,
        candidate_top_k=RETRIEVE_LIMIT,
        reranker=reranker,
        agent_name="eventqa_eval",
    )
    if graph_expansion:
        kwargs["graph_expansion"] = graph_expansion
    return await _retrieve(question, **kwargs)


async def retrieve_context(
    question: str,
    kg,
    archive,
    vmem,
    graph_expansion: int = 0,
    retrieval_path: str = "retrieve",
) -> str:
    """Retrieve relevant context via taOSmd.

    Returns a single assembled context string to pass to the generator.
    Deduplicates across assembler / FTS / vector sources.

    Args:
        graph_expansion: Token budget for the bi-temporal fact-readback block
            (0 = off). Requires ``retrieval_path="retrieve"``; the manual
            legacy path cannot honour it, which is what blocked E-025.
        retrieval_path: ``"retrieve"`` (default) routes the vector stage
            through ``taosmd.retrieval.retrieve`` so runtime controls apply.
            ``"legacy"`` reproduces the pre-wiring hand-rolled stage byte for
            byte and is kept only so the E-016 anchor can be re-measured
            side by side.
    """
    memories: list[str] = []

    assembler = ContextAssembler(kg=kg, archive=archive)
    ctx = await assembler.assemble(query=question, depth="auto", max_total_tokens=ASSEMBLE_TOKENS)
    if ctx.get("context"):
        memories.append(ctx["context"])

    for term in [w for w in question.split()[:5] if len(w) > 3]:
        try:
            for r in await archive.search_fts(term, limit=FTS_LIMIT):
                snippet = (r.get("data_json", "") + " " + r.get("summary", "")).strip()
                if snippet:
                    memories.append(snippet)
        except Exception:
            pass

    if retrieval_path == "legacy":
        if graph_expansion:
            raise ValueError(
                "graph_expansion requires --retrieval-path retrieve; the legacy "
                "path assembles context by hand and cannot honour the control."
            )
        vector_results = await vmem.search(question, limit=RETRIEVE_LIMIT)
        rr = _get_reranker()
        if rr is not None and getattr(rr, "available", False) and vector_results:
            vector_results = rr.rerank(question, vector_results, RERANK_TOP_K)
    else:
        vector_results = await retrieve_vector_hits(
            question, kg, vmem, graph_expansion=graph_expansion
        )
    memories.extend(r["text"] for r in vector_results if r.get("text"))

    seen: set[str] = set()
    deduped: list[str] = []
    for m in memories:
        if m and m not in seen:
            seen.add(m)
            deduped.append(m)

    return "\n\n".join(deduped)


async def ingest_context(context_text: str, kg, archive, vmem) -> int:
    """Ingest an EventQA context into a fresh taOSmd store.

    Chunks the context with chunk_context() and embeds each chunk into vmem.
    Also records the full text in the archive for FTS access. Returns the
    number of vector chunks embedded.
    """
    # Archive the full context for FTS retrieval.
    await archive.record("context", {"text": context_text[:4096]}, summary=context_text[:80])

    # Process as a single conversation turn for KG / memory extraction.
    await process_conversation_turn(
        context_text[:8192],
        agent_name=None,
        kg=kg,
        archive=archive,
        source="eventqa",
    )

    # Chunk and embed.
    chunks = chunk_context(context_text)
    for chunk in chunks:
        await vmem.add(chunk, metadata={})
    return len(chunks)


# ── per-row driver ────────────────────────────────────────────────────────────


async def process_row(
    row: dict,
    row_idx: int,
    use_llm: bool,
    llm_client,
    question_limit: int = 0,
    logger=print,
    graph_expansion: int = 0,
    retrieval_path: str = "retrieve",
    report_retrieval_delta: bool = False,
) -> list[dict]:
    """Ingest one EventQA row, then retrieve+generate+score each question.

    Creates a FRESH archive / kg / vmem per row (mirrors beam_runner
    process_conversation -- no cross-row contamination).

    In dry mode (use_llm=False) skips generation; records retrieved context
    length and scores 0 so the wiring can be verified GPU-free.
    """
    questions, answers, source = parse_eventqa_row(row)
    if question_limit > 0:
        questions = questions[:question_limit]
        answers = answers[:question_limit]

    tmp = tempfile.mkdtemp()
    kg = TemporalKnowledgeGraph(db_path=os.path.join(tmp, "kg.db"))
    archive = ArchiveStore(
        archive_dir=os.path.join(tmp, "archive"),
        index_path=os.path.join(tmp, "idx.db"),
    )
    vmem = VectorMemory(
        db_path=os.path.join(tmp, "vectors.db"),
        embed_mode=os.environ.get("TAOSMD_EMBED_MODE", "onnx"),
        onnx_path=ONNX_PATH,
    )
    await kg.init()
    await archive.init()

    import httpx as _httpx  # noqa: PLC0415
    embed_client = _httpx.AsyncClient(timeout=30)
    await vmem.init(http_client=embed_client)

    t0 = time.time()
    context_text: str = row.get("context") or ""
    nchunks = await ingest_context(context_text, kg, archive, vmem)
    ingest_s = time.time() - t0
    logger(
        f"  row[{row_idx}] source={source!r}: {len(context_text)} chars -> "
        f"{nchunks} chunks in {ingest_s:.1f}s, {len(questions)} questions"
    )

    results: list[dict] = []
    for q_idx, (question, gold_list) in enumerate(zip(questions, answers)):
        retrieved_ctx = await retrieve_context(
            question, kg, archive, vmem,
            graph_expansion=graph_expansion,
            retrieval_path=retrieval_path,
        )

        # Anchor evidence: with the flag on, also build the pre-wiring context
        # and record how far the two diverge, so "does the default path still
        # reproduce E-016?" is answered with a measurement, not an assumption.
        delta: dict | None = None
        if report_retrieval_delta:
            legacy_ctx = await retrieve_context(
                question, kg, archive, vmem, retrieval_path="legacy"
            )
            delta = {
                "legacy_chars": len(legacy_ctx),
                "wired_chars": len(retrieved_ctx),
                "identical": legacy_ctx == retrieved_ctx,
            }
            logger(
                f"    [DELTA] q={q_idx} legacy={delta['legacy_chars']} "
                f"wired={delta['wired_chars']} identical={delta['identical']}"
            )

        if not use_llm:
            logger(
                f"    [DRY] q={q_idx} ctx_len={len(retrieved_ctx)} "
                f"gold={gold_list!r} q={question[:48]!r}"
            )
            results.append({
                "row_idx": row_idx,
                "q_idx": q_idx,
                "source": source,
                "question": question,
                "gold_list": gold_list,
                "predicted": "",
                "score": 0,
                "retrieved_chars": len(retrieved_ctx),
                "retrieval_delta": delta,
                "dry": True,
            })
            continue

        predicted = await llm_answer(llm_client, retrieved_ctx, question)
        if SELF_VERIFY:
            predicted = await self_verify_answer(
                llm_client, retrieved_ctx, question, predicted
            )

        score = substring_exact_match(predicted, gold_list)
        logger(
            f"    q={q_idx} score={score} pred={predicted[:40]!r} "
            f"gold={gold_list!r}"
        )
        results.append({
            "row_idx": row_idx,
            "q_idx": q_idx,
            "source": source,
            "question": question,
            "gold_list": gold_list,
            "predicted": predicted,
            "score": score,
            "retrieved_chars": len(retrieved_ctx),
            "retrieval_delta": delta,
        })

    await archive.close()
    await kg.close()
    await vmem.close()
    await embed_client.aclose()
    return results


# ── metrics ───────────────────────────────────────────────────────────────────


def compute_accuracy(results: list[dict]) -> dict:
    """Compute Accuracy = mean(substring_exact_match) * 100 over all questions.

    Excludes dry-run rows (no generation, score is placeholder 0).
    Returns {"accuracy": float, "n": int, "correct": int}.
    """
    scored = [r for r in results if not r.get("dry")]
    if not scored:
        return {"accuracy": 0.0, "n": 0, "correct": 0}
    correct = sum(r["score"] for r in scored)
    n = len(scored)
    return {
        "accuracy": round(correct / n * 100, 2),
        "n": n,
        "correct": int(correct),
    }


def summarize_retrieval_delta(results: list[dict]) -> dict | None:
    """Summarise the wired-vs-legacy context comparison, or None if not measured.

    Only populated when the runner is given ``--report-retrieval-delta``. This
    is the evidence for whether routing through retrieve() preserved the E-016
    anchor: 100% byte-identical means the anchor carries over untouched, and
    anything less quantifies exactly how much has to be re-anchored.
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


# ── main entrypoint ───────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> None:
    print("=" * 70)
    print(
        f"EventQA Benchmark (taOSmd)  tier={args.tier}  contexts={args.contexts}  "
        f"limit={args.limit}  mode={'LLM' if args.llm else 'DRY'}"
    )
    print("=" * 70)
    print(
        f"  generator={REMOTE_LLM_MODEL}  embedder={os.path.basename(ONNX_PATH)}  "
        f"rerank={RERANK}  self_verify={SELF_VERIFY}  chunk_words={CHUNK_WORDS}"
    )
    print(
        f"  retrieval_path={args.retrieval_path}  graph_expansion={args.graph_expansion}  "
        f"num_ctx={NUM_CTX}"
    )

    if args.graph_expansion and not retrieve_supports_graph_expansion():
        print(
            "  ERROR: the installed taosmd.retrieval.retrieve() has no "
            "graph_expansion parameter, so both E-025 arms would be identical. "
            "Run this on a checkout that includes the bi-temporal fact-readback "
            "work (PR #191).",
            file=sys.stderr,
        )
        return

    rows = load_eventqa_rows(args.tier, args.contexts)
    if not rows:
        print("  ERROR: no rows loaded for tier", args.tier, file=sys.stderr)
        return

    llm_client = None
    if args.llm:
        import httpx as _httpx  # noqa: PLC0415
        llm_client = _httpx.AsyncClient(timeout=120)

    all_results: list[dict] = []
    for idx, row in enumerate(rows):
        row_results = await process_row(
            row, idx, args.llm, llm_client, question_limit=args.limit,
            graph_expansion=args.graph_expansion,
            retrieval_path=args.retrieval_path,
            report_retrieval_delta=args.report_retrieval_delta,
        )
        all_results.extend(row_results)

    if llm_client:
        await llm_client.aclose()

    metrics = compute_accuracy(all_results)
    delta_summary = summarize_retrieval_delta(all_results)

    if delta_summary:
        print("=" * 70)
        print("RETRIEVAL DELTA (wired retrieve() path vs pre-wiring legacy path)")
        print(f"  questions compared:      {delta_summary['n']}")
        print(f"  byte-identical contexts: {delta_summary['identical']}"
              f" ({delta_summary['identical_pct']:.1f}%)")
        print(f"  mean chars legacy/wired: {delta_summary['mean_legacy_chars']:.0f}"
              f" / {delta_summary['mean_wired_chars']:.0f}")

    print("=" * 70)
    if not args.llm:
        n_dry = sum(1 for r in all_results if r.get("dry"))
        print("DRY WIRING CHECK -- pipeline verified, rerun with --llm for real scores")
        print(f"  rows ingested:   {len(rows)}")
        print(f"  questions parsed: {n_dry}")
        print("=" * 70)
        return

    print("EVENTQA RESULTS -- taOSmd")
    print(f"  tier:         {args.tier}")
    print(f"  self_verify:  {SELF_VERIFY}")
    print(f"  n questions:  {metrics['n']}")
    print(f"  correct:      {metrics['correct']}")
    print(f"  Accuracy:     {metrics['accuracy']:.2f}%")
    print("=" * 70)

    if args.out:
        out_path = args.out
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"eventqa_{args.tier}_{int(time.time())}.json"
        )

    result_doc = {
        "tier": args.tier,
        "contexts": args.contexts,
        "question_limit": args.limit,
        "generator": REMOTE_LLM_MODEL,
        "onnx_path": ONNX_PATH,
        "rerank": RERANK,
        "self_verify": SELF_VERIFY,
        "chunk_words": CHUNK_WORDS,
        "num_ctx": NUM_CTX,
        "retrieval_path": args.retrieval_path,
        "graph_expansion": args.graph_expansion,
        "retrieval_delta": delta_summary,
        "metrics": metrics,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    print(f"  results -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate taOSmd on MemoryAgentBench EventQA"
    )
    parser.add_argument(
        "--tier",
        default="eventqa_65536",
        choices=list(VALID_TIERS),
        help="EventQA context-length tier (default eventqa_65536)",
    )
    parser.add_argument(
        "--contexts",
        type=int,
        default=5,
        help="Number of EventQA rows to process (default 5 = smoke run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap total questions per row (0 = all, default 0)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable generation via Ollama (needs a GPU host). "
             "Omit for a GPU-free dry wiring check.",
    )
    parser.add_argument(
        "--graph-expansion",
        type=int,
        default=int(os.environ.get("TAOSMD_GRAPH_EXPANSION", "0")),
        metavar="N",
        help="Bi-temporal fact-readback token budget (0 = off, the default). "
             "This is the E-025 lever; needs --retrieval-path retrieve.",
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
             "wired path diverges (E-016 anchor evidence). Doubles retrieval cost.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Result JSON path (default: benchmarks/results/eventqa_<tier>_<ts>.json)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
