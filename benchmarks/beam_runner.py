#!/usr/bin/env python3
"""Run the BEAM benchmark against taOSmd, scored with mem0's OWN judge.

BEAM (ICLR 2026, dataset ``Mohammadta/BEAM``) probes 10 memory abilities over
long conversations (100K / 500K / 1M / 10M token buckets). Each conversation
carries 20 ``probing_questions`` (2 per ability type), each with a ``rubric``
that is a list of natural-language criteria ("nuggets").

This runner reproduces mem0ai/memory-benchmarks' BEAM pipeline so the resulting
number is a DIRECT head-to-head with mem0's published BEAM scores:

  * Answer generation prompt          -> ported verbatim from mem0's
                                         get_beam_answer_generation_prompt
  * LLM-as-judge prompt + 0/0.5/1.0   -> ported verbatim from mem0's
    per-nugget scoring                   get_beam_nugget_judge_prompt
  * Question score = mean(nugget)      -> mem0's process_question
  * PASS at score >= 0.5, accuracy     -> mem0's compute_beam_metrics
    per question-type + overall

The RETRIEVAL + GENERATION pipeline is taOSmd's (the thing under test): the
exact ingest -> assemble -> FTS -> vector.search -> Ollama chain from
longmemeval_runner.py, with the same E-012 env levers so a winning config
carries over (TAOSMD_OLLAMA_MODEL, TAOSMD_JUDGE_MODEL, TAOSMD_ONNX_PATH,
TAOSMD_ONNX_POOLING, TAOSMD_ONNX_QUERY_PREFIX, TAOSMD_RERANK, TAOSMD_SELF_VERIFY).

Ingest chunking is also sweepable without code edits (untested score lever):
TAOSMD_CHUNK_WORDS (default 100), TAOSMD_CHUNK_OVERLAP (default 20), and
TAOSMD_CHUNK_MODE ("words" default | "sentences" = pack whole sentences, no
mid-sentence splits). Defaults reproduce the original 100/20-word windows.

GPU NOTE: --llm needs a live Ollama (a GPU host). Without it the runner does a
--dry wiring check: ingest + retrieve + print what it WOULD judge. Build/wire/
test it GPU-free now; run --llm on a GPU window later.

Usage:
    # dry wiring check (no LLM, no GPU) — what the default smoke run does:
    python3 benchmarks/beam_runner.py --split 100K --limit 5

    # full head-to-head (needs a live Ollama / GPU):
    TAOSMD_OLLAMA_MODEL=qwen3.5:9b TAOSMD_JUDGE_MODEL=qwen3-4b-instruct \\
    TAOSMD_ONNX_PATH=models/arctic-embed-s-onnx TAOSMD_RERANK=1 \\
    python3 benchmarks/beam_runner.py --split 100K --limit 100 --llm
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taosmd.knowledge_graph import TemporalKnowledgeGraph
from taosmd.archive import ArchiveStore
from taosmd.memory_extractor import process_conversation_turn
from taosmd.context_assembler import ContextAssembler
from taosmd.vector_memory import VectorMemory

# ── env levers (mirror longmemeval_runner so the winning E-012 config carries) ──
REMOTE_LLM_URL = os.environ.get("TAOSMD_OLLAMA_URL", "http://localhost:11434")
REMOTE_LLM_MODEL = os.environ.get("TAOSMD_OLLAMA_MODEL", "qwen2.5:3b")
JUDGE_MODEL = os.environ.get("TAOSMD_JUDGE_MODEL", REMOTE_LLM_MODEL)
CONTEXT_CHARS = int(os.environ.get("TAOSMD_CONTEXT_CHARS", "16000"))
ASSEMBLE_TOKENS = int(os.environ.get("TAOSMD_ASSEMBLE_TOKENS", "4000"))
RETRIEVE_LIMIT = int(os.environ.get("TAOSMD_RETRIEVE_LIMIT", "12"))
FTS_LIMIT = int(os.environ.get("TAOSMD_FTS_LIMIT", "5"))
RERANK = os.environ.get("TAOSMD_RERANK", "0") == "1"
RERANK_PATH = os.environ.get("TAOSMD_RERANK_PATH", "models/bge-reranker-v2-m3-onnx")
RERANK_TOP_K = int(os.environ.get("TAOSMD_RERANK_TOP_K", "8"))
SELF_VERIFY = os.environ.get("TAOSMD_SELF_VERIFY", "0") == "1"
ONNX_PATH = os.environ.get(
    "TAOSMD_ONNX_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx"),
)
# TAOSMD_ONNX_POOLING / TAOSMD_ONNX_QUERY_PREFIX / TAOSMD_ONNX_DOC_PREFIX are
# read directly by VectorMemory's embedder from the environment, so they apply
# automatically — nothing to thread through here.

# Ingest chunking levers (untested score knob: how we window 100K-1M-token
# conversations decides whether the exact-fact chunk gets retrieved). Defaults
# preserve the original hardcoded 100-word / 20-word-overlap behaviour.
CHUNK_WORDS = int(os.environ.get("TAOSMD_CHUNK_WORDS", "100"))
CHUNK_OVERLAP = int(os.environ.get("TAOSMD_CHUNK_OVERLAP", "20"))
CHUNK_MODE = os.environ.get("TAOSMD_CHUNK_MODE", "words")  # "words" (default) | "sentences"

HF_DATASET = "Mohammadta/BEAM"
HF_DATASET_10M = "Mohammadta/BEAM-10M"
VALID_SPLITS = ("100K", "500K", "1M")  # 10M lives in a separate HF dataset

# The 10 BEAM memory-ability types, with mem0's descriptions (prompts.py).
BEAM_QUESTION_TYPES: dict[str, str] = {
    "abstention": "Withholding answers when evidence is absent from the conversation",
    "contradiction_resolution": "Detecting and reconciling inconsistent statements across dialogue turns",
    "event_ordering": "Reconstructing the chronological sequence of events and developments",
    "information_extraction": "Recalling specific entities, dates, numbers, and factual details",
    "instruction_following": "Sustained adherence to user-specified constraints and formatting preferences",
    "knowledge_update": "Revising stored facts when new or corrected information appears",
    "multi_session_reasoning": "Integrating evidence scattered across non-adjacent dialogue segments",
    "preference_following": "Adapting responses to evolving user preferences and personal choices",
    "summarization": "Abstracting and compressing dialogue content into concise summaries",
    "temporal_reasoning": "Reasoning about explicit and implicit time relations, durations, and sequences",
}

_reranker = None


# ─────────────────────────── dataset loading ───────────────────────────


def load_beam_split(split: str, limit: int, logger=print) -> list[dict]:
    """Load ``limit`` BEAM conversations for ``split``.

    Prefers ``datasets.load_dataset`` (full split, HF-cached). Falls back to the
    HF datasets-server rows API (streams just the rows we need, no full download)
    when ``datasets`` is not installed — same path the feasibility probe uses.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; valid: {VALID_SPLITS}")

    rows: list[dict] = []
    try:
        from datasets import load_dataset  # noqa: PLC0415

        logger(f"  loading {HF_DATASET} split={split} via datasets.load_dataset ...")
        ds = load_dataset(HF_DATASET, split=split)
        for i, item in enumerate(ds):
            if i >= limit:
                break
            rows.append(dict(item))
        return rows
    except ImportError:
        logger("  `datasets` not installed; falling back to HF datasets-server rows API")
    except Exception as exc:  # noqa: BLE001
        logger(f"  datasets.load_dataset failed ({exc}); falling back to rows API")

    # Rows-API fallback: page through `limit` rows (server caps length at 100).
    offset = 0
    while len(rows) < limit:
        page = min(100, limit - len(rows))
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={HF_DATASET}&config=default&split={split}"
            f"&offset={offset}&length={page}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "taosmd-beam-runner/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        batch = [row["row"] for row in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:limit]


def _coerce_dict(value):
    """Parse a value that HF may store as a Python-repr or JSON string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for parser in (ast.literal_eval, json.loads):
            try:
                out = parser(value)
                if isinstance(out, dict):
                    return out
            except (ValueError, SyntaxError):
                continue
    return {}


# Answer-bearing keys vary by category in the real dataset:
#   information_extraction/event_ordering -> "answer"
#   abstention                            -> "ideal_response"
#   contradiction_resolution              -> "ideal_answer"
#   summarization                         -> "ideal_summary"
_ANSWER_KEYS = ("answer", "ideal_response", "ideal_answer", "ideal_summary", "ideal")


def parse_probing_questions(row: dict) -> list[dict]:
    """Flatten a BEAM row's ``probing_questions`` into a list of question dicts.

    Returns one normalized dict per question:
        {question_type, question, rubric: [str, ...], gold, difficulty, raw}

    ``probing_questions`` is keyed by the 10 ability types; each value is a list
    of 2 question dicts. The ``rubric`` field is a list of natural-language
    nugget strings (mem0's judge scores each independently). We DO NOT use the
    instruction-following ``user_questions`` field here — ``probing_questions``
    is the rubric-judged surface that makes the score comparable to mem0's.
    """
    pq = _coerce_dict(row.get("probing_questions", {}))
    out: list[dict] = []
    for qtype in BEAM_QUESTION_TYPES:
        for item in pq.get(qtype, []) or []:
            if not isinstance(item, dict):
                # Bare-string question with no rubric; keep it, judge will ERROR.
                out.append({
                    "question_type": qtype,
                    "question": str(item),
                    "rubric": [],
                    "gold": "",
                    "difficulty": "unknown",
                    "raw": item,
                })
                continue
            out.append({
                "question_type": qtype,
                "question": item.get("question", item.get("question_text", "")),
                "rubric": extract_rubric_nuggets(item),
                "gold": next((item[k] for k in _ANSWER_KEYS if item.get(k)), ""),
                "difficulty": item.get("difficulty", "unknown"),
                "raw": item,
            })
    return out


def extract_rubric_nuggets(question_data: dict) -> list[str]:
    """Extract the list of rubric nugget strings from a probing-question dict.

    BEAM stores ``rubric`` as a list of strings. We also tolerate mem0's
    HF-string shape (``{"nuggets": [{"description": ...}]}``) and a bare string,
    so the same parser works across dataset revisions.
    """
    rubric = question_data.get("rubric", [])
    if isinstance(rubric, list):
        return [str(n).strip() for n in rubric if str(n).strip()]
    if isinstance(rubric, dict):
        nuggets = rubric.get("nuggets", [])
        return [
            (n.get("description", str(n)) if isinstance(n, dict) else str(n)).strip()
            for n in nuggets
            if str(n).strip()
        ]
    if rubric:
        return [str(rubric).strip()]
    return []


# ─────────────────────────── chat parsing ───────────────────────────


def parse_beam_chat(chat_data) -> list[tuple[str, str]]:
    """Normalize a BEAM ``chat`` field into a flat list of (role, text) turns.

    Handles the real-data shapes: a 2D list ``[[turn, ...], ...]`` (1M-and-below),
    a list of session/batch dicts with a ``turns`` key (10M/plan formats), and a
    flat list of turn dicts. Each turn may be a dict (``role``/``content``) or a
    bare string.
    """
    turns: list[tuple[str, str]] = []

    def _emit(t):
        if isinstance(t, dict):
            text = t.get("content") or t.get("text") or t.get("message") or ""
            role = t.get("role") or t.get("speaker") or "user"
        else:
            text, role = str(t), "user"
        if text:
            role = role if role in ("user", "assistant") else (
                "assistant" if str(role).lower() in ("assistant", "ai", "bot") else "user"
            )
            turns.append((role, text))

    def _walk(node):
        if isinstance(node, dict):
            if "turns" in node:
                _walk(node["turns"])
            elif "content" in node or "text" in node or "message" in node or "role" in node:
                _emit(node)
            else:  # plan/session dict: recurse into its batch lists
                for v in node.values():
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif node is not None:
            _emit(node)

    _walk(chat_data or [])
    return turns


# ─────────────────────────── ingest chunking ───────────────────────────

import re  # noqa: E402  (kept beside the chunker that uses it)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split ``text`` into overlapping word windows of ``words`` words each.

    Pure helper (testable, no I/O). Slides a fixed window with ``overlap`` words
    of carry-over, exactly reproducing the original hardcoded ``chunk_size=100,
    overlap=20`` behaviour at its defaults. Empty/whitespace chunks are dropped;
    no chunk ever exceeds ``words`` words.
    """
    toks = text.split()
    if not toks:
        return []
    step = max(1, words - overlap)
    chunks: list[str] = []
    for start in range(0, len(toks), step):
        chunk = " ".join(toks[start:start + words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_text_sentences(text: str, words: int = CHUNK_WORDS) -> list[str]:
    """Pack whole sentences into chunks of up to ~``words`` words each.

    Sentence-aware mode: never splits mid-sentence. A single sentence longer than
    ``words`` becomes its own (over-length) chunk rather than being cut. No word
    overlap (sentence boundaries already give natural context edges).
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        slen = len(sent.split())
        if cur and cur_len + slen > words:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += slen
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_chat_text(text: str) -> list[str]:
    """Chunk assembled chat text per the configured TAOSMD_CHUNK_* levers."""
    if CHUNK_MODE == "sentences":
        return chunk_text_sentences(text, CHUNK_WORDS)
    return chunk_text(text, CHUNK_WORDS, CHUNK_OVERLAP)


# ─────────────────────────── mem0 prompts (ported verbatim) ───────────────────────────

# Ported from mem0ai/memory-benchmarks benchmarks/beam/prompts.py.
BEAM_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing whether an AI assistant's response satisfies "
    "specific rubric criteria. You must be objective, fair, and consistent. "
    "Return ONLY valid JSON with the exact format requested."
)


def get_beam_answer_generation_prompt(question: str, memories: list[str]) -> str:
    """mem0's answer-generation prompt (get_beam_answer_generation_prompt)."""
    if not memories:
        memories_text = "(No memories available)"
    else:
        memories_text = "\n".join(f"{i}. {m}" for i, m in enumerate(memories, 1))
    return f"""You are an AI assistant with access to stored memories from prior conversations with a user.
Use these memories to answer the following question as accurately and completely as possible.

IMPORTANT RULES:
1. Scan ALL provided memories before answering — do not stop after the first relevant one.
2. If multiple memories contain relevant information, combine and cross-reference them.
3. If the memories contain contradictory information, prefer the more recent one.
4. If the memories don't contain enough information to answer, say exactly: "I don't have enough information to answer this question."
5. For temporal questions: pay attention to dates and relative time references.
6. For ordering questions: present events in chronological order.
7. For preference questions: use the most recently stated preference.
8. Be specific and direct — include exact names, dates, numbers, and details from the memories.
9. Do NOT invent or assume information that isn't in the memories.

QUESTION: {question}

RETRIEVED MEMORIES:
{memories_text}

ANSWER:"""


def get_beam_nugget_judge_prompt(question: str, nugget: str, llm_response: str) -> str:
    """mem0's per-nugget judge prompt (get_beam_nugget_judge_prompt), verbatim.

    The judge classifies the nugget as a POSITIVE requirement or NEGATIVE
    constraint, then scores compliance 0.0 / 0.5 / 1.0 and returns JSON
    ``{"score": ..., "reason": ...}``. Reproduced exactly so our number is
    comparable to mem0's published BEAM table.
    """
    return f"""Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{question}

LLM RESPONSE:
{llm_response}

RUBRIC CRITERION:
{nugget}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete. The response fully and clearly satisfies the rubric criterion.
- **0.5 (Partial Compliance)**: The required element is partially present, has minor inaccuracies, or is incomplete. The core intent is present but not fully realized.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0 (Complete Compliance)**: The response is responsive to the question AND the prohibited element is absent.
- **0.5 (Partial Compliance)**: The response is responsive but contains a borderline or ambiguous reference to the prohibited element.
- **0.0 (No Compliance)**: The prohibited element is present in the response, OR the response is non-responsive (off-topic, refusal, empty).

**Compound statement handling**: If the rubric criterion contains "and" or commas connecting multiple required elements:
- All elements present and correct = 1.0
- Some (but not all) elements present and correct = 0.5
- No elements present or correct = 0.0

EVALUATION RULES:
1. **Semantic tolerance**: Paraphrases and synonyms are acceptable. The response does not need to use the exact same words as the rubric.
2. **Numeric and date equivalence**: Treat equivalent representations as identical. "$68,000" = "68k" = "sixty-eight thousand dollars". "2 years" = "24 months". Prefer normalized comparison for numbers, currencies, dates, and durations.
3. **Case / punctuation / whitespace tolerance**: Differences in capitalization, punctuation, and whitespace must be ignored when comparing content.
4. **Hedging tolerance**: Do not penalize hedging language ("I think", "probably", "it seems"), passive voice, or verbosity if the substantive content satisfies the rubric criterion.
5. **Style neutrality**: Do not penalize for tone, formatting, or length unless the rubric criterion specifically requires a particular format.
6. **Responsiveness**: If the LLM response is completely off-topic or refuses to answer, score 0.0 for all criteria.
7. **Independence**: Evaluate this criterion in isolation — do not consider other rubric items.
8. **Specificity matters**: Vague or generic answers that could apply to any question score lower than specific, detailed answers.

STEP-BY-STEP EVALUATION:
Follow these steps in order:
1. **Understand the Requirement**: Read the rubric criterion and classify it as a positive requirement or a negative constraint.
2. **Parse Compound Statements**: If the criterion contains multiple sub-requirements joined by "and" or commas, identify each element separately.
3. **Check Compliance**: Compare the LLM response against each element, applying the tolerance rules above (semantic, numeric, case, hedging).
4. **Assign Score**: Use the appropriate scoring table (positive or negative) and compound-statement rule to determine the score.
5. **Provide Reasoning**: Write a concise explanation referencing which elements were or were not satisfied.

Return your evaluation as a JSON object with exactly two fields:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence explaining your score>"}}"""


def clamp_nugget_score(raw_score: float) -> float:
    """mem0's _clamp_nugget_score: snap a raw score to 0.0 / 0.5 / 1.0."""
    if raw_score >= 0.75:
        return 1.0
    if raw_score >= 0.25:
        return 0.5
    return 0.0


def parse_judge_json(raw: str) -> dict:
    """Parse the judge's reply into ``{"score": float, "reason": str}``.

    Tolerates JSON wrapped in prose / code fences (small Ollama judges do this).
    Falls back to scanning for a 1.0 / 0.5 token, else 0.0 — matching mem0's
    judge_single_nugget fallback behaviour.
    """
    text = (raw or "").strip()
    # Try direct JSON, then the first {...} block.
    candidates = []
    if text:
        candidates.append(text)
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "score" in obj:
                return {
                    "score": clamp_nugget_score(float(obj.get("score", 0.0))),
                    "reason": str(obj.get("reason", "")),
                }
        except (ValueError, TypeError):
            continue
    if "1.0" in text or '"score": 1' in text:
        return {"score": 1.0, "reason": text[:200]}
    if "0.5" in text:
        return {"score": 0.5, "reason": text[:200]}
    return {"score": 0.0, "reason": f"Parse error: {text[:200]}"}


def score_question(nugget_scores: list[dict]) -> dict:
    """mem0's question scoring: mean of nugget scores, PASS at >= 0.5."""
    scores = [ns["score"] for ns in nugget_scores]
    avg = statistics.mean(scores) if scores else 0.0
    return {
        "score": round(avg, 4),
        "judgment": "PASS" if avg >= 0.5 else "FAIL",
        "nugget_scores": nugget_scores,
    }


# ─────────────────────────── LLM calls (taOSmd pipeline) ───────────────────────────


async def llm_answer(client, memories: list[str], question: str) -> str:
    """Generate an answer from retrieved memories using mem0's prompt."""
    prompt = get_beam_answer_generation_prompt(question, memories)[:CONTEXT_CHARS + 4000]
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": REMOTE_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt + " /no_think"}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 256},
            },
            timeout=60,
        )
        if resp.status_code == 200:
            ans = resp.json().get("message", {}).get("content", "")
            if "ANSWER:" in ans:
                ans = ans.rsplit("ANSWER:", 1)[-1].strip()
            return ans
    except Exception:
        pass
    return ""


async def judge_nugget(client, question: str, nugget: str, answer: str) -> dict:
    """Judge a single rubric nugget with mem0's judge prompt + scoring."""
    prompt = get_beam_nugget_judge_prompt(question, nugget, answer)
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": JUDGE_MODEL,
                "messages": [
                    {"role": "system", "content": BEAM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt + " /no_think"},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 200},
            },
            timeout=60,
        )
        if resp.status_code == 200:
            raw = resp.json().get("message", {}).get("content", "")
            return {"nugget": nugget, **parse_judge_json(raw)}
    except Exception:
        pass
    return {"nugget": nugget, "score": 0.0, "reason": "judge call failed"}


VERIFY_PROMPT = """You are checking a draft answer against the memories.
If every part of the draft answer is supported by the memories, repeat the draft answer exactly.
If any part is unsupported or contradicted by the memories, write a corrected answer using ONLY the memories.
Be specific and concise. /no_think

Memories:
{context}

Question: {question}

Draft answer: {answer}

Final answer:"""


async def self_verify_answer(client, memories: list[str], question: str, answer: str) -> str:
    """One CoVe-style verification pass (E-012 lever). Never worsens on error."""
    if not answer:
        return answer
    context = "\n".join(memories)[:CONTEXT_CHARS]
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": REMOTE_LLM_MODEL,
                "messages": [{"role": "user", "content": VERIFY_PROMPT.format(
                    context=context, question=question, answer=answer)}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 256},
            },
            timeout=60,
        )
        if resp.status_code == 200:
            revised = resp.json().get("message", {}).get("content", "").strip()
            if revised:
                return revised
    except Exception:
        pass
    return answer


# ─────────────────────────── retrieval (taOSmd pipeline) ───────────────────────────


def _get_reranker():
    global _reranker
    if _reranker is None and RERANK:
        from taosmd.cross_encoder import CrossEncoderReranker  # noqa: PLC0415
        _reranker = CrossEncoderReranker(onnx_path=RERANK_PATH)
    return _reranker


async def retrieve_memories(question: str, kg, archive, vmem) -> list[str]:
    """taOSmd retrieval: assemble + FTS + vector.search, optional rerank.

    Returns an ordered list of memory strings (assembled context first, then FTS
    snippets, then vector chunks) to feed mem0's answer-generation prompt.
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

    vector_results = await vmem.search(question, limit=RETRIEVE_LIMIT)
    rr = _get_reranker()
    if rr is not None and getattr(rr, "available", False) and vector_results:
        vector_results = rr.rerank(question, vector_results, RERANK_TOP_K)
    memories.extend(r["text"] for r in vector_results if r.get("text"))

    # Dedup preserving order.
    seen, deduped = set(), []
    for m in memories:
        if m and m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


async def ingest_conversation(turns: list[tuple[str, str]], kg, archive, vmem) -> int:
    """Ingest a BEAM conversation: KG facts + archive + chunked vector embeds.

    Mirrors longmemeval_runner's ingest (process_conversation_turn + archive +
    word/sentence chunks per the TAOSMD_CHUNK_* levers; defaults to the original
    ~100-word / 20-overlap windows). Returns the number of vector chunks embedded.
    """
    full_text = ""
    for role, content in turns:
        await process_conversation_turn(
            content,
            agent_name="assistant" if role == "assistant" else None,
            kg=kg, archive=archive, source="beam",
        )
        await archive.record("conversation", {"role": role, "content": content}, summary=content[:80])
        full_text += f"\n[{role}]: {content}"

    nchunks = 0
    for chunk in chunk_chat_text(full_text):
        await vmem.add(chunk, metadata={})
        nchunks += 1
    return nchunks


# ─────────────────────────── per-conversation driver ───────────────────────────


async def process_conversation(row: dict, idx: int, use_llm: bool, llm_client) -> list[dict]:
    """Ingest one conversation, then retrieve+generate+judge each question.

    In --dry mode (use_llm False) it stops after retrieval and records what it
    WOULD judge (question, rubric nuggets, retrieved memory count), so wiring is
    verifiable without an LLM/GPU.
    """
    turns = parse_beam_chat(row.get("chat", []))
    questions = parse_probing_questions(row)
    conv_id = str(row.get("conversation_id", idx))

    tmp = tempfile.mkdtemp()
    kg = TemporalKnowledgeGraph(db_path=os.path.join(tmp, "kg.db"))
    archive = ArchiveStore(archive_dir=os.path.join(tmp, "archive"), index_path=os.path.join(tmp, "idx.db"))
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
    nchunks = await ingest_conversation(turns, kg, archive, vmem)
    ingest_s = time.time() - t0
    print(f"  conv[{idx}] id={conv_id}: {len(turns)} turns -> {nchunks} chunks in {ingest_s:.1f}s, {len(questions)} questions")

    results: list[dict] = []
    for q in questions:
        memories = await retrieve_memories(q["question"], kg, archive, vmem)
        rec = {
            "conversation_id": conv_id,
            "question_type": q["question_type"],
            "question": q["question"],
            "rubric": q["rubric"],
            "gold": q["gold"],
            "retrieved_count": len(memories),
        }

        if not use_llm:
            rec["dry"] = True
            print(f"    [DRY] {q['question_type']:24s} q={q['question'][:48]!r} "
                  f"nuggets={len(q['rubric'])} retrieved={len(memories)}")
            results.append(rec)
            continue

        answer = await llm_answer(llm_client, memories, q["question"])
        if SELF_VERIFY:
            answer = await self_verify_answer(llm_client, memories, q["question"], answer)

        if not q["rubric"]:
            rec.update({"judgment": "ERROR", "score": 0.0, "generated_answer": answer,
                        "nugget_scores": [], "error": "No rubric nuggets"})
        else:
            nugget_scores = [await judge_nugget(llm_client, q["question"], n, answer) for n in q["rubric"]]
            rec.update({"generated_answer": answer, **score_question(nugget_scores)})
        print(f"    {q['question_type']:24s} score={rec.get('score', 0.0):.2f} "
              f"[{rec.get('judgment', '?')}] {q['question'][:44]}")
        results.append(rec)

    await archive.close()
    await kg.close()
    await vmem.close()
    await embed_client.aclose()
    return results


# ─────────────────────────── metrics + display ───────────────────────────


def compute_metrics(evaluations: list[dict], pass_threshold: float = 0.5) -> dict:
    """Per-question-type + overall accuracy (mem0's compute_beam_metrics)."""
    scored = [e for e in evaluations if "score" in e]
    by_type: dict[str, list[dict]] = {qt: [] for qt in BEAM_QUESTION_TYPES}
    for e in scored:
        by_type.setdefault(e["question_type"], []).append(e)

    type_metrics = {}
    for qt, items in by_type.items():
        if not items:
            continue
        scores = [i["score"] for i in items]
        correct = sum(1 for s in scores if s >= pass_threshold)
        type_metrics[qt] = {
            "total": len(items),
            "correct": correct,
            "accuracy": correct / len(items) * 100,
            "avg_score": round(statistics.mean(scores), 4),
        }

    all_scores = [e["score"] for e in scored]
    total = len(all_scores)
    correct = sum(1 for s in all_scores if s >= pass_threshold)
    return {
        "overall": {
            "total": total,
            "correct": correct,
            "errors": sum(1 for e in scored if e.get("error")),
            "accuracy": correct / total * 100 if total else 0.0,
            "avg_score": round(statistics.mean(all_scores), 4) if all_scores else 0.0,
        },
        "by_question_type": type_metrics,
    }


def display_results(metrics: dict) -> None:
    o = metrics["overall"]
    print(f"\n{'=' * 70}")
    print("BEAM RESULTS — taOSmd (mem0 judge methodology, head-to-head)")
    print(f"{'=' * 70}")
    print(f"\n  Overall: {o['correct']}/{o['total']} pass (>= 0.5)  "
          f"accuracy={o['accuracy']:.1f}%  avg_score={o['avg_score']:.3f}  errors={o['errors']}")
    print("\n  By memory-ability type:")
    for qt, tm in sorted(metrics["by_question_type"].items()):
        print(f"    {qt:26s} {tm['correct']:3d}/{tm['total']:<3d} "
              f"({tm['accuracy']:5.1f}%)  avg={tm['avg_score']:.3f}")
    print(f"{'=' * 70}")


# ─────────────────────────── main ───────────────────────────


async def run_benchmark(split: str, limit: int, use_llm: bool) -> dict | None:
    print("=" * 70)
    print(f"BEAM Benchmark — taOSmd   split={split}  limit={limit}  "
          f"mode={'LLM judge' if use_llm else 'DRY wiring check'}")
    print("=" * 70)
    print(f"  generator={REMOTE_LLM_MODEL}  judge={JUDGE_MODEL}  embedder={os.path.basename(ONNX_PATH)}  "
          f"rerank={RERANK}  self_verify={SELF_VERIFY}")

    rows = load_beam_split(split, limit)
    if not rows:
        print("  ERROR: no conversations loaded")
        return None
    print(f"  loaded {len(rows)} conversation(s)")

    llm_client = None
    if use_llm:
        import httpx as _httpx  # noqa: PLC0415
        llm_client = _httpx.AsyncClient(timeout=60)

    all_results: list[dict] = []
    for idx, row in enumerate(rows):
        all_results.extend(await process_conversation(row, idx, use_llm, llm_client))

    if llm_client:
        await llm_client.aclose()

    if not use_llm:
        n_q = len(all_results)
        n_with_rubric = sum(1 for r in all_results if r["rubric"])
        avg_retrieved = statistics.mean([r["retrieved_count"] for r in all_results]) if all_results else 0
        print(f"\n{'=' * 70}")
        print("DRY WIRING CHECK — what a full --llm run WOULD judge")
        print(f"{'=' * 70}")
        print(f"  conversations ingested : {len(rows)}")
        print(f"  questions parsed       : {n_q}  (with rubric: {n_with_rubric})")
        print(f"  avg memories retrieved : {avg_retrieved:.1f} per question")
        print(f"  question types present : {sorted({r['question_type'] for r in all_results})}")
        print("  -> pipeline wired; rerun with --llm on a GPU host for the real score")
        print(f"{'=' * 70}")
        return None

    metrics = compute_metrics(all_results)
    display_results(metrics)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"beam_{split}_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump({"split": split, "limit": limit, "generator": REMOTE_LLM_MODEL,
                   "judge": JUDGE_MODEL, "metrics": metrics, "results": all_results}, f, indent=2)
    print(f"\n  results -> {out_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEAM against taOSmd with mem0's judge")
    parser.add_argument("--split", default="100K", choices=list(VALID_SPLITS),
                        help="BEAM token bucket (default 100K)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of conversations (default 5 = smoke run)")
    parser.add_argument("--llm", action="store_true",
                        help="Enable generate+judge via Ollama (needs a GPU host). "
                             "Omit for a GPU-free dry wiring check.")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.split, args.limit, args.llm))


if __name__ == "__main__":
    main()
