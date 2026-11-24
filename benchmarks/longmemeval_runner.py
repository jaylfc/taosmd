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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.product_context import (
    CONTEXT_MODES,
    PRODUCT_AGENT,
    PRODUCT_PROMPT,
    count_in,
    check_ingest,
    new_product_dir,
    product_environment,
    require_healthy,
    product_meta as _product_meta,
    reader_ctx_problem,
    refuse,
    release_product_stores,
    resolve_context_mode as _resolve_context_mode,
)
from taosmd import api as _api
from taosmd.answer_context import answer_context as _answer_context
from taosmd.knowledge_graph import TemporalKnowledgeGraph
from taosmd.archive import ArchiveStore
from taosmd.memory_extractor import process_conversation_turn
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
VALID_STREAMS = {"assembler", "fts", "vector"}


def _date_frame_from_env() -> str:
    """Parse TAOSMD_LME_DATE_FRAME, exiting 2 on invalid values.

    Valid values: off, question, sessions, both, labelled.
    Unset or empty-after-strip -> 'off'.
    Any other value -> print error naming the bad value and the five valid ones,
    then sys.exit(2). No silent fallback.
    """
    raw = os.environ.get("TAOSMD_LME_DATE_FRAME")
    if not raw or not raw.strip():
        return "off"
    val = raw.strip()
    valid = ("off", "question", "sessions", "both", "labelled")
    if val not in valid:
        print(
            f"  ERROR: TAOSMD_LME_DATE_FRAME={val!r} is not valid. "
            f"Valid values: {', '.join(valid)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return val


def _parse_haystack_date(value: str):
    """Parse a LongMemEval haystack date string.

    Format: '2023/05/20 (Sat) 02:21'
    Returns datetime or raises ValueError.
    """
    return datetime.strptime(value, "%Y/%m/%d (%a) %H:%M")


def _hit_session_date(hit: dict) -> str | None:
    """Extract session_date from a hit, tolerating flat or nested metadata."""
    meta = hit.get("metadata", {}) or {}
    sd = meta.get("session_date")
    if sd:
        return sd
    inner = meta.get("metadata")
    if isinstance(inner, dict):
        return inner.get("session_date")
    return None


def _validate_dates(item: dict, date_frame: str, question_id: str | None = None):
    """Validate that required dates are present and parseable for non-off arms.

    Exits 1 with question_id and offending value on failure.
    """
    if date_frame == "off":
        return

    qid = question_id or item.get("question_id")
    if not qid:
        qid = f"<no question_id> {item.get('question', 'unknown')[:50]}"

    # question_date required for question/both/labelled
    if date_frame in ("question", "both", "labelled"):
        qdate = item.get("question_date")
        if not qdate:
            print(
                f"  ERROR: question {qid!r} missing question_date (required for date_frame={date_frame})",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            _parse_haystack_date(qdate)
        except ValueError:
            print(
                f"  ERROR: question {qid!r} has unparseable question_date: {qdate!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    # haystack_dates required for sessions/both/labelled
    if date_frame in ("sessions", "both", "labelled"):
        haystack_dates = item.get("haystack_dates")
        haystack_sessions = item.get("haystack_sessions", [])
        if haystack_dates is None:
            print(
                f"  ERROR: question {qid!r} missing haystack_dates (required for date_frame={date_frame})",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(haystack_dates) != len(haystack_sessions):
            print(
                f"  ERROR: question {qid!r} haystack_dates length ({len(haystack_dates)}) "
                f"!= haystack_sessions length ({len(haystack_sessions)})",
                file=sys.stderr,
            )
            sys.exit(1)
        for d in haystack_dates:
            try:
                _parse_haystack_date(d)
            except ValueError:
                print(
                    f"  ERROR: question {qid!r} has unparseable haystack_date: {d!r}",
                    file=sys.stderr,
                )
                sys.exit(1)


def _parse_streams_env() -> list[str]:
    """Parse TAOSMD_LME_STREAMS env var into a sorted list of stream names.

    Value is a comma-separated, non-empty subset of {assembler, fts, vector},
    order-insensitive, whitespace-tolerant.

    - unset: returns all three streams (default behavior)
    - unknown token, duplicate, empty token, or empty list: prints error to stderr and exits 2
    """
    raw = os.environ.get("TAOSMD_LME_STREAMS")
    if raw is None:
        return ["assembler", "fts", "vector"]

    tokens = [t.strip() for t in raw.split(",")]
    # Check for empty tokens (from leading/trailing/comma-only)
    if any(not t for t in tokens):
        print(
            f"ERROR: TAOSMD_LME_STREAMS={raw!r} contains empty token; valid tokens are: "
            "assembler, fts, vector",
            file=sys.stderr,
        )
        sys.exit(2)

    if not tokens:
        print(
            f"ERROR: TAOSMD_LME_STREAMS={raw!r} is empty; valid tokens are: "
            "assembler, fts, vector",
            file=sys.stderr,
        )
        sys.exit(2)

    seen = set()
    for token in tokens:
        if token not in VALID_STREAMS:
            print(
                f"ERROR: TAOSMD_LME_STREAMS={raw!r} contains unknown token {token!r}; "
                "valid tokens are: assembler, fts, vector",
                file=sys.stderr,
            )
            sys.exit(2)
        if token in seen:
            print(
                f"ERROR: TAOSMD_LME_STREAMS={raw!r} contains duplicate token {token!r}; "
                "valid tokens are: assembler, fts, vector",
                file=sys.stderr,
            )
            sys.exit(2)
        seen.add(token)

    # Return sorted for deterministic order (assembler, fts, vector)
    return sorted(seen)


def _num_ctx_from_env(raw: str | None) -> int:
    """Parse TAOSMD_LME_NUM_CTX, falling back to the default on a bad value.

    A malformed or empty value used to raise ValueError at module import, which
    killed the runner before it could say what was wrong. Falling back keeps the
    historical default (num_ctx unset), and the warning names the offending
    value so a mis-set window is never mistaken for a deliberate one.
    """
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        print(
            f"  WARNING: TAOSMD_LME_NUM_CTX={raw!r} is not an integer; falling "
            "back to the Ollama default (num_ctx unset).",
            file=sys.stderr,
        )
        return 0


NUM_CTX = _num_ctx_from_env(os.environ.get("TAOSMD_LME_NUM_CTX"))
_reranker = None


def _gen_options(**extra):
    """Ollama options dict, carrying num_ctx only when it was explicitly set."""
    opts = dict(extra)
    if NUM_CTX and "num_ctx" not in opts:
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

# LLM-as-judge prompts for scoring. JUDGE_PROMPT is master's prompt byte for
# byte (non-abstention questions are unchanged). ABSTENTION_JUDGE_PROMPT is used
# for abstention questions (question_id ending "_abs"), where the correct
# behaviour is to say the information is not available: the prediction is
# CORRECT when it says so and INCORRECT when it asserts a specific answer.
JUDGE_PROMPT = """You are a strict answer evaluator. Determine if the predicted answer contains the same factual information as the reference answer.

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

ABSTENTION_JUDGE_PROMPT = """You are a strict answer evaluator judging an abstention question, where the
correct behaviour is to say the information is not available.

Rules:
- The predicted answer is CORRECT if it says the information is not available
  / was not mentioned / cannot be determined (it may add that only part was
  mentioned).
- The predicted answer is INCORRECT if it asserts a specific answer to the
  question.

Reply with exactly one word: CORRECT or INCORRECT

Question: {question}
Reference answer: {gold}
Predicted answer: {predicted}

Verdict: /no_think"""


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


class JudgeError(RuntimeError):
    """The judge call failed (timeout, HTTP error): not a verdict."""


async def score_answer_llm(client, predicted: str, gold: str, question: str, question_id: str = "",
                           timeout: float = 30, raise_on_error: bool = False) -> bool:
    """Score using LLM-as-judge (official LongMemEval approach).

    For abstention questions (question_id ending in "_abs") the string filter is
    skipped and ABSTENTION_JUDGE_PROMPT is used instead of JUDGE_PROMPT: the
    prediction is CORRECT when it says the information is not available / was not
    mentioned / cannot be determined, and INCORRECT when it asserts a specific
    answer. Non-abstention questions use JUDGE_PROMPT, byte-identical to master.
    """
    is_abs = question_id.endswith("_abs")
    template = ABSTENTION_JUDGE_PROMPT if is_abs else JUDGE_PROMPT
    prompt = template.format(question=question, gold=gold, predicted=predicted)
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
            timeout=timeout,
        )
        if resp.status_code == 200:
            content = resp.json().get("message", {}).get("content", "")
            return _parse_verdict(content)
        if raise_on_error:
            raise JudgeError(f"HTTP {resp.status_code}")
    except JudgeError:
        raise
    except Exception as exc:
        if raise_on_error:
            raise JudgeError(f"{type(exc).__name__}: {exc}") from exc
    return False


def score_answer(predicted: str, gold: str) -> bool:
    """Fast substring check (used when LLM judge not available)."""
    return score_answer_substring(predicted, gold)


async def llm_answer(client, context: str, question: str, date_frame: str = "off", question_date: str | None = None) -> str:
    """Use remote LLM to generate answer from recalled context.

    Args:
        date_frame: Date framing arm ("off", "question", "sessions", "both", "labelled").
        question_date: The question's date string for the "Today is" line (question/both/labelled).
    """
    prompt = ANSWER_PROMPT.format(context=context[:CONTEXT_CHARS], question=question)
    if date_frame in ("question", "both", "labelled") and question_date:
        # Insert "Today is <date>." immediately before "Context:" line
        prompt = prompt.replace("Context:", f"Today is {question_date}.\n\nContext:", 1)
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": REMOTE_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
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


def _check_product_mode(args, reader_ctx_tokens, date_frame: str) -> None:
    """Refuse product-mode runs that would silently ignore a setting.

    Product mode builds the context only through taosmd.answer_context, so the
    legacy levers do not apply. A flag that is set but ignored would make two
    arms look different when they are the same run. Self-verify is the one
    exception: production has no verify pass, so it is forced off (see
    run_benchmark) rather than refused.
    """
    problems = reader_ctx_problem(reader_ctx_tokens)
    if getattr(args, "graph_expansion", 0):
        problems.append("--graph-expansion does not apply")
    if getattr(args, "retrieval_path", "retrieve") != "retrieve":
        problems.append("--retrieval-path does not apply")
    if getattr(args, "report_retrieval_delta", False):
        problems.append("--report-retrieval-delta does not apply")
    if date_frame != "off":
        problems.append("TAOSMD_LME_DATE_FRAME does not apply (the context always carries session dates)")
    if "TAOSMD_LME_STREAMS" in os.environ:
        problems.append("TAOSMD_LME_STREAMS does not apply")
    for flag, name in ((RERANK, "TAOSMD_RERANK"), (DECOMPOSE, "TAOSMD_DECOMPOSE")):
        if flag:
            problems.append(f"{name} does not apply")
    refuse(problems)


def _haystack_session_ids(item: dict) -> list[str]:
    """The dataset's own session ids when it has them, else ``s<index>``."""
    given = item.get("haystack_session_ids")
    sessions = item.get("haystack_sessions", [])
    if isinstance(given, list) and len(given) == len(sessions):
        return [str(g) for g in given]
    return [f"s{si}" for si in range(len(sessions))]


def gold_evidence(item: dict) -> dict:
    """What the dataset marks as the evidence: ``turns`` (a set of
    ``(session_id, turn_index)`` from ``has_answer``) and ``sessions``
    (``answer_session_ids``). Either may be empty when the item lacks it."""
    ids = _haystack_session_ids(item)
    turns = set()
    for si, session in enumerate(item.get("haystack_sessions", [])):
        for ti, turn in enumerate(session):
            if turn.get("has_answer"):
                turns.add((ids[si], ti))
    return {"turns": turns, "sessions": {str(x) for x in item.get("answer_session_ids") or []}}


def haystack_turns(item: dict) -> list[dict]:
    """The haystack as normal ingest turns: one dict per turn, in order.

    Each carries the session id, its position in the session and the session's
    own date, so a real ``taosmd.ingest`` call stores everything answer_context
    needs; the runner holds nothing the product does not.
    """
    dates = item.get("haystack_dates") or []
    ids = _haystack_session_ids(item)
    turns = []
    for si, session in enumerate(item.get("haystack_sessions", [])):
        for ti, turn in enumerate(session):
            content = turn.get("content", "")
            if not content:
                continue
            row = {
                "role": turn.get("role", "user"),
                "content": content,
                "session_id": ids[si],
                "turn_index": ti,
            }
            if si < len(dates) and dates[si]:
                row["session_date"] = dates[si]
            turns.append(row)
    return turns


async def ingest_haystack_product(item: dict, data_dir: str) -> dict:
    """Ingest one question's haystack through the normal product ingest path."""
    result = await _api.ingest(haystack_turns(item), agent=PRODUCT_AGENT, data_dir=data_dir)
    check_ingest(result, f"question {item.get('question_id', '')!r}")
    return result


async def build_product_context(
    item: dict,
    data_dir: str,
    reader_ctx_tokens: int,
    *,
    limit: int | None = None,
    neighbors: int | None = None,
) -> dict:
    """The context for one question, built only by taosmd.answer_context.

    ``limit`` / ``neighbors`` left as None use the product defaults. No
    character cut is applied anywhere on this path: the token budget in
    answer_context is the only limit.
    """
    opts = {}
    if limit is not None:
        opts["limit"] = limit
    if neighbors is not None:
        opts["neighbors"] = neighbors
    return await _answer_context(
        item["question"],
        agent=PRODUCT_AGENT,
        data_dir=data_dir,
        budget_tokens=reader_ctx_tokens,
        now=item.get("question_date"),
        **opts,
    )


class ReaderError(RuntimeError):
    """The reader call failed (timeout, HTTP error). Never scored as an answer."""


async def llm_answer_product(client, context: str, question: str, guidance: str,
                             num_ctx: int, timeout: float = 600.0) -> tuple[str, int | None]:
    """Generate the answer from an answer_context result.

    The guidance text is the instruction, the context goes in whole (no
    character cut), and the Ollama window is set to the budget the context was
    built for. Returns ``(answer, prompt_tokens)``; ``prompt_tokens`` is the
    server's own count of the prompt it read (Ollama's ``prompt_eval_count``)
    so a prompt the server truncated or a token estimate that was too low shows
    in the results. A failed call raises :class:`ReaderError` instead of
    returning an empty answer, so a timeout is never scored as a wrong answer.
    """
    prompt = PRODUCT_PROMPT.format(guidance=guidance, context=context, question=question)
    try:
        resp = await client.post(
            f"{REMOTE_LLM_URL}/api/chat",
            json={
                "model": REMOTE_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": _gen_options(num_ctx=num_ctx, temperature=0, num_predict=100),
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        body = resp.json()
    except Exception as exc:
        raise ReaderError(f"{type(exc).__name__}: {exc}") from exc
    return body.get("message", {}).get("content", ""), body.get("prompt_eval_count")


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


def _apply_date_framing(
    vector_results: list[dict],
    date_frame: str,
    haystack_dates: list[str] | None,
) -> tuple[list[dict], str]:
    """Apply date framing to vector results.

    Returns (updated_vector_results, vector_text) where vector_text includes
    the note line if applicable, or empty string if no vector results.

    The logic is shared between the all-three-streams path and the subset-streams
    path in retrieve_context().
    """
    if not (date_frame in ("sessions", "both", "labelled") and haystack_dates):
        vector_text = " ".join(r["text"] for r in vector_results if r.get("text"))
        return vector_results, vector_text

    dated_results = []
    for r in vector_results:
        sd = _hit_session_date(r)
        if sd:
            try:
                dt = datetime.strptime(sd, "%Y/%m/%d (%a) %H:%M")
                meta = r.get("metadata", {}) or {}
                inner = meta.get("metadata") if isinstance(meta, dict) else None
                user_md = inner if isinstance(inner, dict) else meta
                si = user_md.get("session", 0)
                dated_results.append((dt, si, r))
            except ValueError:
                dated_results.append((datetime.max, 0, r))
        else:
            dated_results.append((datetime.max, 0, r))

    has_parseable_date = any(dt != datetime.max for dt, _, _ in dated_results)
    if not has_parseable_date:
        vector_text = " ".join(r["text"] for r in vector_results if r.get("text"))
        return vector_results, vector_text

    dated_results.sort(key=lambda x: (x[0], x[1]))
    vector_results = [r for _, _, r in dated_results]

    if date_frame == "labelled":
        served_dates = []
        for _, _, r in dated_results:
            sd = _hit_session_date(r)
            if sd:
                try:
                    dt = datetime.strptime(sd, "%Y/%m/%d (%a) %H:%M")
                    served_dates.append(dt)
                except ValueError:
                    pass
        if served_dates:
            latest_dt = max(served_dates)
            new_results = []
            for _, _, r in dated_results:
                text = r.get("text", "")
                sd = _hit_session_date(r)
                is_latest = False
                if sd:
                    try:
                        if datetime.strptime(sd, "%Y/%m/%d (%a) %H:%M") == latest_dt:
                            is_latest = True
                    except ValueError:
                        pass
                if sd and is_latest:
                    text = text.replace(
                        f"[Session date: {sd}]",
                        f"[Session date: {sd}, most recent]",
                        1
                    )
                elif sd:
                    text = text.replace(
                        f"[Session date: {sd}]",
                        f"[Session date: {sd}, historical]",
                        1
                    )
                new_r = dict(r)
                new_r["text"] = text
                new_results.append(new_r)
            vector_results = new_results
            note_line = ("Memories below are in chronological order, oldest first. "
                         "Where memories disagree, the most recent one is current and the earlier ones are historical.")
        else:
            note_line = "Memories below are in chronological order, oldest first."
    else:
        note_line = "Memories below are in chronological order, oldest first."

    vector_text = note_line + " " + " ".join(r["text"] for r in vector_results if r.get("text"))
    return vector_results, vector_text


async def retrieve_context(
    question: str,
    kg,
    archive,
    vmem,
    llm_client=None,
    graph_expansion: int = 0,
    retrieval_path: str = "retrieve",
    streams: list[str] | None = None,
    date_frame: str = "off",
    haystack_dates: list[str] | None = None,
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
        streams: Subset of {"assembler", "fts", "vector"} to enable.
            Defaults to all three. Order is fixed: assembler, fts, vector.
            Disabled streams are not called and contribute nothing.
        date_frame: Date framing arm ("off", "question", "sessions", "both", "labelled").
        haystack_dates: List of session dates for chronological ordering (sessions/both/labelled).
    """
    if streams is None:
        streams = ["assembler", "fts", "vector"]

    # Byte-identical fallback for the default all-three-streams case:
    # the original code always returned
    #   ctx["context"] + " " + archive_text + " " + vector_text
    # with two trailing spaces if both archive_text and vector_text are empty,
    # and archive_text preserving its leading spaces per FTS result.
    # Replicate that exact behaviour when all three streams are enabled.
    all_three = streams == ["assembler", "fts", "vector"]

    if all_three:
        assembler = ContextAssembler(kg=kg, archive=archive)
        ctx = await assembler.assemble(
            query=question,
            depth="auto",
            max_total_tokens=ASSEMBLE_TOKENS,
        )
        assembler_ctx = ctx["context"]

        archive_text = ""
        for term in question.split()[:5]:
            if len(term) > 3:
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

        # Date framing: sessions/both/labelled reorder vector results chronologically
        # and add a note line. labelled also rewrites prefixes at read time.
        _, vector_text = _apply_date_framing(vector_results, date_frame, haystack_dates)

        return assembler_ctx + " " + archive_text + " " + vector_text

    parts = []

    if "assembler" in streams:
        assembler = ContextAssembler(kg=kg, archive=archive)
        ctx = await assembler.assemble(
            query=question,
            depth="auto",
            max_total_tokens=ASSEMBLE_TOKENS,
        )
        parts.append(ctx["context"])

    if "fts" in streams:
        archive_text = ""
        for term in question.split()[:5]:
            if len(term) > 3:
                try:
                    fts_results = await archive.search_fts(term, limit=FTS_LIMIT)
                    for r in fts_results:
                        archive_text += " " + r.get("data_json", "") + " " + r.get("summary", "")
                except Exception:
                    pass
        if archive_text:
            parts.append(archive_text.strip())

    if "vector" in streams:
        vector_results = await retrieve_vector_results(
            question, kg, vmem,
            llm_client=llm_client,
            graph_expansion=graph_expansion,
            retrieval_path=retrieval_path,
        )

        # Date framing applies to vector stream in BOTH paths (all-three and subset)
        _, vector_text = _apply_date_framing(vector_results, date_frame, haystack_dates)

        if vector_text:
            parts.append(vector_text)

    return " ".join(parts)


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


def _default_out_dir() -> str:
    """Directory the runner writes results into when --out is not given."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _prepare_out_dir(out_path: str) -> str:
    """Create the result directory and prove it is writable. Returns it.

    Called before the question loop so an unusable --out costs nothing. The
    probe is a real create-and-delete rather than an os.access() check, because
    access() answers about permission bits and not about read-only mounts,
    full filesystems or a parent component that is a regular file.
    """
    parent = os.path.dirname(os.path.abspath(out_path)) if out_path else _default_out_dir()
    try:
        os.makedirs(parent, exist_ok=True)
        fd, probe = tempfile.mkstemp(dir=parent, prefix=".out-probe-")
        os.close(fd)
        os.unlink(probe)
    except OSError as exc:
        print(
            f"  ERROR: results cannot be written to {parent}: {exc}. Fix --out "
            "before starting the run, because a path that only fails at write "
            "time discards every question the run has already answered.",
            file=sys.stderr,
        )
        sys.exit(1)
    return parent


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
    writer_url = None
    writer_model = "default"
    context_mode = "legacy"
    reader_ctx_tokens = None
    answer_limit = None
    answer_neighbors = None
    reader_timeout = 600.0
    product_env = None
    answer_failures = 0
    if args is not None:
        limit = args.limit
        question_type = args.type
        use_llm = args.llm
        graph_expansion = args.graph_expansion
        retrieval_path = args.retrieval_path
        report_retrieval_delta = args.report_retrieval_delta
        out_path = args.out
        writer_url = getattr(args, "writer_url", None)
        writer_model = getattr(args, "writer_model", None)
        context_mode = _resolve_context_mode(args)
        reader_ctx_tokens = getattr(args, "reader_ctx_tokens", None)
        answer_limit = getattr(args, "answer_limit", None)
        answer_neighbors = getattr(args, "answer_neighbors", None)
        reader_timeout = float(getattr(args, "reader_timeout", None) or 600.0)

    # Parse TAOSMD_LME_STREAMS (exits on invalid value)
    streams = _parse_streams_env()

    # Date framing arm (E-036)
    date_frame = _date_frame_from_env()
    self_verify = SELF_VERIFY
    if context_mode == "product":
        _check_product_mode(args, reader_ctx_tokens, date_frame)
        if self_verify:
            print("  NOTE: TAOSMD_SELF_VERIFY is ignored in product mode (production has no verify pass).")
            self_verify = False

    print("=" * 70)
    print("LongMemEval Benchmark -- taOSmd")
    print("=" * 70)
    print(
        f"  retrieval_path={retrieval_path}  graph_expansion={graph_expansion}  "
        f"num_ctx={NUM_CTX or 'ollama-default'}  streams={streams}  date_frame={date_frame}"
    )
    if context_mode == "product":
        print(f"  context=product  reader_ctx_tokens={reader_ctx_tokens}  "
              f"answer_limit={answer_limit or 'default'}  neighbors={answer_neighbors if answer_neighbors is not None else 'default'}")

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

    # The delta builds the legacy context and compares it against the context
    # the run itself used. On the legacy path those are the same assembly, so
    # the comparison is legacy-against-legacy: identical by construction, and
    # a 100% figure that measures nothing. Refuse rather than report it.
    if report_retrieval_delta and retrieval_path == "legacy":
        print(
            "  ERROR: --report-retrieval-delta compares the run's context "
            "against the legacy context, so with --retrieval-path legacy it "
            "compares legacy to legacy and is identical by construction. Drop "
            "--retrieval-path legacy (the wired path is the default) to measure "
            "the two paths against each other.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Prove the result path is writable now, not after the run. A --out into a
    # directory that cannot be created used to fail at the final write, by which
    # point every question had been answered and the results were unrecoverable.
    _prepare_out_dir(out_path)

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
    method_llm_count = 0
    method_regex_count = 0
    total_turns = 0
    abstention_correct = 0
    abstention_total = 0

    # Create LLM client if needed
    llm_client = None
    if use_llm:
        import httpx as _httpx
        llm_client = _httpx.AsyncClient(timeout=30)

    # Create writer client for ingest-time fact extraction (E-050)
    writer_client = None
    if writer_url:
        import httpx as _httpx
        writer_client = _httpx.AsyncClient(timeout=30)

    for i, item in enumerate(dataset):
        qtype = item["question_type"]
        question = item["question"]
        gold_answer = item["answer"]
        sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])
        question_date = item.get("question_date")

        # Validate dates for non-off arms (exits 1 on failure)
        _validate_dates(
            item, "both" if context_mode == "product" else date_frame,
            question_id=item.get("question_id"),
        )
        product_meta = None
        answer_error = None

        if context_mode == "product":
            product_dir = new_product_dir()
            try:
                t0 = time.time()
                await ingest_haystack_product(item, product_dir)
                if product_env is None:
                    product_env = await product_environment(product_dir)
                    print(f"  product environment: {product_env}")
                    require_healthy(product_env)
                ingest_time = time.time() - t0
                t1 = time.time()
                prod = await build_product_context(
                    item, product_dir, reader_ctx_tokens,
                    limit=answer_limit, neighbors=answer_neighbors,
                )
                query_time = time.time() - t1
            finally:
                await release_product_stores(product_dir)
            full_context = prod["context"]
            delta = None
            product_meta = _product_meta(prod)
            gold = gold_evidence(item)
            product_meta["evidence"] = {
                "turns_included": count_in(prod["included"], gold["turns"]),
                "turns_dropped": count_in(prod["dropped"], gold["turns"]),
                "turns_total": len(gold["turns"]),
                "sessions_included": len(gold["sessions"] & {b["session_id"] for b in prod["included"]}),
                "sessions_dropped": len(gold["sessions"] & {b["session_id"] for b in prod["dropped"]}),
                "sessions_total": len(gold["sessions"]),
            }
        else:
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
                session_date = haystack_dates[si] if si < len(haystack_dates) else ""
                date_prefix = f"[Session date: {session_date}]\n" if (date_frame in ("sessions", "both", "labelled") and session_date) else ""
                # Build session-level text blocks for embedding
                session_text = ""
                for turn in session:
                    content = turn.get("content", "")
                    role = turn.get("role", "user")
                    if content:
                        # Extract facts into KG
                        if writer_url:
                            result = await process_conversation_turn(
                                content, agent_name="assistant" if role == "assistant" else None,
                                kg=kg, archive=archive, source="longmemeval",
                                llm_url=writer_url,
                                http_client=writer_client,
                                extraction_model=writer_model,
                            )
                            method = result.get("method", "regex")
                            if method == "llm":
                                method_llm_count += 1
                            else:
                                method_regex_count += 1
                            total_turns += 1
                        else:
                            result = await process_conversation_turn(
                                content, agent_name="assistant" if role == "assistant" else None,
                                kg=kg, archive=archive, source="longmemeval",
                            )
                            total_turns += 1
                            method_regex_count += 1
                        # Archive raw content (with date prefix for sessions/both/labelled)
                        archived_content = date_prefix + content if date_prefix else content
                        await archive.record(
                            "conversation",
                            {"role": role, "content": archived_content},
                            summary=archived_content[:80],
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
                        # Add date prefix to each chunk for sessions/both/labelled
                        prefixed_chunk = date_prefix + chunk if date_prefix else chunk
                        metadata = {"session": si}
                        if date_frame in ("sessions", "both", "labelled") and session_date:
                            metadata["session_date"] = session_date
                        await vmem.add(prefixed_chunk, metadata=metadata)

            ingest_time = time.time() - t0

            # Query taOSmd: assembled context + raw archive FTS + the vector stage,
            # the last of which now runs through taosmd.retrieval.retrieve().
            t1 = time.time()
            full_context = await retrieve_context(
                question, kg, archive, vmem,
                llm_client=llm_client,
                graph_expansion=graph_expansion,
                retrieval_path=retrieval_path,
                streams=streams,
                date_frame=date_frame,
                haystack_dates=haystack_dates,
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
                    streams=streams,
                    date_frame=date_frame,
                    haystack_dates=haystack_dates,
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

        question_id = item.get("question_id", "")
        is_abstention = question_id.endswith("_abs")

        async def _judge_checked(client, ans, gold, q, qid):
            """Judge call; in product mode a failed call is counted, not scored."""
            nonlocal answer_failures
            if context_mode != "product":
                return await score_answer_llm(client, ans, gold, q, question_id=qid)
            try:
                return await score_answer_llm(client, ans, gold, q, question_id=qid,
                                              timeout=reader_timeout, raise_on_error=True)
            except JudgeError as exc:
                answer_failures += 1
                product_meta["judge_error"] = str(exc)
                print(f"    [JUDGE FAILED] q={i}: {exc}", file=sys.stderr)
                return False

        if use_llm and llm_client is not None:
            t_llm = time.time()
            # Step 1: LLM generates answer from recalled context
            if context_mode == "product":
                try:
                    answer, prompt_tokens = await llm_answer_product(
                        llm_client, full_context, question, prod["guidance"],
                        reader_ctx_tokens, timeout=reader_timeout)
                    product_meta["prompt_eval_count"] = prompt_tokens
                except ReaderError as exc:
                    answer, answer_error = "", str(exc)
                    product_meta["answer_error"] = answer_error
                    answer_failures += 1
                    print(f"    [READER FAILED] q={i}: {answer_error}", file=sys.stderr)
            else:
                answer = await llm_answer(llm_client, full_context, question, date_frame=date_frame, question_date=question_date)
                if self_verify:
                    answer = await self_verify_answer(llm_client, full_context, question, answer)
            # Step 2: LLM judges whether answer matches gold (official eval method)
            if answer_error:
                correct = False
            elif is_abstention:
                # Abstention questions: never apply the "I don't know" string
                # filter, because the correct behaviour IS to say the information
                # is unavailable. Always send to the judge with the abstention
                # prompt so a correct abstention is scored correct.
                correct = await _judge_checked(llm_client, answer, gold_answer, question, question_id)
            elif answer and not any(idk in answer.lower() for idk in ("i don't know", "i do not know", "i'm sorry", "not in the context", "does not contain", "no information")):
                correct = await _judge_checked(llm_client, answer, gold_answer, question, question_id)
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

        if is_abstention:
            abstention_total += 1
            if correct:
                abstention_correct += 1

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
            "context_chars": len(full_context),
            "retrieval_delta": delta,
            **({"product": product_meta} if product_meta else {}),
        })

        status = "✓" if correct else "✗"
        print(f"  [{i+1:3d}/{len(dataset)}] {status} {qtype:25s} | ingest:{ingest_time:.1f}s query:{query_time:.3f}s | {question[:50]}")

        if context_mode != "product":
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

    print("\n  By question type:")
    for qtype, data in sorted(results_by_type.items()):
        pct = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"    {qtype:30s} {data['correct']:3d}/{data['total']:<3d} ({pct:.1f}%)")

    print("\n  Comparison:")
    print("    MemPalace (raw verbatim):     96.6%")
    print("    SuperMemory:                  81.6%")
    print("    GPT-4o (full context):        ~70%")
    print(f"    taOSmd (Pi NPU, no cloud):    {overall:.1f}%")
    print(f"{'='*70}")

    if llm_client:
        await llm_client.aclose()

    if writer_client:
        await writer_client.aclose()

    if total_questions:
        if not out_path:
            out_dir = _default_out_dir()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"longmemeval_{int(time.time())}.json")
        result_doc = {
            "question_type": question_type,
            "limit": limit,
            "generator": REMOTE_LLM_MODEL,
            "judge": JUDGE_MODEL,
            "rerank": RERANK,
            "decompose": DECOMPOSE,
            "self_verify": self_verify,
            "assemble_tokens": ASSEMBLE_TOKENS,
            "retrieve_limit": RETRIEVE_LIMIT,
            "fts_limit": FTS_LIMIT,
            "context_chars": CONTEXT_CHARS,
            "num_ctx": NUM_CTX,
            "retrieval_path": retrieval_path,
            "graph_expansion": graph_expansion,
            "streams": streams,
            "date_frame": date_frame,
            "context_mode": context_mode,
            "reader_ctx_tokens": reader_ctx_tokens,
            "answer_limit": answer_limit,
            "answer_neighbors": answer_neighbors,
            "reader_timeout": reader_timeout,
            "product_environment": product_env,
            "answer_failures": answer_failures,
            "retrieval_delta": delta_summary,
            "metrics": {
                "n": total_questions,
                "correct": total_correct,
                "accuracy": overall,
                "by_type": results_by_type,
                "abstention_correct": abstention_correct,
                "abstention_total": abstention_total,
            },
            "results": all_results,
            "writer": {
                "url_set": bool(writer_url),
                "model": writer_model if writer_url else None,
                "turns": total_turns,
                "method_llm": method_llm_count,
                "method_regex": method_regex_count,
            },
        }
        with open(out_path, "w") as f:
            json.dump(result_doc, f, indent=2)
        print(f"  results -> {out_path}")
    print(f"  writer: model={writer_model or 'default'} llm={method_llm_count} regex={method_regex_count}")

    if answer_failures:
        print(
            f"  ERROR: {answer_failures} reader calls failed (timeout or HTTP error) and were "
            "scored incorrect; the run is not a valid measurement. See answer_error per question.",
            file=sys.stderr,
        )
        sys.exit(1)

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
    parser.add_argument(
        "--writer-url",
        default="",
        help="OpenAI-compatible base URL (the extractor POSTs to <url>/v1/chat/completions); empty keeps regex-only path.",
    )
    parser.add_argument(
        "--writer-model",
        default="default",
        help="Model name for ingest-time fact extraction; used with --writer-url.",
    )
    parser.add_argument(
        "--context",
        default=None,
        choices=CONTEXT_MODES,
        help="'legacy' (default; or env TAOSMD_CONTEXT_MODE) builds the context "
             "the runner always built. 'product' ingests the haystack through "
             "taosmd.ingest and builds the context only with "
             "taosmd.answer_context: no character cut, the guidance text is "
             "the instruction. Needs --reader-ctx-tokens.",
    )
    parser.add_argument(
        "--reader-ctx-tokens",
        type=int,
        default=None,
        help="Reader model context window in tokens; the answer_context budget "
             "and the Ollama num_ctx. Required with --context product.",
    )
    parser.add_argument(
        "--reader-timeout",
        type=float,
        default=None,
        help="Product mode: seconds to wait for the reader per question (default 600). "
             "A failed or timed-out call is counted in answer_failures and the run "
             "exits 1; it is never scored as a wrong answer.",
    )
    parser.add_argument(
        "--answer-limit",
        type=int,
        default=None,
        help="Product mode: hits fetched per question (default: the product default).",
    )
    parser.add_argument(
        "--answer-neighbors",
        type=int,
        default=None,
        help="Product mode: neighbouring turns either side of each hit (default: the product default).",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(args=args))


if __name__ == "__main__":
    main()
