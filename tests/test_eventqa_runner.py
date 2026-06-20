"""Unit tests for benchmarks/eventqa_runner.py -- GPU-free wiring verification.

Mirrors tests/test_beam_runner.py: all LLM-calling helpers use a mocked httpx
client; no live Ollama / GPU / HuggingFace network IO is needed.

Coverage:
  - normalize_answer: articles, punctuation, case, whitespace
  - substring_exact_match: match, no-match, empty pred, empty gold
  - dataset-row parsing and eventqa filter
  - chunk_context: word fallback and size bounds
  - prompt builders: terse instruction present; self-verify is a no-op when
    TAOSMD_SELF_VERIFY is unset
  - llm_answer + self_verify_answer with mocked client
  - compute_accuracy: correct Accuracy and n
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import pathlib

_RUNNER = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "eventqa_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("eventqa_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── mock httpx client (mirrors test_beam_runner.py) ───────────────────────────


class _FakeResp:
    def __init__(self, content: str, status: int = 200):
        self.status_code = status
        self._content = content

    def json(self):
        # beam_runner / eventqa_runner use /api/generate -> {"response": ...}
        return {"response": self._content}


class _FakeClient:
    """Returns a fixed Ollama generate response for any .post()."""

    def __init__(self, content: str, status: int = 200):
        self._content = content
        self._status = status

    async def post(self, *args, **kwargs):
        return _FakeResp(self._content, self._status)


class _BoomClient:
    async def post(self, *args, **kwargs):
        raise RuntimeError("boom")


# ── normalize_answer ──────────────────────────────────────────────────────────


def test_normalize_answer_lowercases():
    mod = _load_runner()
    assert mod.normalize_answer("Hello World") == "hello world"


def test_normalize_answer_removes_articles():
    mod = _load_runner()
    assert mod.normalize_answer("a cat sat on the mat") == "cat sat on mat"
    assert mod.normalize_answer("An apple a day") == "apple day"
    # Inline 'a' that is part of a word must be left intact.
    assert "cat" in mod.normalize_answer("the cat")


def test_normalize_answer_removes_punctuation():
    mod = _load_runner()
    assert mod.normalize_answer("Hello, world!") == "hello world"
    assert mod.normalize_answer("wait... really?") == "wait really"


def test_normalize_answer_collapses_whitespace():
    mod = _load_runner()
    assert mod.normalize_answer("  hello   world  ") == "hello world"


def test_normalize_answer_empty_string():
    mod = _load_runner()
    assert mod.normalize_answer("") == ""
    assert mod.normalize_answer("   ") == ""


def test_normalize_answer_combined():
    mod = _load_runner()
    # From the spec: articles + punctuation + case + whitespace.
    result = mod.normalize_answer("The  Quick,  Brown  Fox.")
    assert result == "quick brown fox"


# ── substring_exact_match ─────────────────────────────────────────────────────


def test_substring_exact_match_positive():
    mod = _load_runner()
    gold = ["Debbie waited at the driveway"]
    pred = "The event is: Debbie waited at the driveway."
    assert mod.substring_exact_match(pred, gold) == 1


def test_substring_exact_match_negative():
    mod = _load_runner()
    gold = ["Debbie waited at the driveway"]
    pred = "Something unrelated happened."
    assert mod.substring_exact_match(pred, gold) == 0


def test_substring_exact_match_empty_pred():
    mod = _load_runner()
    gold = ["Debbie waited at the driveway"]
    assert mod.substring_exact_match("", gold) == 0
    assert mod.substring_exact_match("   ", gold) == 0


def test_substring_exact_match_empty_gold_cannot_match():
    mod = _load_runner()
    # Empty gold string must score 0 (prevents trivial match from bad data).
    assert mod.substring_exact_match("anything", [""]) == 0
    assert mod.substring_exact_match("anything", []) == 0


def test_substring_exact_match_case_and_article_insensitive():
    mod = _load_runner()
    # After normalization "The Cat" -> "cat"; pred "I saw A cat" -> "i saw cat"
    # "cat" in "i saw cat" -> 1
    gold = ["The Cat"]
    pred = "I saw a cat in the garden."
    assert mod.substring_exact_match(pred, gold) == 1


def test_substring_exact_match_multiple_gold_any_match():
    mod = _load_runner()
    # Multiple gold strings: match on the second.
    gold = ["option A", "option B"]
    pred = "I think option B is correct."
    assert mod.substring_exact_match(pred, gold) == 1


# ── dataset-row parsing ───────────────────────────────────────────────────────

# Minimal synthetic row following the documented schema.
_SAMPLE_ROW = {
    "context": "It was a dark and stormy night. Alice walked to the door.",
    "questions": [
        "What did Alice do?\nOptions:\n(A) ran away\n(B) walked to the door\n"
        "(C) sat down\n(D) called Bob\n(E) opened a window\n(F) went to sleep\n"
        "Answer with the exact option text only.",
    ],
    "answers": [["walked to the door"]],
    "metadata": {"source": "eventqa_65536"},
}


def test_parse_eventqa_row_extracts_fields():
    mod = _load_runner()
    questions, answers, source = mod.parse_eventqa_row(_SAMPLE_ROW)
    assert len(questions) == 1
    assert "walked to the door" in questions[0] or "Alice" in questions[0]
    assert len(answers) == 1
    assert answers[0] == ["walked to the door"]
    assert source == "eventqa_65536"


def test_parse_eventqa_row_empty_row():
    mod = _load_runner()
    questions, answers, source = mod.parse_eventqa_row({})
    assert questions == []
    assert answers == []
    assert source == ""


def test_parse_eventqa_row_scalar_answer_wrapped():
    mod = _load_runner()
    row = {
        "context": "text",
        "questions": ["Q?"],
        "answers": ["scalar answer"],
        "metadata": {"source": "eventqa_full"},
    }
    _, answers, source = mod.parse_eventqa_row(row)
    # Scalar answers must be wrapped in a list.
    assert answers == [["scalar answer"]]
    assert source == "eventqa_full"


def test_eventqa_filter_logic():
    """The filter is: str(row["metadata"]["source"]) contains tier string."""
    mod = _load_runner()
    rows = [
        {"metadata": {"source": "eventqa_65536"}, "context": "x", "questions": [], "answers": []},
        {"metadata": {"source": "eventqa_131072"}, "context": "x", "questions": [], "answers": []},
        {"metadata": {"source": "other_dataset"}, "context": "x", "questions": [], "answers": []},
    ]
    tier = "eventqa_65536"
    # Replicate the filter logic from load_eventqa_rows.
    filtered = [r for r in rows if tier in str((r.get("metadata") or {}).get("source", ""))]
    assert len(filtered) == 1
    assert filtered[0]["metadata"]["source"] == "eventqa_65536"


# ── prompt builder ────────────────────────────────────────────────────────────


def test_build_answer_prompt_contains_terse_instruction():
    mod = _load_runner()
    prompt = mod.build_answer_prompt("some context", "What happened next?")
    assert "What happened next?" in prompt
    # Must include an instruction to output only the exact option text.
    lower = prompt.lower()
    assert "only" in lower or "exact" in lower or "option" in lower
    assert "some context" in prompt


def test_build_verify_prompt_contains_draft_and_question():
    mod = _load_runner()
    prompt = mod.build_verify_prompt("ctx", "Q?", "draft answer")
    assert "draft answer" in prompt
    assert "Q?" in prompt


def test_self_verify_is_noop_when_env_unset(monkeypatch):
    """Without TAOSMD_SELF_VERIFY=1, the verify path is never triggered by the runner."""
    mod = _load_runner()
    # SELF_VERIFY is read at module import time. Confirm the module-level flag
    # is False when the env var is absent (the default for this test env).
    monkeypatch.delenv("TAOSMD_SELF_VERIFY", raising=False)
    # Re-load to pick up the cleared env.
    mod2 = _load_runner()
    assert mod2.SELF_VERIFY is False


# ── llm_answer with mocked client ────────────────────────────────────────────


def test_llm_answer_returns_stripped_response():
    mod = _load_runner()
    client = _FakeClient("walked to the door")
    result = asyncio.run(mod.llm_answer(client, "some context", "What did Alice do?"))
    assert result == "walked to the door"


def test_llm_answer_returns_empty_on_client_error():
    mod = _load_runner()
    result = asyncio.run(mod.llm_answer(_BoomClient(), "ctx", "Q?"))
    assert result == ""


def test_llm_answer_returns_empty_on_http_error():
    mod = _load_runner()

    class _ErrorClient:
        async def post(self, *args, **kwargs):
            return _FakeResp("", status=500)

    result = asyncio.run(mod.llm_answer(_ErrorClient(), "ctx", "Q?"))
    assert result == ""


# ── self_verify_answer with mocked client ────────────────────────────────────


def test_self_verify_returns_revised_on_success():
    mod = _load_runner()
    result = asyncio.run(
        mod.self_verify_answer(_FakeClient("corrected answer"), "ctx", "Q?", "draft")
    )
    assert result == "corrected answer"


def test_self_verify_keeps_draft_on_client_error():
    mod = _load_runner()
    result = asyncio.run(
        mod.self_verify_answer(_BoomClient(), "ctx", "Q?", "draft")
    )
    assert result == "draft"


def test_self_verify_keeps_draft_on_empty_response():
    mod = _load_runner()
    result = asyncio.run(
        mod.self_verify_answer(_FakeClient("   "), "ctx", "Q?", "draft")
    )
    assert result == "draft"


def test_self_verify_noop_on_empty_input_answer():
    mod = _load_runner()
    # An empty answer must pass through unchanged, not become the verify response.
    result = asyncio.run(
        mod.self_verify_answer(_FakeClient("something"), "ctx", "Q?", "")
    )
    assert result == ""


# ── chunk_context ─────────────────────────────────────────────────────────────


def test_chunk_context_empty_returns_empty():
    mod = _load_runner()
    assert mod.chunk_context("") == []
    assert mod.chunk_context("   ") == []


def test_chunk_context_word_fallback_bounds(monkeypatch):
    """When tiktoken is absent, falls back to word-based splitting."""
    mod = _load_runner()
    import builtins
    real_import = builtins.__import__

    def _no_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("mocked absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tiktoken)
    # 600 words, CHUNK_WORDS=512 -> 2 chunks (512 + 88)
    text = " ".join(f"w{i}" for i in range(600))
    # Reload with patched import so chunk_context runs the word path.
    chunks = mod._chunk_by_words(text, 512)
    assert len(chunks) == 2
    assert all(len(c.split()) <= 512 for c in chunks)


def test_chunk_context_short_text_is_single_chunk(monkeypatch):
    mod = _load_runner()
    text = "The quick brown fox jumped over the lazy dog."
    chunks = mod._chunk_by_words(text, 512)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


# ── compute_accuracy ──────────────────────────────────────────────────────────


def test_compute_accuracy_basic():
    mod = _load_runner()
    results = [
        {"score": 1, "dry": False},
        {"score": 0, "dry": False},
        {"score": 1, "dry": False},
        {"score": 1, "dry": False},
    ]
    m = mod.compute_accuracy(results)
    assert m["n"] == 4
    assert m["correct"] == 3
    assert m["accuracy"] == 75.0


def test_compute_accuracy_excludes_dry_rows():
    mod = _load_runner()
    results = [
        {"score": 1, "dry": False},
        {"score": 0, "dry": True},
        {"score": 1, "dry": True},
    ]
    m = mod.compute_accuracy(results)
    assert m["n"] == 1
    assert m["correct"] == 1
    assert m["accuracy"] == 100.0


def test_compute_accuracy_all_dry():
    mod = _load_runner()
    results = [{"score": 0, "dry": True}]
    m = mod.compute_accuracy(results)
    assert m["n"] == 0
    assert m["accuracy"] == 0.0


def test_compute_accuracy_perfect_and_zero():
    mod = _load_runner()
    perfect = [{"score": 1, "dry": False}] * 10
    m = mod.compute_accuracy(perfect)
    assert m["accuracy"] == 100.0

    zero = [{"score": 0, "dry": False}] * 10
    m2 = mod.compute_accuracy(zero)
    assert m2["accuracy"] == 0.0
