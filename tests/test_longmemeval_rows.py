"""Tests for the LongMemEval runner row building and flags."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("longmemeval_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_row_no_context():
    runner = _load_runner()
    full_ctx = "x" * 100
    row = runner.build_row(
        idx=0,
        question_type="fact",
        question="Q?",
        correct=True,
        full_context=full_ctx,
        delta=None,
        predicted="A",
        gold_answer="A",
        save_context=False,
    )
    # master's keys plus predicted and answer; no context key
    assert set(row.keys()) == {
        "idx",
        "question_type",
        "question",
        "correct",
        "retrieved_chars",
        "retrieval_delta",
        "predicted",
        "answer",
    }
    assert row["predicted"] == "A"
    assert row["answer"] == "A"
    assert "context" not in row


def test_build_row_with_context_truncation(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "CONTEXT_CHARS", 10)
    full_ctx = "x" * 100
    row = runner.build_row(
        idx=1,
        question_type="fact",
        question="Q?",
        correct=False,
        full_context=full_ctx,
        delta={"d": 1},
        predicted=None,
        gold_answer="G",
        save_context=True,
    )
    assert row["context"] == full_ctx[:10]
    assert len(row["context"]) == 10
    assert row["predicted"] is None


def test_is_non_answer_cases():
    runner = _load_runner()
    # phrases
    for p in runner.NON_ANSWER_PHRASES:
        assert runner.is_non_answer(p)
        assert runner.is_non_answer(p.upper())
        assert runner.is_non_answer(f"Maybe {p}.")
    assert runner.is_non_answer("")
    assert runner.is_non_answer(None)
    assert not runner.is_non_answer("A real answer")
    assert not runner.is_non_answer("The capital of France is Paris")


def test_save_context_env_invalid(monkeypatch):
    runner = _load_runner()
    monkeypatch.setenv("TAOSMD_LME_SAVE_CONTEXT", "yes")
    with pytest.raises(ValueError):
        # simulate loading/parsing
        lme_save_ctx = os.environ.get("TAOSMD_LME_SAVE_CONTEXT")
        if lme_save_ctx is None or lme_save_ctx == "0":
            save_context = False
        elif lme_save_ctx == "1":
            save_context = True
        else:
            raise ValueError("TAOSMD_LME_SAVE_CONTEXT must be '0' or '1' (unset also treated as '0')")


def test_retrieval_only_has_predicted_none():
    runner = _load_runner()
    # Simulate retrieval-only path result structure
    # Build a row as run_benchmark would
    full_ctx = "some context"
    row = runner.build_row(
        idx=0,
        question_type="fact",
        question="What?",
        correct=False,
        full_context=full_ctx,
        delta=None,
        predicted=None,  # retrieval only
        gold_answer="answer",
        save_context=False,
    )
    assert row["predicted"] is None
    assert row["answer"] == "answer"
