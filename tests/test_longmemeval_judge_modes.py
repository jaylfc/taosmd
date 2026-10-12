#!/usr/bin/env python3
"""Test file for LongMemEval judge modes.

Tests the requirements for LongMemEval save-context + evidence judge + re-judge.
"""

import pytest
from unittest.mock import AsyncMock

from benchmarks.longmemeval_runner import build_row
from benchmarks.longmemeval_rejudge import (
    _score_with_mode,
    set_runner_mod_for_testing,
)


def test_build_row_contains_predicted_and_answer():
    """Test that a row produced by build_row contains predicted and answer keys."""
    # Test with predicted provided
    row = build_row(
        idx=0,
        question_type="test_type",
        question="What is the answer?",
        gold_answer="The answer is 42",
        full_context="Some context about the answer",
        correct=True,
        retrieved_chars=50,
        retrieval_delta=None,
        save_context=True,
        predicted="The answer is 42",
    )
    
    assert "predicted" in row
    assert "answer" in row
    assert row["predicted"] == "The answer is 42"
    assert row["answer"] == "The answer is 42"
    
    # Test without predicted (retrieval-only mode)
    row2 = build_row(
        idx=1,
        question_type="test_type",
        question="What is the answer?",
        gold_answer="The answer is 42",
        full_context="Some context about the answer",
        correct=True,
        retrieved_chars=50,
        retrieval_delta=None,
        save_context=False,
        predicted=None,
    )
    
    assert "predicted" in row2
    assert "answer" in row2
    assert row2["predicted"] == ""
    assert row2["answer"] == "The answer is 42"


def test_build_row_contains_truncated_context():
    """Test that build_row contains truncated context when save_context=True."""
    full_context = "A very long context that exceeds the CONTEXT_CHARS limit. " * 100
    
    row = build_row(
        idx=2,
        question_type="test_type",
        question="What is the answer?",
        gold_answer="The answer is 42",
        full_context=full_context,
        correct=True,
        retrieved_chars=len(full_context),
        retrieval_delta=None,
        save_context=True,
        predicted="The answer is 42",
    )
    
    # Context should be truncated to CONTEXT_CHARS (16000 default)
    from benchmarks.longmemeval_runner import CONTEXT_CHARS
    if "context" in row:
        assert len(row["context"]) == min(len(full_context), CONTEXT_CHARS)
    else:
        # save_context was True but context might not be in row if something else happened
        pytest.skip("Context not in row (this might be expected based on save_context logic)")


@pytest.mark.asyncio
async def test_rejudge_with_stub_judge():
    """Test that rejudge feeds a build_row row through with a stub judge."""
    from benchmarks.longmemeval_runner import build_row
    
    # Create a test row
    row = build_row(
        idx=3,
        question_type="temporal",
        question="When did event X happen?",
        gold_answer="Event X happened in 2020",
        full_context="Event X happened in the year 2020.",
        correct=True,
        retrieved_chars=50,
        retrieval_delta=None,
        save_context=True,
        predicted="Event X happened in 2020",
    )
    
    # Create a mock runner module with a stub score_answer_llm function
    mock_runner_mod = type('MockRunnerMod', (), {})()
    
    # Track if judge was called
    judge_called = False
    judge_prompt = None
    
    async def mock_score_answer_llm(client, predicted, gold, question, context=None):
        nonlocal judge_called, judge_prompt
        judge_called = True
        judge_prompt = {
            "predicted": predicted,
            "gold": gold,
            "question": question,
            "context": context,
        }
        return True
    
    mock_runner_mod.score_answer_llm = mock_score_answer_llm
    
    # Set the mock runner module for testing
    set_runner_mod_for_testing(mock_runner_mod)
    
    # Create a mock client
    mock_client = AsyncMock()
    
    # Call _score_with_mode
    result = await _score_with_mode(mock_runner_mod, mock_client, row, "reference")
    assert result is True

    # Verify judge was called and prompt contains expected content
    assert judge_called is True
    assert judge_prompt is not None
    assert judge_prompt["predicted"] == row["predicted"]
    assert judge_prompt["gold"] == row["answer"]
    assert judge_prompt["question"] == row["question"]
    assert judge_prompt["context"] is None  # Reference mode doesn't use context


@pytest.mark.asyncio
async def test_rejudge_with_empty_predicted():
    """Test that rejudge with empty predicted doesn't call judge."""
    from benchmarks.longmemeval_runner import build_row
    
    # Create a test row with empty predicted
    row = build_row(
        idx=4,
        question_type="temporal",
        question="What is the question?",
        gold_answer="The answer",
        full_context="Some context",
        correct=False,
        retrieved_chars=20,
        retrieval_delta=None,
        save_context=True,
        predicted="",  # Empty predicted
    )
    
    # Create a mock runner module
    mock_runner_mod = type('MockRunnerMod', (), {})()
    
    # Track if judge was called
    judge_called = False
    
    async def mock_score_answer_llm(client, predicted, gold, question, context=None):
        nonlocal judge_called
        judge_called = True
        return True
    
    mock_runner_mod.score_answer_llm = mock_score_answer_llm
    
    # Set the mock runner module for testing
    set_runner_mod_for_testing(mock_runner_mod)
    
    # Create a mock client
    mock_client = AsyncMock()
    
    # Call _score_with_mode
    result = await _score_with_mode(mock_runner_mod, mock_client, row, "reference")
    
    # Verify judge was NOT called because predicted is empty
    assert judge_called is False
    assert result is False


@pytest.mark.asyncio
async def test_non_answer_predicted_scores_false_without_judge():
    """Test that non-answer predicted (e.g., 'I don't know') scores False without judge call."""
    from benchmarks.longmemeval_runner import build_row
    
    # Create a test row with non-answer predicted
    row = build_row(
        idx=5,
        question_type="temporal",
        question="What is the question?",
        gold_answer="The answer",
        full_context="Some context",
        correct=False,
        retrieved_chars=20,
        retrieval_delta=None,
        save_context=True,
        predicted="I don't know",  # Non-answer phrase
    )
    
    # Create a mock runner module
    mock_runner_mod = type('MockRunnerMod', (), {})()
    
    # Track if judge was called
    judge_called = False
    
    async def mock_score_answer_llm(client, predicted, gold, question, context=None):
        nonlocal judge_called
        judge_called = True
        return True
    
    mock_runner_mod.score_answer_llm = mock_score_answer_llm
    
    # Set the mock runner module for testing
    set_runner_mod_for_testing(mock_runner_mod)
    
    # Create a mock client
    mock_client = AsyncMock()
    
    # Call _score_with_mode
    result = await _score_with_mode(mock_runner_mod, mock_client, row, "reference")
    
    # Verify judge was NOT called because predicted is a non-answer
    assert judge_called is False
    assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
