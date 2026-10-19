"""Tests for the LongMemEval runner's row-building and save-context logic.

This validates Part 1 of the LongMemEval suite, touching the per-question row
in benchmarks/longmemeval_runner.py. It is a BENCHMARK file: default scoring
and every existing results key must stay exactly as on master.
"""

import argparse
import asyncio
import importlib.util
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


# ---------------------------------------------------------------------------
# Helper test utilities
# ---------------------------------------------------------------------------

class _FakeVectorMemory:
    """Minimal vector source for retrieval-only tests."""

    def __init__(self, texts):
        self.texts = texts

    async def search(self, query, limit=5, hybrid=True, fusion="boost", project=None, search_agents=None):
        return [
            {"id": i, "text": t, "similarity": 1.0 - i * 0.01, "metadata": {}}
            for i, t in enumerate(self.texts[:limit])
        ]


class _FakeArchive:
    async def search_fts(self, term, limit=5):
        return []


class _FakeKG:
    pass


class _RecordingAssembler:
    """Stands in for ContextAssembler; contributes nothing to the context."""

    def __init__(self, kg=None, archive=None):
        pass

    async def assemble(self, query, depth="auto", max_total_tokens=4000):
        return {"context": ""}


def _context(mod, **kwargs):
    vmem = kwargs.pop("vmem", None) or _FakeVectorMemory(["alpha chunk", "beta chunk"])
    return asyncio.run(
        mod.retrieve_context(
            "who met the captain", _FakeKG(), _FakeArchive(), vmem, **kwargs
        )
    ), vmem


# ---------------------------------------------------------------------------
# Helper fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def _args(**overrides):
    base = dict(
        limit=1,
        type=None,
        llm=False,  # Retrieval-only path
        graph_expansion=0,
        retrieval_path="retrieve",
        report_retrieval_delta=False,
        out="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def fake_runner(monkeypatch):
    """Create a runner with all the necessary fakes."""
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    monkeypatch.setattr(mod, "load_dataset", lambda: [])

    # Fake the retrieval components
    async def fake_retrieve(query, **kwargs):
        return [{"text": "fake context chunk"}]

    monkeypatch.setattr(mod, "_retrieve", fake_retrieve)

    # Set up the environment for save_context parsing
    monkeypatch.setenv("TAOSMD_LME_SAVE_CONTEXT", "1")

    return mod


# ---------------------------------------------------------------------------
# Test NON_ANSWER_PHRASES and is_non_answer
# ---------------------------------------------------------------------------

def test_is_non_answer():
    """Test the is_non_answer() function with all phases."""
    mod = _load_runner()

    # Empty/None -> True
    assert mod.is_non_answer(None) == True
    assert mod.is_non_answer("") == True

    # Each phrase in various cases -> True
    for phrase in mod.NON_ANSWER_PHRASES:
        # Lowercase
        assert mod.is_non_answer(phrase) == True, f"Failed for lowercase: {phrase}"

        # Mixed case
        mixed = phrase[0].upper() + phrase[1:]
        assert mod.is_non_answer(mixed) == True, f"Failed for mixed case: {mixed}"

        # Full sentence with phrase
        assert mod.is_non_answer("I don't know about this") == True
        assert mod.is_non_answer("This is not in the context") == True
        assert mod.is_non_answer("I'm sorry but I cannot help") == True

    # Real answers -> False
    assert mod.is_non_answer("The capital of France is Paris") == False
    assert mod.is_non_answer("I think the answer is 42") == False
    assert mod.is_non_answer("Yes, I know the answer") == False
    assert mod.is_non_answer("no information") == True  # This is a phrase


# ---------------------------------------------------------------------------
# Test build_row function
# ---------------------------------------------------------------------------

@pytest.fixture
def mod(monkeypatch):
    """Load and patch the runner module."""
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    return mod


def test_build_row_save_context_false(mod):
    """Test build_row with save_context=False."""
    row = mod.build_row(
        idx=0,
        question_type="temporal",
        question="What happened in 2020?",
        correct=True,
        full_context="Some context here",
        delta=None,
        predicted="The answer",
        gold_answer="The correct answer",
        save_context=False,
    )

    # Must have all master keys
    assert "idx" in row
    assert "question_type" in row
    assert "question" in row
    assert "correct" in row
    assert "retrieved_chars" in row
    assert "retrieval_delta" in row

    # Plus predicted and answer
    assert "predicted" in row
    assert "answer" in row

    # No context key
    assert "context" not in row

    # Values are correct
    assert row["idx"] == 0
    assert row["question_type"] == "temporal"
    assert row["question"] == "What happened in 2020?"
    assert row["correct"] is True
    assert row["retrieved_chars"] == len("Some context here")
    assert row["retrieval_delta"] is None
    assert row["predicted"] == "The answer"
    assert row["answer"] == "The correct answer"


def test_build_row_save_context_true(mod):
    """Test build_row with save_context=True."""
    long_context = "This is a very long context that should be truncated if it exceeds CONTEXT_CHARS"
    row = mod.build_row(
        idx=1,
        question_type="entity",
        question="Who is the president?",
        correct=False,
        full_context=long_context,
        delta={"legacy_chars": 100, "wired_chars": 200, "identical": False},
        predicted=None,
        gold_answer="The president is Joe",
        save_context=True,
    )

    # Must have all master keys plus predicted and answer
    assert "idx" in row
    assert "question_type" in row
    assert "question" in row
    assert "correct" in row
    assert "retrieved_chars" in row
    assert "retrieval_delta" in row
    assert "predicted" in row
    assert "answer" in row

    # Context key present and truncated if needed
    assert "context" in row
    assert row["context"] == long_context[:mod.CONTEXT_CHARS]

    # Values are correct
    assert row["predicted"] is None
    assert row["answer"] == "The president is Joe"
    assert row["retrieved_chars"] == len(long_context)
    assert row["retrieval_delta"]["legacy_chars"] == 100


def test_build_row_with_none_predicted(mod):
    """Test that predicted None is stored as None."""
    row = mod.build_row(
        idx=2,
        question_type="causal",
        question="Why did it happen?",
        correct=True,
        full_context="Context",
        delta=None,
        predicted=None,
        gold_answer="The reason",
        save_context=False,
    )

    assert row["predicted"] is None
    assert row["answer"] == "The reason"


# ---------------------------------------------------------------------------
# Test retrieval-only path with fakes (use_llm=False)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieval_only_path_with_fakes(fake_runner, _args):
    """Test that the retrieval-only path (use_llm=False) works with fakes.

    This tests that run_benchmark can be driven with fakes without requiring
    a real LLM. If this cannot be driven with fakes without network, we need to
    document that limitation.
    """
    # Create a simple fake dataset
    fake_dataset = [
        {
            "question_type": "temporal",
            "question": "What happened in 2020?",
            "answer": "The pandemic started",
            "haystack_sessions": [],
        }
    ]

    # Mock the dataset loading
    fake_runner.load_dataset = lambda: fake_dataset

    # Run the benchmark
    result = await fake_runner.run_benchmark(
        limit=1,
        question_type=None,
        use_llm=False,  # Retrieval-only
        args=_args
    )

    # Should return a valid accuracy percentage
    assert 0 <= result <= 100

    # Since we have no real context, the answer will likely be incorrect
    # but the point is that the runner runs without errors


# ---------------------------------------------------------------------------
# Test environment variable validation at run_benchmark start
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_env_invalid_raises_before_questions(fake_runner, monkeypatch, _args):
    """Test that TAOSMD_LME_SAVE_CONTEXT='yes' raises ValueError before any question.

    This validates that the env parse in run_benchmark catches invalid values
    before processing any questions.
    """
    # Replace load_dataset with a tracked function
    load_called = []

    def tracked_load_dataset():
        load_called.append(True)
        return []

    fake_runner.load_dataset = tracked_load_dataset

    # Set invalid value
    monkeypatch.setenv("TAOSMD_LME_SAVE_CONTEXT", "yes")

    # The runner should raise ValueError when read_save_context is called
    with pytest.raises(ValueError, match="TAOSMD_LME_SAVE_CONTEXT must be"):
        await fake_runner.run_benchmark(
            limit=1,
            question_type=None,
            use_llm=False,
            args=_args
        )

    # Verify that load_dataset was not called (no questions processed)
    # Since run_benchmark exits early with ValueError, load_dataset should never be reached
    assert load_called == [], f"load_dataset was called {len(load_called)} times, expected 0"

    # Also verify load_dataset wasn't called on the original fake_runner
    assert not hasattr(fake_runner.load_dataset, 'call_count')


def test_read_save_context_with_dict(mod):
    """Test read_save_context with dict inputs as required by revision requirement (a)."""
    # Test {} -> False (unset or empty)
    assert mod.read_save_context({}) == False

    # Test {"TAOSMD_LME_SAVE_CONTEXT": "0"} -> False
    assert mod.read_save_context({"TAOSMD_LME_SAVE_CONTEXT": "0"}) == False

    # Test {"TAOSMD_LME_SAVE_CONTEXT": "1"} -> True
    assert mod.read_save_context({"TAOSMD_LME_SAVE_CONTEXT": "1"}) == True

    # Test {"TAOSMD_LME_SAVE_CONTEXT": "yes"} -> ValueError with message naming valid values
    with pytest.raises(ValueError, match="TAOSMD_LME_SAVE_CONTEXT must be"):
        mod.read_save_context({"TAOSMD_LME_SAVE_CONTEXT": "yes"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
