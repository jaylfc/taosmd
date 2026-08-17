"""Test that realworld_llm_benchmark exits with non-zero code when LLM model is missing."""

import pytest
from benchmarks.realworld_llm_benchmark import run_benchmark


@pytest.mark.asyncio
async def test_exit_code_non_zero_when_model_missing():
    """Verify sys.exit(1) is called when LLM model is not available in Ollama.

    This mirrors the test pattern from PR #290 (eventqa_runner.py): catch
    SystemExit and assert the exit code is non-zero, rather than using a bare
    pytest.raises(SystemExit) which would pass on sys.exit(0).
    """
    with pytest.raises(SystemExit) as exc:
        await run_benchmark(limit=1, top_k=1)
    assert exc.value.code != 0, (
        f"Expected non-zero exit code, got {exc.value.code}"
    )