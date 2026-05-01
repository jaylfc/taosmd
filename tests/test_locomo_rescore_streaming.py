"""Regression tests for benchmarks/locomo_rescore_streaming.py print_scorecard.

Covers:
- LoCoMo-shaped results (integer categories 1-4) — print the per-category breakdown
- LongMemEval-KU-shaped results (no category) — fall back to one Overall bucket
- Empty results — print "no results" instead of dividing by zero
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESCORE_PATH = REPO_ROOT / "benchmarks" / "locomo_rescore_streaming.py"


def _load_rescore_module():
    """Load the rescore script as a module without running its CLI."""
    spec = importlib.util.spec_from_file_location("locomo_rescore_streaming", RESCORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_print_scorecard_locomo_categories():
    """LoCoMo-shaped results: per-category lines plus Overall."""
    rescore = _load_rescore_module()
    data = {
        "results": [
            {"category": 1, "f1": 0.5, "judge": 0.6, "judge_rejudged": 0.7},
            {"category": 1, "f1": 0.4, "judge": 0.5, "judge_rejudged": 0.6},
            {"category": 4, "f1": 0.8, "judge": 0.9, "judge_rejudged": 0.85},
        ],
    }
    out = _capture(lambda: rescore.print_scorecard(Path("test.json"), data, "qwen3:4b"))
    assert "Single-hop" in out  # category 1
    assert "Open-dom" in out  # category 4
    assert "Overall" in out
    # Spot-check the Overall count is 3 and the table prints
    assert " 3 " in out


def test_print_scorecard_uncategorised_results_lmeku_path():
    """Regression: LongMemEval-KU results have no category — must not crash."""
    rescore = _load_rescore_module()
    data = {
        "results": [
            {"f1": 0.2, "judge": 0.0, "judge_rejudged": 0.5},
            {"f1": 0.4, "judge": 0.0, "judge_rejudged": 0.6},
            {"f1": 0.3, "judge": 0.0, "judge_rejudged": 0.7},
            {"category": None, "f1": 0.5, "judge": 0.0, "judge_rejudged": 0.4},
        ],
    }
    out = _capture(lambda: rescore.print_scorecard(Path("lmeku.json"), data, "qwen3:4b"))
    # No per-category line should print (none match c in [1,2,3,4])
    assert "Single-hop" not in out
    # Falls back to one Overall bucket covering all 4 results
    assert "Overall" in out
    assert " 4 " in out  # count column
    # Must not raise — that was the original ZeroDivisionError bug


def test_print_scorecard_empty_results():
    """Empty results print a 'no results' line instead of crashing."""
    rescore = _load_rescore_module()
    out = _capture(lambda: rescore.print_scorecard(Path("empty.json"), {"results": []}, "qwen3:4b"))
    assert "no results" in out


def test_print_scorecard_locomo_with_partial_rescore():
    """Partial rescore coverage (some judge_rejudged is None) still prints."""
    rescore = _load_rescore_module()
    data = {
        "results": [
            {"category": 1, "f1": 0.5, "judge": 0.6, "judge_rejudged": 0.7},
            {"category": 1, "f1": 0.4, "judge": 0.5, "judge_rejudged": None},
            {"category": 2, "f1": 0.8, "judge": 0.9, "judge_rejudged": 0.85},
        ],
    }
    out = _capture(lambda: rescore.print_scorecard(Path("partial.json"), data, "qwen3:4b"))
    assert "Single-hop" in out
    assert "Temporal" in out
    # Coverage on category 1 should reflect 1/2 rescored
    assert "1/2" in out
