"""Tests for the three-number bench summary (MemScore-style) in locomo_runner.

Covers:
  - _summary returns mean_latency_ms, p95_latency_ms, mean_context_tokens.
  - _print_summary emits a latency/context line alongside the accuracy table.
  - Empty rows return zeros without crashing.
  - p95 is correctly computed over sorted latencies.
  - context_chars / 4 estimate is applied correctly.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "locomo_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("locomo_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _summary
# ---------------------------------------------------------------------------


def test_summary_empty_returns_zero_memscore_fields():
    runner = _load_runner()
    s = runner._summary([])
    assert s["mean_latency_ms"] == 0.0
    assert s["p95_latency_ms"] == 0.0
    assert s["mean_context_tokens"] == 0.0
    assert s["count"] == 0


def test_summary_single_row():
    runner = _load_runner()
    row = {
        "f1": 0.5, "bleu1": 0.4, "judge": 0.6,
        "retrieval_ms": 200.0, "gen_ms": 800.0,
        "context_chars": 400,
        "evidence_hits": 0, "evidence_total": 0,
    }
    s = runner._summary([row])
    assert s["mean_latency_ms"] == 1000.0
    assert s["p95_latency_ms"] == 1000.0
    assert s["mean_context_tokens"] == 100.0  # 400 / 4


def test_summary_multiple_rows_mean_latency():
    runner = _load_runner()
    rows = [
        {"f1": 0.5, "bleu1": 0.4, "judge": 0.6,
         "retrieval_ms": 100.0, "gen_ms": 900.0, "context_chars": 800,
         "evidence_hits": 0, "evidence_total": 0},
        {"f1": 0.4, "bleu1": 0.3, "judge": 0.5,
         "retrieval_ms": 200.0, "gen_ms": 300.0, "context_chars": 400,
         "evidence_hits": 0, "evidence_total": 0},
    ]
    s = runner._summary(rows)
    # Mean latency: (1000 + 500) / 2 = 750
    assert s["mean_latency_ms"] == 750.0
    # Mean context tokens: (800/4 + 400/4) / 2 = (200 + 100) / 2 = 150
    assert s["mean_context_tokens"] == 150.0


def test_summary_p95_with_20_rows():
    """p95 index = floor(0.95 * 20) = 19 (last element when sorted)."""
    runner = _load_runner()
    rows = []
    for i in range(20):
        rows.append({
            "f1": 0.5, "bleu1": 0.4, "judge": 0.6,
            "retrieval_ms": float(i * 10), "gen_ms": 0.0, "context_chars": 100,
            "evidence_hits": 0, "evidence_total": 0,
        })
    s = runner._summary(rows)
    # Sorted latencies: [0, 10, 20, ..., 190]. p95 index = min(floor(0.95*20), 19) = 19.
    assert s["p95_latency_ms"] == 190.0


def test_summary_p95_with_10_rows():
    """p95 index = floor(0.95 * 10) = 9 (last element) for 10 rows."""
    runner = _load_runner()
    rows = []
    for i in range(10):
        rows.append({
            "f1": 0.5, "bleu1": 0.4, "judge": 0.6,
            "retrieval_ms": float(i + 1) * 100.0, "gen_ms": 0.0, "context_chars": 0,
            "evidence_hits": 0, "evidence_total": 0,
        })
    s = runner._summary(rows)
    # Sorted latencies: [100, 200, ..., 1000]. p95 idx = min(9, 9) = 9.
    assert s["p95_latency_ms"] == 1000.0


def test_summary_context_tokens_estimate():
    """mean_context_tokens = mean(context_chars) / 4."""
    runner = _load_runner()
    rows = [
        {"f1": 0.5, "bleu1": 0.4, "judge": 0.6,
         "retrieval_ms": 0.0, "gen_ms": 0.0, "context_chars": 1200,
         "evidence_hits": 0, "evidence_total": 0},
        {"f1": 0.5, "bleu1": 0.4, "judge": 0.6,
         "retrieval_ms": 0.0, "gen_ms": 0.0, "context_chars": 800,
         "evidence_hits": 0, "evidence_total": 0},
    ]
    s = runner._summary(rows)
    # mean chars = 1000, tokens = 1000 / 4 = 250
    assert s["mean_context_tokens"] == 250.0


def test_summary_missing_context_chars_defaults_to_zero():
    """Rows from older result files without context_chars default to 0."""
    runner = _load_runner()
    row = {
        "f1": 0.5, "bleu1": 0.4, "judge": 0.6,
        "retrieval_ms": 100.0, "gen_ms": 200.0,
        # No context_chars key
        "evidence_hits": 0, "evidence_total": 0,
    }
    s = runner._summary([row])
    assert s["mean_context_tokens"] == 0.0
    assert s["mean_latency_ms"] == 300.0


# ---------------------------------------------------------------------------
# _print_summary
# ---------------------------------------------------------------------------


def _make_meta(**overrides):
    base = {
        "git_sha": "abc1234",
        "conversations": 2,
        "total_qa": 5,
        "model": "qwen3:4b",
        "strategy": "vector-only",
        "top_k": 10,
    }
    base.update(overrides)
    return base


def _make_overall(**overrides):
    base = {
        "count": 5,
        "f1": 0.55,
        "bleu1": 0.42,
        "judge": 0.61,
        "retrieval_recall": 0.78,
        "mean_latency_ms": 1200.0,
        "p95_latency_ms": 2500.0,
        "mean_context_tokens": 320.0,
    }
    base.update(overrides)
    return base


def test_print_summary_includes_latency_line():
    runner = _load_runner()
    overall = _make_overall()
    meta = _make_meta()
    out = _capture(lambda: runner._print_summary(meta, {}, overall))
    assert "Latency" in out or "latency" in out.lower()
    assert "p95" in out or "P95" in out
    assert "1200" in out or "1200.0" in out  # mean latency
    assert "2500" in out or "2500.0" in out  # p95


def test_print_summary_includes_context_tokens():
    runner = _load_runner()
    overall = _make_overall()
    meta = _make_meta()
    out = _capture(lambda: runner._print_summary(meta, {}, overall))
    # Context token count should appear
    assert "320" in out


def test_print_summary_still_has_accuracy_table():
    """The existing accuracy columns must not be removed."""
    runner = _load_runner()
    overall = _make_overall()
    by_cat = {
        "1": {
            "count": 3, "f1": 0.60, "bleu1": 0.45,
            "judge": 0.70, "retrieval_recall": 0.80,
            "mean_latency_ms": 1000.0, "p95_latency_ms": 2000.0,
            "mean_context_tokens": 300.0,
        }
    }
    meta = _make_meta()
    out = _capture(lambda: runner._print_summary(meta, by_cat, overall))
    assert "F1" in out
    assert "Judge" in out
    assert "R@K" in out
    assert "Overall" in out
    assert "Single-hop" in out


def test_print_summary_zero_latency_does_not_crash():
    """Overall with all zeros must print without errors."""
    runner = _load_runner()
    overall = _make_overall(
        mean_latency_ms=0.0, p95_latency_ms=0.0, mean_context_tokens=0.0,
    )
    meta = _make_meta()
    out = _capture(lambda: runner._print_summary(meta, {}, overall))
    assert "Latency" in out or "latency" in out.lower()
