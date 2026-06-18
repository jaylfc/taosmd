"""Tests for the MemX-inspired frequency + importance rerank signals (Lever 2).

MemX folds a retrieval-frequency term freq = min(1, ln(count + 1) / 10) and a
stored importance scalar into ranking. This lever adds those as OPTIONAL,
DEFAULT-OFF auxiliary rerank signals (env TAOSMD_FREQ_RERANK).

IMPORTANT: taOSmd does not currently track per-memory retrieval counts at the
retrieval layer (AccessTracker.track_access is not called from retrieve(), and
vector memories carry no stored importance scalar). So this lever ships as
clean plumbing only: the maths is implemented and unit-tested, but the gate
stays off and a # upgrade-path: TODO marks the missing access-count wiring.
These tests cover the maths and the default-off plumbing, not a faked counter.
"""

from __future__ import annotations

import math

from taosmd.retrieval import apply_freq_rerank, freq_weight


# ---------------------------------------------------------------------------
# freq_weight(): MemX freq = min(1, ln(count + 1) / 10)
# ---------------------------------------------------------------------------


def test_freq_weight_zero_count():
    assert freq_weight(0) == 0.0


def test_freq_weight_matches_memx_formula():
    # ln(10 + 1) / 10 ~= 0.2398
    assert math.isclose(freq_weight(10), math.log(11) / 10, rel_tol=1e-9)


def test_freq_weight_saturates_at_one():
    # ln(count + 1) / 10 only reaches 1.0 around count ~= e^10 - 1; clamp at 1.
    assert freq_weight(10**9) == 1.0


def test_freq_weight_monotonic():
    assert freq_weight(1) < freq_weight(5) < freq_weight(50)


# ---------------------------------------------------------------------------
# apply_freq_rerank(): folds freq + importance into rrf_score and re-sorts
# ---------------------------------------------------------------------------


def _hit(source_id: str, rrf: float, count: int = 0, importance: float = 0.0) -> dict:
    return {
        "text": source_id,
        "source": "vector",
        "source_id": source_id,
        "rank": 0,
        "source_score": 0.5,
        "rrf_score": rrf,
        "metadata": {"access_count": count, "importance": importance},
    }


def test_freq_rerank_promotes_frequent_memory():
    """A tie on rrf_score is broken in favour of the more-frequently-retrieved
    memory once the freq signal is folded in."""
    hits = [_hit("rare", 0.10, count=0), _hit("frequent", 0.10, count=50)]
    out = apply_freq_rerank(hits, freq_weight_factor=0.1, importance_weight=0.0)
    assert out[0]["source_id"] == "frequent"


def test_freq_rerank_uses_importance_scalar():
    hits = [_hit("low", 0.10, importance=0.0), _hit("high", 0.10, importance=0.9)]
    out = apply_freq_rerank(hits, freq_weight_factor=0.0, importance_weight=0.1)
    assert out[0]["source_id"] == "high"


def test_freq_rerank_empty_is_noop():
    assert apply_freq_rerank([], freq_weight_factor=0.1, importance_weight=0.1) == []


def test_freq_rerank_missing_metadata_is_safe():
    """Hits with no access_count / importance keys are treated as count 0,
    importance 0 and are left in their original relative order."""
    hits = [
        {"source_id": "a", "rrf_score": 0.20, "metadata": {}},
        {"source_id": "b", "rrf_score": 0.10, "metadata": {}},
    ]
    out = apply_freq_rerank(hits, freq_weight_factor=0.1, importance_weight=0.1)
    assert [h["source_id"] for h in out] == ["a", "b"]
