"""Tests for benchmarks/e048/qrels.py (red-first)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the benchmark module importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.e048.qrels import recall_at_k_qrels


# ---------------------------------------------------------------------------
# Tests for recall_at_k_qrels
# ---------------------------------------------------------------------------

def test_recall_two_relevant_second_at_pos3_counts_at_r5_not_r1():
    """Two relevant docs where only the second is at position 3 counts at R@5 and not R@1."""
    ranked_doc_ids = {
        "q1": ["d1", "d2", "d3", "d4", "d5"],  # d3 is at position 3 (index 2)
    }
    qrels = {
        "q1": {"d3", "d4"},  # Two relevant docs: d3 at pos 3, d4 at pos 4
    }
    
    result = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1, 5, 10))
    
    assert "R@1" in result
    assert "R@5" in result
    assert "R@10" in result
    assert result["R@1"] == 0.0  # d3 is at index 2, not in top 1
    assert result["R@5"] == 1.0  # d3 is in top 5
    assert result["R@10"] == 1.0  # d3 is in top 10


def test_qrels_qid_missing_from_ranked_counts_as_miss():
    """A qrels qid missing from ranked_doc_ids counts as a miss (fraction drops, it is never skipped)."""
    ranked_doc_ids = {
        "q1": ["d1", "d2", "d3"],
    }
    qrels = {
        "q1": {"d1"},
        "q2": {"d2"},  # q2 missing from ranked_doc_ids
    }
    
    result = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1, 5, 10))
    
    assert result["R@1"] == 0.5  # q1 hits, q2 misses -> 1/2 = 0.5
    assert result["R@5"] == 0.5
    assert result["R@10"] == 0.5


def test_extra_ranked_qid_not_in_qrels_changes_nothing():
    """An extra ranked qid not in qrels changes nothing."""
    ranked_doc_ids = {
        "q1": ["d1", "d2", "d3"],
        "q_extra": ["d4", "d5", "d6"],  # Not in qrels
    }
    qrels = {
        "q1": {"d1"},
    }
    
    result = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1, 5, 10))
    
    assert result["R@1"] == 1.0  # Only q1 matters
    assert result["R@5"] == 1.0
    assert result["R@10"] == 1.0


def test_empty_qrels_raises():
    """Empty qrels raises ValueError."""
    ranked_doc_ids = {
        "q1": ["d1", "d2"],
    }
    qrels = {}
    
    with pytest.raises(ValueError, match="qrels is empty"):
        recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1, 5, 10))


def test_custom_ks_returns_exact_keys():
    """ks=(2,) returns exactly {'R@2': ...}."""
    ranked_doc_ids = {
        "q1": ["d1", "d2", "d3"],
        "q2": ["d4", "d5", "d6"],
    }
    qrels = {
        "q1": {"d2"},  # At position 2 (index 1)
        "q2": {"d6"},  # At position 3 (index 2)
    }
    
    result = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(2,))
    
    assert list(result.keys()) == ["R@2"]
    assert result["R@2"] == 0.5  # q1 hits at R@2, q2 misses at R@2 -> 1/2 = 0.5


def test_multiple_queries_mixed_hits():
    """Multiple queries with mixed hit/miss at different K values."""
    ranked_doc_ids = {
        "q1": ["d1", "d2", "d3", "d4"],  # Relevant at pos 1
        "q2": ["d5", "d6", "d7", "d8"],  # Relevant at pos 3
        "q3": ["d9", "d10", "d11"],      # Relevant missing
    }
    qrels = {
        "q1": {"d1"},
        "q2": {"d7"},
        "q3": {"d12"},
    }
    
    result = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1, 3, 5))
    
    # q1 hits at R@1, q2 misses at R@1, q3 misses at R@1 -> 1/3
    assert result["R@1"] == pytest.approx(1/3)
    # q1 hits at R@3, q2 hits at R@3, q3 misses at R@3 -> 2/3
    assert result["R@3"] == pytest.approx(2/3)
    # q1 hits at R@5, q2 hits at R@5, q3 misses at R@5 -> 2/3
    assert result["R@5"] == pytest.approx(2/3)


def test_relevant_at_boundary_k():
    """Relevant doc exactly at position K counts as hit."""
    ranked_doc_ids = {
        "q1": ["d1", "d2", "d3"],
    }
    qrels = {
        "q1": {"d3"},  # At position 3 (index 2)
    }
    
    result = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(2, 3))
    
    assert result["R@2"] == 0.0  # d3 at index 2, not in top 2
    assert result["R@3"] == 1.0  # d3 at index 2, in top 3


def test_iterates_qrels_not_ranked_doc_ids():
    """Function iterates over qrels keys, not ranked_doc_ids keys."""
    # qrels has q1, q2; ranked_doc_ids has q1, q2, q3, q4
    # Only q1 and q2 from qrels should be counted
    ranked_doc_ids = {
        "q1": ["d1"],
        "q2": ["d2"],
        "q3": ["d3"],  # Extra, should be ignored
        "q4": ["d4"],  # Extra, should be ignored
    }
    qrels = {
        "q1": {"d1"},
        "q2": {"d99"},  # Miss
    }
    
    result = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1,))
    
    # Only q1 and q2 from qrels: 1 hit, 1 miss = 0.5
    assert result["R@1"] == 0.5


def test_relevant_doc_not_in_top_k_is_miss():
    """Relevant doc present but beyond K is a miss."""
    ranked_doc_ids = {
        "q1": ["d1", "d2", "d3", "d4", "d5", "d6"],
    }
    qrels = {
        "q1": {"d6"},  # At position 6 (index 5)
    }
    
    result = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(3, 5, 10))
    
    assert result["R@3"] == 0.0
    assert result["R@5"] == 0.0
    assert result["R@10"] == 1.0