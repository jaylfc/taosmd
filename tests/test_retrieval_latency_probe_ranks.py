"""Tests for benchmarks/retrieval_latency_probe.py rank helpers (red-first)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the benchmark module importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.retrieval_latency_probe import (  # noqa: E402
    _first_evidence_rank,
    _recall_at_ks,
    _summarize,
)


class TestFirstEvidenceRank:
    """Tests for _first_evidence_rank helper."""

    def test_evidence_at_rank_3(self):
        """Evidence found at 1-based position 3."""
        hits = [
            {"metadata": {"dia_id": "D1:1"}},
            {"metadata": {"dia_id": "D1:2"}},
            {"metadata": {"dia_id": "D1:3"}},
        ]
        evidence = ["D1:3"]
        assert _first_evidence_rank(hits, evidence) == 3

    def test_evidence_not_found(self):
        """Evidence not in hits returns None."""
        hits = [
            {"metadata": {"dia_id": "D1:1"}},
            {"metadata": {"dia_id": "D1:2"}},
            {"metadata": {"dia_id": "D1:3"}},
        ]
        evidence = ["D9:9"]
        assert _first_evidence_rank(hits, evidence) is None

    def test_empty_evidence_returns_none(self):
        """Empty evidence list returns None."""
        hits = [
            {"metadata": {"dia_id": "D1:1"}},
            {"metadata": {"dia_id": "D1:2"}},
        ]
        evidence = []
        assert _first_evidence_rank(hits, evidence) is None

    def test_hit_with_no_metadata_skipped(self):
        """Hit with missing metadata dict is skipped without error."""
        hits = [
            {"metadata": {"dia_id": "D1:1"}},
            {},  # no metadata key
            {"metadata": {"dia_id": "D1:3"}},
        ]
        evidence = ["D1:3"]
        assert _first_evidence_rank(hits, evidence) == 3

    def test_hit_with_none_metadata_skipped(self):
        """Hit with metadata=None is skipped without error."""
        hits = [
            {"metadata": {"dia_id": "D1:1"}},
            {"metadata": None},
            {"metadata": {"dia_id": "D1:3"}},
        ]
        evidence = ["D1:3"]
        assert _first_evidence_rank(hits, evidence) == 3

    def test_first_hit_matches(self):
        """Evidence at rank 1 returns 1."""
        hits = [
            {"metadata": {"dia_id": "D1:1"}},
            {"metadata": {"dia_id": "D1:2"}},
        ]
        evidence = ["D1:1"]
        assert _first_evidence_rank(hits, evidence) == 1

    def test_multiple_evidence_returns_first_rank(self):
        """When multiple evidence items match, returns rank of first match."""
        hits = [
            {"metadata": {"dia_id": "D1:1"}},
            {"metadata": {"dia_id": "D1:2"}},
            {"metadata": {"dia_id": "D1:3"}},
        ]
        evidence = ["D1:2", "D1:3"]
        assert _first_evidence_rank(hits, evidence) == 2


class TestRecallAtKs:
    """Tests for _recall_at_ks helper."""

    def test_basic_ranks(self):
        """Four rows with ranks 1, 4, 12, None; n_hits=20."""
        rows = [
            {"first_evidence_rank": 1, "evidence_total": 1, "n_hits": 20},
            {"first_evidence_rank": 4, "evidence_total": 1, "n_hits": 20},
            {"first_evidence_rank": 12, "evidence_total": 1, "n_hits": 20},
            {"first_evidence_rank": None, "evidence_total": 1, "n_hits": 20},
        ]
        result = _recall_at_ks(rows)
        assert result == {
            "1": 0.25,
            "3": 0.25,
            "5": 0.5,
            "10": 0.5,
            "20": 0.75,
        }

    def test_evidence_total_zero_excluded(self):
        """Row with evidence_total=0 is excluded from denominator."""
        rows = [
            {"first_evidence_rank": 1, "evidence_total": 1, "n_hits": 20},
            {"first_evidence_rank": 4, "evidence_total": 0, "n_hits": 20},  # excluded
            {"first_evidence_rank": 12, "evidence_total": 1, "n_hits": 20},
            {"first_evidence_rank": None, "evidence_total": 1, "n_hits": 20},
        ]
        # Only 3 rows have evidence_total > 0; ranks 1, 12, None
        result = _recall_at_ks(rows)
        assert result == {
            "1": 1 / 3,
            "3": 1 / 3,
            "5": 1 / 3,
            "10": 1 / 3,
            "20": 2 / 3,
        }

    def test_max_n_hits_limits_reported_ks(self):
        """K > max n_hits seen is not reported."""
        rows = [
            {"first_evidence_rank": 1, "evidence_total": 1, "n_hits": 10},
            {"first_evidence_rank": 4, "evidence_total": 1, "n_hits": 10},
            {"first_evidence_rank": 8, "evidence_total": 1, "n_hits": 10},
        ]
        result = _recall_at_ks(rows)
        # 20 > max n_hits (10), so "20" key should be absent
        assert "20" not in result
        assert result == {
            "1": 1 / 3,
            "3": 1 / 3,
            "5": 2 / 3,
            "10": 1.0,
        }

    def test_empty_rows_returns_empty(self):
        """Empty rows list returns empty dict."""
        result = _recall_at_ks([])
        assert result == {}

    def test_no_evidence_rows_returns_empty(self):
        """Rows with no evidence_total > 0 returns empty dict."""
        rows = [
            {"first_evidence_rank": 1, "evidence_total": 0, "n_hits": 20},
            {"first_evidence_rank": 4, "evidence_total": 0, "n_hits": 20},
        ]
        result = _recall_at_ks(rows)
        assert result == {}


class TestSummarize:
    """Tests for _summarize output structure."""

    def test_summary_has_r_at_top_level(self):
        """_summarize output has r_at at top level."""
        rows = [
            {
                "conversation_id": "c1",
                "question": "q1",
                "category": 1,
                "n_hits": 20,
                "n_adjacent": 0,
                "evidence_hits": 1,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": 1,
            }
        ]
        errors = []
        ingest_stats = []
        summary = _summarize(rows, errors, ingest_stats)
        assert "r_at" in summary
        assert isinstance(summary["r_at"], dict)
        assert "1" in summary["r_at"]

    def test_summary_has_r_at_in_each_category(self):
        """_summarize output has r_at in each by_category entry."""
        rows = [
            {
                "conversation_id": "c1",
                "question": "q1",
                "category": 1,
                "n_hits": 20,
                "n_adjacent": 0,
                "evidence_hits": 1,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": 1,
            },
            {
                "conversation_id": "c1",
                "question": "q2",
                "category": 2,
                "n_hits": 20,
                "n_adjacent": 0,
                "evidence_hits": 0,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": None,
            },
        ]
        errors = []
        ingest_stats = []
        summary = _summarize(rows, errors, ingest_stats)
        assert "by_category" in summary
        for cat_data in summary["by_category"].values():
            assert "r_at" in cat_data
            assert isinstance(cat_data["r_at"], dict)

    def test_r_at_k_unchanged(self):
        """r_at_k value is unchanged from before (same as _recall_at_k)."""
        rows = [
            {
                "conversation_id": "c1",
                "question": "q1",
                "category": 1,
                "n_hits": 20,
                "n_adjacent": 0,
                "evidence_hits": 1,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": 1,
            },
            {
                "conversation_id": "c1",
                "question": "q2",
                "category": 1,
                "n_hits": 20,
                "n_adjacent": 0,
                "evidence_hits": 0,
                "evidence_total": 1,
                "retrieval_ms": 10.0,
                "adjacency_ms": 0.0,
                "first_evidence_rank": None,
            },
        ]
        errors = []
        ingest_stats = []
        summary = _summarize(rows, errors, ingest_stats)
        # r_at_k should be 0.5 (1 hit out of 2 with evidence)
        assert summary["r_at_k"] == 0.5
        assert summary["by_category"]["1"]["r_at_k"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
