"""Tests for benchmarks/e048/fusion.py (red-first)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the benchmark module importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.e048.fusion import fuse_queries, rrf_fuse


def test_rrf_fuse_two_spaces_basic():
    """Test basic 2-space RRF fusion with hand-computed scores."""
    # Space A ranks: doc1 (rank 0), doc2 (rank 1), doc3 (rank 2), k=60
    # Space B ranks: doc2 (rank 0), doc1 (rank 1), doc4 (rank 2), k=60
    # doc1: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62 ≈ 0.016393 + 0.016129 = 0.032522
    # doc2: 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61 ≈ 0.016129 + 0.016393 = 0.032522
    # doc3: 1/(60+2+1) = 1/63 ≈ 0.015873
    # doc4: 1/(60+2+1) = 1/63 ≈ 0.015873
    ranked_lists = {
        "space_a": ["doc1", "doc2", "doc3"],
        "space_b": ["doc2", "doc1", "doc4"],
    }
    result = rrf_fuse(ranked_lists, k=60)
    assert len(result) == 4
    # doc1 and doc2 have same score (tie) - tie broken by doc id ascending
    # doc1 vs doc2: "doc1" < "doc2", so doc1 comes first
    assert result[0][0] == "doc1"
    assert result[0][1] == pytest.approx(1.0 / 61 + 1.0 / 62)
    assert result[1][0] == "doc2"
    assert result[1][1] == pytest.approx(1.0 / 62 + 1.0 / 61)
    assert result[2][0] in ("doc3", "doc4")
    assert result[3][0] in ("doc3", "doc4")
    # doc3 and doc4 have same score
    assert result[2][1] == pytest.approx(1.0 / 63)
    assert result[3][1] == pytest.approx(1.0 / 63)
    # tie between doc3 and doc4 broken by doc id ascending
    assert result[2][0] < result[3][0]  # "doc3" < "doc4"


def test_rrf_fuse_doc_only_in_one_space():
    """A doc only in one space still ranks (contributes its RRF term)."""
    ranked_lists = {
        "space_a": ["doc1", "doc2"],
        "space_b": ["doc1", "doc3"],
    }
    result = rrf_fuse(ranked_lists, k=60)
    # doc1: 1/61 + 1/62
    # doc2: appears only in space_a at rank 1: 1/62
    # doc3: appears only in space_b at rank 1: 1/62
    assert len(result) == 3
    assert result[0][0] == "doc1"
    # doc2 and doc3 tie at same score 1/62, broken by id
    assert result[1][1] == pytest.approx(1.0 / 62)
    assert result[2][1] == pytest.approx(1.0 / 62)
    assert result[1][0] < result[2][0]


def test_rrf_fuse_tie_broken_by_id():
    """Ties are broken by doc id ascending."""
    ranked_lists = {
        "space_a": ["b", "a"],  # b at rank 0 (1/(k+1)), a at rank 1 (1/(k+2))
        "space_b": ["a", "b"],  # a at rank 0 (1/(k+1)), b at rank 1 (1/(k+2))
    }
    # scores equal for a and b: 1/(k+1) + 1/(k+2) each; tie -> sort by id asc
    result = rrf_fuse(ranked_lists, k=60)
    assert result[0][0] == "a"
    assert result[1][0] == "b"


def test_rrf_fuse_top_n_truncation():
    """top_n truncates results."""
    ranked_lists = {
        "space_a": ["doc1", "doc2", "doc3"],
        "space_b": ["doc2", "doc1", "doc4"],
    }
    result = rrf_fuse(ranked_lists, k=60, top_n=2)
    assert len(result) == 2
    assert result[0][0] == "doc1"
    assert result[1][0] == "doc2"


@pytest.mark.parametrize(
    "case,ranked_lists,k,match",
    [
        ("needs_two_spaces", {"space_a": ["doc1"]}, 60, "at least 2 spaces"),
        (
            "k_too_small",
            {"space_a": ["doc1"], "space_b": ["doc1"]},
            0,
            "k must be >= 1",
        ),
        (
            "empty_list_a",
            {"space_a": [], "space_b": ["doc1"]},
            60,
            "empty ranked list",
        ),
        (
            "empty_list_b",
            {"space_a": ["doc1"], "space_b": []},
            60,
            "empty ranked list",
        ),
        (
            "duplicate_in_a",
            {"space_a": ["doc1", "doc1"], "space_b": ["doc2"]},
            60,
            "duplicate doc id",
        ),
        (
            "duplicate_in_b",
            {"space_a": ["doc1"], "space_b": ["doc2", "doc2"]},
            60,
            "duplicate doc id",
        ),
    ],
)
def test_rrf_fuse_refusals(case, ranked_lists, k, match):
    """Each refusal raises ValueError with appropriate message."""
    with pytest.raises(ValueError, match=match):
        rrf_fuse(ranked_lists, k=k)


def test_fuse_queries_basic():
    """fuse_queries applies rrf_fuse per qid."""
    per_space = {
        "image": {
            "q1": ["img1", "img2"],
            "q2": ["img3", "img1"],
        },
        "text": {
            "q1": ["img2", "img1"],
            "q2": ["img1", "img3"],
        },
    }
    result = fuse_queries(per_space, k=60, top_n=10)
    # Check structure
    assert set(result.keys()) == {"q1", "q2"}
    assert isinstance(result["q1"], list)
    assert isinstance(result["q2"], list)
    # q1 fused order should match rrf_fuse of the two lists
    fused_q1 = rrf_fuse({"image": ["img1", "img2"], "text": ["img2", "img1"]}, k=60)
    assert result["q1"] == [doc for doc, _ in fused_q1]
    fused_q2 = rrf_fuse({"image": ["img3", "img1"], "text": ["img1", "img3"]}, k=60)
    assert result["q2"] == [doc for doc, _ in fused_q2]


def test_fuse_queries_missing_qid_raises():
    """qid missing from any space raises ValueError naming qid and space."""
    per_space = {
        "image": {"q1": ["img1"]},
        "text": {"q1": ["img1"], "q2": ["img2"]},  # q2 missing from image
    }
    with pytest.raises(ValueError, match="qid q2 missing from space image"):
        fuse_queries(per_space, k=60)


def test_fuse_queries_top_n_applied():
    """top_n is applied per query."""
    per_space = {
        "image": {"q1": ["a", "b", "c"]},
        "text": {"q1": ["b", "a", "c"]},
    }
    result = fuse_queries(per_space, k=60, top_n=1)
    assert len(result["q1"]) == 1
    assert result["q1"][0] == "a"  # a and b tie - a comes first
