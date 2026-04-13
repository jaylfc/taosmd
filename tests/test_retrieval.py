"""Tests for taosmd.retrieval — source adapter normalisation."""

from __future__ import annotations

from taosmd.retrieval import (
    _adapt_catalog,
    _adapt_kg,
    _adapt_vector,
    _deduplicate,
    _rrf_merge,
)


def test_adapt_vector():
    raw = [
        {
            "id": 42,
            "text": "The capital of France is Paris.",
            "similarity": 0.91,
            "metadata": {"tag": "geo"},
            "created_at": 1744531200.0,
        },
        {
            "id": 99,
            "text": "Python is a programming language.",
            "similarity": 0.78,
            "metadata": {},
            "created_at": 1744531300.0,
        },
    ]

    results = _adapt_vector(raw)

    assert len(results) == 2

    first = results[0]
    assert first["text"] == "The capital of France is Paris."
    assert first["source"] == "vector"
    assert first["source_id"] == "42"
    assert first["rank"] == 0
    assert first["source_score"] == 0.91
    assert first["metadata"] is raw[0]

    second = results[1]
    assert second["source_id"] == "99"
    assert second["rank"] == 1
    assert second["source_score"] == 0.78


def test_adapt_kg():
    raw = [
        {
            "subject_id": "person:alice",
            "subject_name": "Alice",
            "predicate": "works_at",
            "object_id": "org:jan_labs",
            "object_name": "JAN Labs",
            "direction": "outgoing",
            "confidence": 0.95,
        },
        {
            "subject_id": "person:bob",
            "subject_name": "Bob",
            "predicate": "knows",
            "object_id": "person:alice",
            "object_name": "Alice",
            "direction": "outgoing",
            "confidence": 0.80,
        },
    ]

    results = _adapt_kg(raw)

    assert len(results) == 2

    first = results[0]
    assert first["text"] == "Alice works_at JAN Labs"
    assert first["source"] == "kg"
    assert first["source_id"] == "person:alice"
    assert first["rank"] == 0
    assert first["source_score"] == 0.95
    assert first["metadata"] is raw[0]

    second = results[1]
    assert second["text"] == "Bob knows Alice"
    assert second["rank"] == 1
    assert second["source_score"] == 0.80


def test_adapt_catalog():
    raw = [
        {
            "id": 7,
            "topic": "Morning standup",
            "description": "Discussed sprint goals.",
            "date": "2025-04-13",
            "start_str": "09:00",
            "end_str": "09:15",
            "category": "meeting",
        },
        {
            "id": 8,
            "topic": "Deep work",
            "description": "Implemented retrieval adapters.",
            "date": "2025-04-13",
            "start_str": "10:00",
            "end_str": "12:00",
            "category": "work",
        },
    ]

    results = _adapt_catalog(raw)

    assert len(results) == 2

    first = results[0]
    assert "2025-04-13" in first["text"]
    assert "09:00" in first["text"]
    assert "09:15" in first["text"]
    assert "Morning standup" in first["text"]
    assert "Discussed sprint goals." in first["text"]
    assert first["source"] == "catalog"
    assert first["source_id"] == "7"
    assert first["rank"] == 0
    assert first["metadata"] is raw[0]

    second = results[1]
    assert "Deep work" in second["text"]
    assert "10:00" in second["text"]
    assert second["rank"] == 1


def _make_result(source: str, source_id: str, rank: int, text: str = "some text") -> dict:
    return {
        "text": text,
        "source": source,
        "source_id": source_id,
        "rank": rank,
        "source_score": 1.0,
        "metadata": {},
    }


def test_rrf_merge_two_sources():
    # "shared" appears in both lists under different sources but same source_id
    list_a = [
        _make_result("vector", "1", 0, "memory about taOS"),
        _make_result("vector", "2", 1, "another vector result"),
    ]
    list_b = [
        _make_result("kg", "1", 0, "knowledge graph fact"),
        _make_result("kg", "3", 1, "second kg result"),
    ]

    merged = _rrf_merge([list_a, list_b])

    # Result is sorted by rrf_score descending
    scores = [r["rrf_score"] for r in merged]
    assert scores == sorted(scores, reverse=True)

    # All results are present (4 unique source:source_id combos)
    assert len(merged) == 4

    # Each result has an rrf_score field
    for r in merged:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0

    # Rank-0 results from both lists should share the two highest scores
    top_two_scores = scores[:2]
    rank0_keys = {"vector:1", "kg:1"}
    top_two_keys = {f"{r['source']}:{r['source_id']}" for r in merged[:2]}
    assert top_two_keys == rank0_keys, (
        f"Expected top-2 to be rank-0 results, got {top_two_keys}"
    )
    # Their scores should be higher than rank-1 results
    rank1_score = next(
        r["rrf_score"] for r in merged if r["source"] == "vector" and r["source_id"] == "2"
    )
    assert all(s > rank1_score for s in top_two_scores)


def test_rrf_intent_boost():
    list_a = [_make_result("vector", "10", 0)]
    list_b = [_make_result("kg", "20", 0)]

    # Boost vector results
    merged_boosted = _rrf_merge([list_a, list_b], intent_primary="vector", intent_boost=2.0)
    merged_plain = _rrf_merge([list_a, list_b])

    vector_boosted = next(r for r in merged_boosted if r["source"] == "vector")
    vector_plain = next(r for r in merged_plain if r["source"] == "vector")

    kg_boosted = next(r for r in merged_boosted if r["source"] == "kg")
    kg_plain = next(r for r in merged_plain if r["source"] == "kg")

    # Vector score should be doubled
    assert abs(vector_boosted["rrf_score"] - vector_plain["rrf_score"] * 2.0) < 1e-9

    # KG score should be unaffected
    assert abs(kg_boosted["rrf_score"] - kg_plain["rrf_score"]) < 1e-9

    # Boosted vector should rank first
    assert merged_boosted[0]["source"] == "vector"


def test_deduplicate_removes_near_duplicates():
    results = [
        {
            "text": "Jay created taOS on Orange Pi",
            "source": "vector",
            "source_id": "1",
            "rank": 0,
            "source_score": 0.9,
            "metadata": {},
            "rrf_score": 0.02,
        },
        {
            "text": "Jay created taOS on the Orange Pi 5 Plus",
            "source": "vector",
            "source_id": "2",
            "rank": 1,
            "source_score": 0.85,
            "metadata": {},
            "rrf_score": 0.015,
        },
        {
            "text": "Completely unrelated result about Python",
            "source": "kg",
            "source_id": "3",
            "rank": 0,
            "source_score": 0.7,
            "metadata": {},
            "rrf_score": 0.01,
        },
    ]

    filtered = _deduplicate(results, threshold=0.6)

    # Should keep 2 results: one from the near-duplicate pair (higher rrf_score)
    # and the unrelated result
    assert len(filtered) == 2

    # The surviving near-duplicate should be the one with higher rrf_score
    sources_ids = {r["source_id"] for r in filtered}
    assert "1" in sources_ids, "Higher-scored near-duplicate should be kept"
    assert "2" not in sources_ids, "Lower-scored near-duplicate should be removed"
    assert "3" in sources_ids, "Unrelated result should be kept"
