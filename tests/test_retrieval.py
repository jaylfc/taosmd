"""Tests for taosmd.retrieval — source adapter normalisation."""

from __future__ import annotations

from taosmd.retrieval import _adapt_catalog, _adapt_kg, _adapt_vector


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
