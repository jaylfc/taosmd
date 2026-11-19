# tests/test_fast_strategy_sources.py
"""Test that vector source is always queried in fast and minimal strategies."""
from __future__ import annotations

import asyncio
import pytest

from taosmd import intent_classifier
from taosmd import retrieval


class _FakeVector:
    """Records the kwargs retrieve() passes to the vector source search."""
    def __init__(self):
        self.calls = []

    async def search(self, query, *, limit=5, hybrid=True, fusion="boost",
                     project=None, search_agents=None):
        self.calls.append({"limit": limit, "fusion": fusion})
        return [{"id": "v1", "text": "hit", "similarity": 1.0, "metadata": {}}]


def test_factual_question_reaches_vector():
    """Test that 'what is ...' questions reach the vector source."""
    query = "what is the deploy check phrase"
    # Verify intent classification gives 'factual'
    intent = intent_classifier.classify_intent(query)
    assert intent == "factual", f"Expected 'factual' intent, got {intent}"

    vec = _FakeVector()

    class FakeKG:
        async def query_entity(self, word):
            return []

    class FakeArchive:
        async def search_fts(self, query, limit=5):
            return []

    sources = {"vector": vec, "kg": FakeKG(), "archive": FakeArchive()}

    # Run retrieval with fast strategy
    results = asyncio.run(retrieval.retrieve(query, sources=sources, strategy="fast"))

    # The vector fake should have been called
    assert vec.calls, "vector source was not queried for factual question"
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0]["text"] == "hit"


def test_exploratory_uses_vector_by_name():
    """Test that exploratory queries use vector as primary when specified."""
    query = "deploy check phrase"

    vec = _FakeVector()

    class FakeKG:
        async def query_entity(self, word):
            return []

    class FakeArchive:
        async def search_fts(self, query, limit=5):
            return []

    # Use sources ordered so vector is first, but not as primary in strategy
    # This tests that vector is queried by name (not just fallback)
    sources = {"kg": FakeKG(), "archive": FakeArchive(), "vector": vec}

    # Mock the strategy to have vector as primary
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        intent_classifier, "get_search_strategy",
        lambda q: {"primary": "vector", "secondary": "kg", "tertiary": "archive"}
    )

    try:
        results = asyncio.run(
            retrieval.retrieve(query, sources=sources, strategy="fast")
        )
        assert vec.calls, "vector source was not queried when specified in sources"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    finally:
        monkeypatch.undo()


def test_minimal_falls_back_to_vector_when_primary_empty():
    """Test that minimal strategy falls back to vector when primary returns empty."""
    query = "what is the deploy check phrase"
    intent = intent_classifier.classify_intent(query)
    assert intent == "factual", f"Expected 'factual' intent, got {intent}"

    vec = _FakeVector()

    class FakeKG:
        async def query_entity(self, word):
            return []

    class FakeArchive:
        async def search_fts(self, query, limit=5):
            return []

    sources = {"vector": vec, "kg": FakeKG(), "archive": FakeArchive()}

    # Run retrieval with minimal strategy
    results = asyncio.run(retrieval.retrieve(query, sources=sources, strategy="minimal"))

    # The vector fake should have been called (primary kg returned empty)
    assert vec.calls, "vector source was not queried when primary returned empty"
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0]["text"] == "hit"
    # All returned hits should have source 'vector'
    for r in results:
        assert r["source"] == "vector", f"Expected source 'vector', got {r['source']}"
    # Archive fake should NOT have been called (we can't directly test this but verify results only have vector)
    assert all(r["source"] == "vector" for r in results)


def test_minimal_stays_on_primary_when_it_has_results():
    """Test that minimal strategy stays on primary when it returns results."""
    query = "what is the deploy check phrase"
    intent = intent_classifier.classify_intent(query)
    assert intent == "factual", f"Expected 'factual' intent, got {intent}"

    vec = _FakeVector()

    class FakeKG:
        async def query_entity(self, word):
            # Return a hit so primary has results
            return [{"subject_id": "s1", "predicate": "is", "object_id": "o1",
                     "subject_name": "deploy check", "object_name": "phrase",
                     "direction": "outgoing", "confidence": 1.0}]

    class FakeArchive:
        async def search_fts(self, query, limit=5):
            return []

    sources = {"vector": vec, "kg": FakeKG(), "archive": FakeArchive()}

    # Run retrieval with minimal strategy
    results = asyncio.run(retrieval.retrieve(query, sources=sources, strategy="minimal"))

    # The vector fake should NOT have been called
    assert not vec.calls, "vector source was incorrectly queried when primary had results"
    # Results should come from kg only
    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
    for r in results:
        assert r["source"] == "kg", f"Expected source 'kg', got {r['source']}"


def test_thorough_unchanged():
    """Test that thorough strategy calls all sources as before."""
    query = "what is the deploy check phrase"

    vec = _FakeVector()

    class FakeKG:
        async def query_entity(self, word):
            return []

    class FakeArchive:
        async def search_fts(self, query, limit=5):
            return []

    sources = {"vector": vec, "kg": FakeKG(), "archive": FakeArchive()}

    # Mock the strategy to return a specific order
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        intent_classifier, "get_search_strategy",
        lambda q: {"primary": "kg", "secondary": "archive", "tertiary": "vector"}
    )

    try:
        results = asyncio.run(
            retrieval.retrieve(query, sources=sources, strategy="thorough")
        )
        # Thorough strategy should call all sources
        assert vec.calls, "vector source was not queried in thorough strategy"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    finally:
        monkeypatch.undo()


def test_vector_fact_survives_full_pages_from_other_sources():
    """A lone vector hit must not be crowded out by full kg and archive pages."""
    query = "what is the deploy check phrase"
    assert intent_classifier.classify_intent(query) == "factual"
    fact = "The deploy check phrase is heliotrope-lantern-42."

    class FullKG:
        """One full page of 5 hits in total (retrieve() calls this once per query word)."""
        def __init__(self):
            self.pages = 0

        async def query_entity(self, word):
            self.pages += 1
            if self.pages > 1:
                return []
            return [
                {"subject_id": f"s{i}", "predicate": "relates to", "object_id": f"o{i}",
                 "subject_name": f"unrelated-subject-{i}", "object_name": f"unrelated-object-{i}",
                 "direction": "outgoing", "confidence": 1.0}
                for i in range(5)
            ]

    class FullArchive:
        async def search_fts(self, query, limit=5):
            return [
                {"id": f"a{i}", "summary": f"unrelated archive note number {i}",
                 "event_type": "conversation", "timestamp": 0.0, "score": 1.0}
                for i in range(5)
            ]

    class OneHitVector:
        def __init__(self):
            self.calls = []

        async def search(self, query, *, limit=5, hybrid=True, fusion="boost",
                         project=None, search_agents=None):
            self.calls.append({"limit": limit})
            return [{"id": "v1", "text": fact, "similarity": 1.0, "metadata": {}}]

    vec = OneHitVector()
    sources = {"kg": FullKG(), "archive": FullArchive(), "vector": vec}
    results = asyncio.run(
        retrieval.retrieve(query, sources=sources, strategy="fast", limit=5)
    )

    assert vec.calls, "vector source was not queried"
    assert fact in [r["text"] for r in results], (
        f"vector fact was crowded out: {[r['text'] for r in results]}"
    )
