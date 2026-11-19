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


def test_vector_always_queried_in_minimal():
    """Test that minimal strategy always queries vector when present."""
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

    # The vector fake should have been called
    assert vec.calls, "vector source was not queried in minimal strategy"
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0]["text"] == "hit"


def test_qmd_alias_still_accepted():
    """Test that 'qmd' source key is still accepted as an alias for 'vector'."""
    query = "what is the deploy check phrase"

    vec = _FakeVector()

    class FakeKG:
        async def query_entity(self, word):
            return []

    class FakeArchive:
        async def search_fts(self, query, limit=5):
            return []

    # Use 'qmd' as the source key (should be aliased to 'vector')
    sources = {"qmd": vec, "kg": FakeKG(), "archive": FakeArchive()}

    # Run retrieval with fast strategy
    results = asyncio.run(retrieval.retrieve(query, sources=sources, strategy="fast"))

    # The vector fake should have been called (via alias)
    assert vec.calls, "vector source was not queried via 'qmd' alias"
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0]["text"] == "hit"


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
