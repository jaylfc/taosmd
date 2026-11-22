"""Tests for hit confidence propagation across all retrieval paths.

Ensures that the confidence an agent sees for a hit is the hit's own source
similarity/score regardless of which path brought it in (fast primary, fast
always-vector, minimal fallback, thorough). Also ensures score-less adapters
report confidence: null (not 1.0).
"""

from __future__ import annotations

import asyncio

import pytest

from taosmd import retrieval
from taosmd import api as taosmd_api


class _FakeVector:
    """Vector fake that returns a hit with a specific similarity."""

    def __init__(self, similarity: float = 0.6, return_hits=None):
        self.calls = []
        self.similarity = similarity
        self.return_hits = return_hits

    async def search(self, query, *, limit=5, hybrid=True, fusion="boost",
                     project=None, search_agents=None):
        self.calls.append({"limit": limit, "fusion": fusion})
        if self.return_hits is not None:
            return self.return_hits
        return [{"id": "v1", "text": "The deploy check phrase is heliotrope-lantern-42.",
                 "similarity": self.similarity, "metadata": {}}]


class _FakeKG:
    """KG fake that returns empty results by default."""

    def __init__(self, return_hits=None):
        self.calls = []
        self.return_hits = return_hits or []

    async def query_entity(self, word):
        self.calls.append(word)
        return self.return_hits


class _FakeArchive:
    """Archive fake that returns empty results by default."""

    def __init__(self, return_hits=None):
        self.calls = []
        self.return_hits = return_hits or []

    async def search_fts(self, query, limit=5):
        self.calls.append(query)
        return self.return_hits


class _FakeCatalog:
    """Catalog fake that returns empty results by default."""

    def __init__(self, return_hits=None):
        self.calls = []
        self.return_hits = return_hits or []

    async def search_topic(self, query, limit=5):
        self.calls.append(query)
        return self.return_hits


class _FakeCrystals:
    """Crystals fake that returns empty results by default."""

    def __init__(self, return_hits=None):
        self.calls = []
        self.return_hits = return_hits or []

    async def search(self, query, limit=10):
        self.calls.append(query)
        return self.return_hits


# ---------------------------------------------------------------------------
# Test (a): same vector hit (similarity 0.6) reports confidence 0.6 on every path
# ---------------------------------------------------------------------------

def test_vector_confidence_same_on_every_path():
    """The same vector hit (similarity 0.6) must report confidence 0.6 via:
    - fast with vector primary (exploratory query)
    - fast with vector added by always-query-vector block (factual query)
    - minimal fallback (when primary returns empty)
    - thorough (all sources merged)
    """
    similarity = 0.6
    vec = _FakeVector(similarity=similarity)
    kg = _FakeKG()
    archive = _FakeArchive()
    catalog = _FakeCatalog()
    crystals = _FakeCrystals()

    sources = {"vector": vec, "kg": kg, "archive": archive, "catalog": catalog, "crystals": crystals}

    # 1. Fast with vector primary (exploratory: "deploy check phrase")
    # This uses vector as primary per the intent classifier
    results = asyncio.run(retrieval.retrieve(
        "deploy check phrase",
        sources=sources,
        strategy="fast",
        limit=5,
    ))
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    hit = results[0]
    assert hit["source"] == "vector"
    assert hit["source_score"] == similarity, (
        f"Fast primary: expected source_score {similarity}, got {hit['source_score']}"
    )

    # Reset calls
    vec.calls.clear()
    kg.calls.clear()
    archive.calls.clear()

    # 2. Fast with vector added by always-query-vector block (factual: "what is the deploy check phrase")
    # For factual queries, primary is kg, secondary is archive, vector is added by the always-query block
    from taosmd import intent_classifier
    assert intent_classifier.classify_intent("what is the deploy check phrase") == "factual"

    results = asyncio.run(retrieval.retrieve(
        "what is the deploy check phrase",
        sources=sources,
        strategy="fast",
        limit=5,
    ))
    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
    # Find the vector hit
    vector_hits = [r for r in results if r["source"] == "vector"]
    assert len(vector_hits) == 1, f"Expected exactly 1 vector hit, got {len(vector_hits)}"
    hit = vector_hits[0]
    assert hit["source_score"] == similarity, (
        f"Fast always-vector: expected source_score {similarity}, got {hit['source_score']}"
    )

    # Reset calls
    vec.calls.clear()
    kg.calls.clear()
    archive.calls.clear()

    # 3. Minimal fallback (primary kg returns empty, falls back to vector)
    results = asyncio.run(retrieval.retrieve(
        "what is the deploy check phrase",
        sources=sources,
        strategy="minimal",
        limit=5,
    ))
    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
    vector_hits = [r for r in results if r["source"] == "vector"]
    assert len(vector_hits) == 1, f"Expected exactly 1 vector hit, got {len(vector_hits)}"
    hit = vector_hits[0]
    assert hit["source_score"] == similarity, (
        f"Minimal fallback: expected source_score {similarity}, got {hit['source_score']}"
    )

    # Reset calls
    vec.calls.clear()
    kg.calls.clear()
    archive.calls.clear()

    # 4. Thorough (all sources merged via RRF)
    results = asyncio.run(retrieval.retrieve(
        "deploy check phrase",
        sources=sources,
        strategy="thorough",
        limit=5,
    ))
    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
    vector_hits = [r for r in results if r["source"] == "vector"]
    assert len(vector_hits) == 1, f"Expected exactly 1 vector hit, got {len(vector_hits)}"
    hit = vector_hits[0]
    assert hit["source_score"] == similarity, (
        f"Thorough: expected source_score {similarity}, got {hit['source_score']}"
    )


# ---------------------------------------------------------------------------
# Test (b): score-less adapter hits report confidence null
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "source_name,fake_kwargs,query",
    [
        (
            "kg",
            {"return_hits": [{
                "subject_id": "s1", "predicate": "is", "object_id": "o1",
                "subject_name": "deploy check", "object_name": "phrase",
                "direction": "outgoing",
            }]},
            "deploy check phrase",
        ),
        (
            "catalog",
            {"return_hits": [{
                "id": "c1", "topic": "deploy", "description": "check phrase",
                "date": "2026-01-01", "start_str": "00:00", "end_str": "01:00",
                "category": "test",
            }]},
            "deploy check phrase",
        ),
        (
            "archive",
            {"return_hits": [{
                "id": "a1", "summary": "deploy check phrase", "data_json": "{}",
                "event_type": "conversation", "timestamp": 0.0,
            }]},
            "deploy check phrase",
        ),
        (
            "crystals",
            {"return_hits": [{
                "id": "x1", "narrative": "deploy check phrase", "session_id": "s1",
                "outcomes": "[]", "lessons": "[]",
            }]},
            "deploy check phrase",
        ),
    ],
)
def test_scoreless_adapter_hit_reports_null_confidence(source_name, fake_kwargs, query):
    """A score-less row from any of the four adapters must produce a hit with
    source_score == None through the real retrieve(), and _format_hit must emit
    confidence: null.
    """
    vec = _FakeVector(return_hits=[])
    kg = _FakeKG(**(fake_kwargs if source_name == "kg" else {}))
    archive = _FakeArchive(**(fake_kwargs if source_name == "archive" else {}))
    catalog = _FakeCatalog(**(fake_kwargs if source_name == "catalog" else {}))
    crystals = _FakeCrystals(**(fake_kwargs if source_name == "crystals" else {}))

    sources = {"vector": vec, "kg": kg, "archive": archive, "catalog": catalog, "crystals": crystals}

    results = asyncio.run(retrieval.retrieve(
        query,
        sources=sources,
        strategy="custom",
        memory_layers=[source_name],
        limit=5,
    ))
    hits = [r for r in results if r["source"] == source_name]
    assert hits, f"Expected at least one {source_name} hit, got {results}"
    for hit in hits:
        assert hit["source_score"] is None, (
            f"{source_name}: expected source_score None, got {hit['source_score']!r}"
        )
        formatted = taosmd_api._format_hit(hit)
        assert formatted["confidence"] is None, (
            f"{source_name}: _format_hit should emit confidence null, got {formatted['confidence']!r}"
        )


# ---------------------------------------------------------------------------
# Test (c): hits with real scores keep them exactly (0.912 stays 0.912)
# ---------------------------------------------------------------------------

def test_real_scores_unchanged():
    """Hits with real scores from any adapter must preserve them exactly
    through all retrieval paths and formatting.
    """
    # Vector with specific similarity
    vec = _FakeVector(similarity=0.912)
    # KG with specific confidence
    kg = _FakeKG(return_hits=[
        {"subject_id": "s1", "predicate": "is", "object_id": "o1",
         "subject_name": "test", "object_name": "value",
         "direction": "outgoing", "confidence": 0.837}
    ])
    # Archive with specific score
    archive = _FakeArchive(return_hits=[
        {"id": "a1", "summary": "archive note", "data_json": "{}",
         "event_type": "conversation", "timestamp": 0.0, "score": 0.721}
    ])
    # Catalog with specific score
    catalog = _FakeCatalog(return_hits=[
        {"id": "c1", "topic": "test", "description": "catalog entry",
         "date": "2026-01-01", "start_str": "00:00", "end_str": "01:00",
         "category": "test", "score": 0.654}
    ])
    # Crystals with specific score
    crystals = _FakeCrystals(return_hits=[
        {"id": "x1", "narrative": "crystal narrative", "session_id": "s1",
         "outcomes": "[]", "lessons": "[]", "score": 0.543}
    ])

    sources = {"vector": vec, "kg": kg, "archive": archive, "catalog": catalog, "crystals": crystals}

    # Test each strategy preserves scores
    for strategy in ["thorough", "fast", "minimal", "custom"]:
        if strategy == "custom":
            results = asyncio.run(retrieval.retrieve(
                "test query",
                sources=sources,
                strategy=strategy,
                memory_layers=["vector", "kg", "archive", "catalog", "crystals"],
                limit=5,
            ))
        else:
            results = asyncio.run(retrieval.retrieve(
                "test query",
                sources=sources,
                strategy=strategy,
                limit=5,
            ))

        # Check each source's hit has the correct source_score
        for hit in results:
            source = hit["source"]
            expected_score = {
                "vector": 0.912,
                "kg": 0.837,
                "archive": 0.721,
                "catalog": 0.654,
                "crystals": 0.543,
            }.get(source)
            if expected_score is not None:
                assert hit["source_score"] == expected_score, (
                    f"{strategy} strategy: {source} hit source_score {hit['source_score']} != {expected_score}"
                )
                # Also verify _format_hit preserves it
                formatted = taosmd_api._format_hit(hit)
                assert formatted["confidence"] == expected_score, (
                    f"{strategy} strategy: {source} formatted confidence {formatted['confidence']} != {expected_score}"
                )


# ---------------------------------------------------------------------------
# Additional: verify _format_hit handles None source_score correctly
# ---------------------------------------------------------------------------

def test_format_hit_none_source_score():
    """_format_hit must emit confidence: null when source_score is None."""
    hit = {
        "text": "test",
        "source": "kg",
        "source_id": "1",
        "rank": 0,
        "source_score": None,
        "metadata": {},
    }
    formatted = taosmd_api._format_hit(hit)
    assert formatted["confidence"] is None
    assert formatted["source"] == "kg"


def test_format_hit_zero_source_score():
    """_format_hit must emit confidence: 0.0 when source_score is 0.0 (not None)."""
    hit = {
        "text": "test",
        "source": "kg",
        "source_id": "1",
        "rank": 0,
        "source_score": 0.0,
        "metadata": {},
    }
    formatted = taosmd_api._format_hit(hit)
    assert formatted["confidence"] == 0.0
    assert formatted["source"] == "kg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
