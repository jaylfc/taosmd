"""Tests for taosmd.retrieval — source adapter normalisation and retrieve()."""

from __future__ import annotations

import asyncio

from taosmd.retrieval import (
    _adapt_catalog,
    _adapt_kg,
    _adapt_vector,
    _attach_neighbors,
    _deduplicate,
    _rrf_merge,
    _user_metadata,
    retrieve,
)


# ---------------------------------------------------------------------------
# Mock sources for retrieve() tests
# ---------------------------------------------------------------------------


class MockVectorMemory:
    async def search(self, query, limit=5, hybrid=True, fusion="rrf"):
        return [
            {"id": 1, "text": "Jay created taOS", "similarity": 0.95, "metadata": {}, "created_at": 1744531200.0},
            {"id": 2, "text": "taOS runs on Orange Pi", "similarity": 0.85, "metadata": {}, "created_at": 1744531200.0},
        ]


class MockKG:
    async def query_entity(self, name, **kwargs):
        if "jay" in name.lower():
            return [{"subject_id": "jay", "predicate": "created", "object_id": "taos",
                     "object_name": "taOS", "subject_name": "Jay", "direction": "outgoing",
                     "confidence": 1.0, "id": "t1"}]
        return []


class MockCatalog:
    async def search_topic(self, query, limit=5):
        return [{"id": 1, "topic": "Working on taOS", "description": "Building memory system",
                 "date": "2026-04-13", "start_str": "09:00", "end_str": "11:30", "category": "coding"}]


class MockArchive:
    async def search_fts(self, query, limit=20):
        return [{"id": 1, "summary": "Discussed taOS architecture", "data_json": "{}",
                 "event_type": "conversation", "timestamp": 1744531200.0}]


class MockCrystals:
    async def search(self, query, limit=10):
        return [{"id": "c1", "narrative": "Built the memory pipeline", "session_id": "s1",
                 "outcomes": "[]", "lessons": "[]"}]


ALL_SOURCES = {
    "vector": MockVectorMemory(),
    "kg": MockKG(),
    "catalog": MockCatalog(),
    "archive": MockArchive(),
    "crystals": MockCrystals(),
}


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


# ---------------------------------------------------------------------------
# retrieve() integration tests
# ---------------------------------------------------------------------------


def test_retrieve_thorough():
    """Thorough mode queries all sources and returns fused results."""
    results = asyncio.run(retrieve(
        query="Jay taOS memory",
        strategy="thorough",
        sources=ALL_SOURCES,
        limit=5,
    ))

    assert isinstance(results, list)
    assert len(results) <= 5

    # Results should come from multiple sources — at least 2 distinct sources
    sources_seen = {r["source"] for r in results}
    assert len(sources_seen) >= 2, f"Expected multiple sources, got: {sources_seen}"

    # Each result must have the normalised schema fields
    for r in results:
        assert "text" in r
        assert "source" in r
        assert "source_id" in r
        assert "rank" in r
        assert "source_score" in r
        assert "metadata" in r
        assert "rrf_score" in r


def test_retrieve_fast():
    """Fast mode returns results quickly from primary (and optionally secondary) source."""
    results = asyncio.run(retrieve(
        query="What did Jay build recently?",
        strategy="fast",
        sources=ALL_SOURCES,
        limit=5,
    ))

    assert isinstance(results, list)
    assert len(results) <= 5
    assert len(results) > 0

    for r in results:
        assert "text" in r
        assert "source" in r


def test_retrieve_minimal():
    """Minimal mode queries only the primary source."""
    results = asyncio.run(retrieve(
        query="What happened recently with taOS?",
        strategy="minimal",
        sources=ALL_SOURCES,
        limit=5,
    ))

    assert isinstance(results, list)
    assert len(results) <= 5
    assert len(results) > 0

    # Minimal mode: only one source type should appear
    sources_seen = {r["source"] for r in results}
    assert len(sources_seen) == 1, (
        f"Minimal mode should use only one source, got: {sources_seen}"
    )


def test_retrieve_custom():
    """Custom mode queries only the sources listed in memory_layers."""
    results = asyncio.run(retrieve(
        query="Jay created taOS",
        strategy="custom",
        memory_layers=["vector", "kg"],
        sources=ALL_SOURCES,
        limit=5,
    ))

    assert isinstance(results, list)
    assert len(results) <= 5
    assert len(results) > 0

    # Only vector and kg results should appear
    allowed_sources = {"vector", "kg"}
    for r in results:
        assert r["source"] in allowed_sources, (
            f"Custom mode with memory_layers=['vector','kg'] returned source {r['source']!r}"
        )


# ---------------------------------------------------------------------------
# adjacent_neighbors — positional neighbour injection
# ---------------------------------------------------------------------------


class PositionalVectorMemory:
    """Test double exposing both search() and get_by_position().

    Backs onto an in-memory list of rows. Supports a single hit-by-position
    lookup with optional group constraint, mirroring the SQLite implementation.
    """

    def __init__(self, rows: list[dict]):
        self._rows = rows

    async def search(self, query, limit=5, hybrid=True, fusion="rrf"):
        # Return the first `limit` rows verbatim for a deterministic ranking.
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "similarity": 0.9 - 0.01 * i,
                "metadata": r["metadata"],
                "created_at": 0.0,
            }
            for i, r in enumerate(self._rows[:limit])
        ]

    async def get_by_position(
        self,
        position_value,
        *,
        position_key="position",
        group_key=None,
        group_value=None,
    ):
        for r in self._rows:
            md = r["metadata"]
            if md.get(position_key) != position_value:
                continue
            if group_key is not None and group_value is not None:
                if md.get(group_key) != group_value:
                    continue
            return {
                "id": r["id"],
                "text": r["text"],
                "metadata": md,
                "created_at": 0.0,
            }
        return None


def _row(rid: int, text: str, **meta) -> dict:
    return {"id": rid, "text": text, "metadata": dict(meta)}


def test_user_metadata_unwraps_vector_nesting():
    """Vector hits keep user metadata under metadata.metadata; helper unwraps it."""
    vector_hit = {
        "text": "t",
        "source": "vector",
        "source_id": "1",
        "rank": 0,
        "source_score": 0.9,
        "metadata": {"id": 1, "metadata": {"position": 7, "session": "s1"}},
    }
    assert _user_metadata(vector_hit) == {"position": 7, "session": "s1"}


def test_user_metadata_passthrough_for_flat_metadata():
    """Non-vector hits store user metadata at the top level."""
    catalog_hit = {
        "text": "t",
        "source": "catalog",
        "source_id": "1",
        "rank": 0,
        "source_score": 1.0,
        "metadata": {"topic": "x", "category": "meeting"},
    }
    assert _user_metadata(catalog_hit) == {"topic": "x", "category": "meeting"}


def test_attach_neighbors_basic_window():
    """adj=2: each hit gets up to 4 surrounding neighbours, primaries excluded."""
    rows = [_row(i, f"turn {i}", position=i) for i in range(10)]
    sources = {"vector": PositionalVectorMemory(rows)}
    hit = {
        "text": "turn 5",
        "source": "vector",
        "source_id": "5",
        "metadata": {"id": 5, "metadata": {"position": 5}},
    }

    result = asyncio.run(_attach_neighbors([hit], sources, n=2, position_key="position", group_key=None))

    neighbours = result[0]["neighbors"]
    positions = [n["metadata"]["position"] for n in neighbours]
    assert sorted(positions) == [3, 4, 6, 7]


def test_attach_neighbors_skips_primary_positions():
    """Two adjacent primaries — neighbours of one must not duplicate the other."""
    rows = [_row(i, f"turn {i}", position=i) for i in range(10)]
    sources = {"vector": PositionalVectorMemory(rows)}
    hits = [
        {"text": "turn 5", "source": "vector", "source_id": "5",
         "metadata": {"id": 5, "metadata": {"position": 5}}},
        {"text": "turn 6", "source": "vector", "source_id": "6",
         "metadata": {"id": 6, "metadata": {"position": 6}}},
    ]

    asyncio.run(_attach_neighbors(hits, sources, n=2, position_key="position", group_key=None))

    pos_a = sorted(n["metadata"]["position"] for n in hits[0].get("neighbors", []))
    pos_b = sorted(n["metadata"]["position"] for n in hits[1].get("neighbors", []))
    # 5's neighbours are 3,4,7 (6 is a primary). 6's neighbours are 8 only
    # (5 is a primary, 4 and 7 already attached to 5 via the dedupe `seen` set).
    assert pos_a == [3, 4, 7]
    assert pos_b == [8]


def test_attach_neighbors_respects_boundaries():
    """No neighbours beyond the smallest/largest stored position."""
    rows = [_row(i, f"turn {i}", position=i) for i in range(3)]  # positions 0,1,2
    sources = {"vector": PositionalVectorMemory(rows)}
    hit = {
        "text": "turn 0",
        "source": "vector",
        "source_id": "0",
        "metadata": {"id": 0, "metadata": {"position": 0}},
    }

    asyncio.run(_attach_neighbors([hit], sources, n=2, position_key="position", group_key=None))

    positions = sorted(n["metadata"]["position"] for n in hit.get("neighbors", []))
    assert positions == [1, 2]  # -1 and -2 don't exist


def test_attach_neighbors_group_filter():
    """group_key constrains neighbours to share the host's group value."""
    rows = [
        _row(1, "a1", position=0, session="A"),
        _row(2, "a2", position=1, session="A"),
        _row(3, "b1", position=0, session="B"),
        _row(4, "b2", position=1, session="B"),
    ]
    sources = {"vector": PositionalVectorMemory(rows)}
    hit = {
        "text": "a2",
        "source": "vector",
        "source_id": "2",
        "metadata": {"id": 2, "metadata": {"position": 1, "session": "A"}},
    }

    asyncio.run(_attach_neighbors([hit], sources, n=1, position_key="position", group_key="session"))

    neighbours = hit.get("neighbors", [])
    assert len(neighbours) == 1
    assert neighbours[0]["text"] == "a1"
    assert neighbours[0]["metadata"]["session"] == "A"


def test_attach_neighbors_skips_hits_missing_required_group():
    """When group_key is configured, a hit lacking that key must not pull cross-group neighbours.

    Regression for the bug where group_value=None silently disabled the SQL
    group filter, letting a group-less hit sample any group's neighbours.
    """
    rows = [
        _row(1, "a1", position=0, session="A"),
        _row(2, "a2", position=1, session="A"),
        _row(3, "b1", position=0, session="B"),
        _row(4, "b2", position=1, session="B"),
    ]
    sources = {"vector": PositionalVectorMemory(rows)}
    hit = {
        "text": "stray",
        "source": "vector",
        "source_id": "99",
        # position is set but session is NOT — group filtering is requested
        # by the caller, so this hit should be left alone, not pull
        # arbitrary same-position rows from other sessions.
        "metadata": {"id": 99, "metadata": {"position": 1}},
    }

    asyncio.run(_attach_neighbors([hit], sources, n=1, position_key="position", group_key="session"))

    assert "neighbors" not in hit


def test_attach_neighbors_zero_is_noop():
    rows = [_row(1, "x", position=0)]
    sources = {"vector": PositionalVectorMemory(rows)}
    hit = {"text": "x", "source": "vector", "source_id": "1",
           "metadata": {"id": 1, "metadata": {"position": 0}}}

    asyncio.run(_attach_neighbors([hit], sources, n=0, position_key="position", group_key=None))

    assert "neighbors" not in hit


def test_attach_neighbors_skipped_when_source_lacks_method():
    """Older mocks without get_by_position must not crash retrieve()."""
    sources = {"vector": MockVectorMemory()}  # no get_by_position
    hit = {"text": "x", "source": "vector", "source_id": "1",
           "metadata": {"id": 1, "metadata": {"position": 5}}}

    asyncio.run(_attach_neighbors([hit], sources, n=2, position_key="position", group_key=None))

    assert "neighbors" not in hit


def test_attach_neighbors_skips_hits_without_position():
    """Hits whose user metadata has no position_key are left untouched."""
    rows = [_row(i, f"turn {i}", position=i) for i in range(5)]
    sources = {"vector": PositionalVectorMemory(rows)}
    hits = [
        {"text": "turn 2", "source": "vector", "source_id": "2",
         "metadata": {"id": 2, "metadata": {"position": 2}}},
        {"text": "stray", "source": "vector", "source_id": "99",
         "metadata": {"id": 99, "metadata": {"some_other_key": "x"}}},
    ]

    asyncio.run(_attach_neighbors(hits, sources, n=1, position_key="position", group_key=None))

    assert "neighbors" in hits[0]
    assert "neighbors" not in hits[1]


def test_retrieve_thorough_attaches_neighbours():
    """End-to-end: retrieve(adjacent_neighbors=N) attaches neighbours via the vector source."""
    rows = [_row(i, f"turn {i}", position=i, session="conv1") for i in range(8)]
    sources = {"vector": PositionalVectorMemory(rows)}
    results = asyncio.run(retrieve(
        query="anything",
        strategy="thorough",
        sources=sources,
        limit=3,
        adjacent_neighbors=2,
        position_key="position",
        group_key="session",
    ))

    assert results
    # At least one of the survivors should have neighbours attached
    with_neighbours = [r for r in results if "neighbors" in r]
    assert with_neighbours, "expected at least one hit to gain neighbours"
