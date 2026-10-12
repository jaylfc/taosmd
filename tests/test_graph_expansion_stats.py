"""Tests for graph expansion stats reporting (issue #197)."""

from __future__ import annotations

import asyncio

from taosmd import graph_expansion, retrieval


class _FakeKG:
    def __init__(self, results_by_entity: dict[str, list[dict]] | None = None):
        self.results_by_entity = results_by_entity or {}
        self.queries: list[str] = []

    async def query_entity(self, entity: str, as_of=None, track_access=False) -> list[dict]:
        self.queries.append(entity)
        res = self.results_by_entity.get(entity, [])
        return [dict(r) for r in res]


def test_expand_from_results_stats_empty():
    """KG returns [] for everything: stats has correct counts."""
    kg = _FakeKG({})
    results = [{"text": "Alice met Bob"}]
    stats: dict = {}
    expanded = asyncio.run(graph_expansion.expand_from_results(kg, results, stats=stats))
    assert expanded == []
    assert stats["seeds_tried"] == 2
    assert stats["entities_matched"] == 0
    assert stats["triples_returned"] == 0


def test_expand_from_results_stats_with_match():
    """KG returns one triple for one entity."""
    kg = _FakeKG({"Alice": [{"subject_id": "Alice", "predicate": "knows", "object_id": "Bob", "object_name": "Bob", "confidence": 0.9, "current": True, "direction": "outgoing"}]})
    results = [{"text": "Alice met Bob"}]
    stats: dict = {}
    expanded = asyncio.run(graph_expansion.expand_from_results(kg, results, stats=stats))
    assert len(expanded) >= 1
    assert stats["seeds_tried"] == 2
    assert stats["entities_matched"] == 1
    assert stats["triples_returned"] >= 1


def test_expand_from_results_stats_none_unchanged():
    """stats=None: return value equals call without parameter."""
    kg = _FakeKG({"Alice": [{"subject_id": "Alice", "predicate": "knows", "object_id": "Bob", "object_name": "Bob", "confidence": 0.9, "current": True, "direction": "outgoing"}]})
    results = [{"text": "Alice met Bob"}]
    expanded_with = asyncio.run(graph_expansion.expand_from_results(kg, results, stats=None))
    expanded_without = asyncio.run(graph_expansion.expand_from_results(kg, results))
    assert expanded_with == expanded_without


def test_append_graph_expansion_stats_empty():
    """KG returns [] for everything: block_appended False."""
    kg = _FakeKG({})
    results = [{"text": "Alice met Bob"}]
    stats: dict = {}
    out = asyncio.run(retrieval._append_graph_expansion(results, kg, max_tokens=100, graph_expansion_stats=stats))
    assert out == results  # unchanged
    assert stats["seeds_tried"] == 2
    assert stats["entities_matched"] == 0
    assert stats["triples_returned"] == 0
    assert stats["block_appended"] is False


def test_append_graph_expansion_stats_with_match():
    """KG returns one triple for one entity: block_appended True."""
    kg = _FakeKG({"Alice": [{"subject_id": "Alice", "predicate": "knows", "object_id": "Bob", "object_name": "Bob", "confidence": 0.9, "current": True, "direction": "outgoing"}]})
    results = [{"text": "Alice met Bob"}]
    stats: dict = {}
    out = asyncio.run(retrieval._append_graph_expansion(results, kg, max_tokens=100, graph_expansion_stats=stats))
    assert len(out) == len(results)
    assert stats["seeds_tried"] == 2
    assert stats["entities_matched"] == 1
    assert stats["triples_returned"] >= 1
    assert stats["block_appended"] is True


def test_append_graph_expansion_stats_kg_none():
    """kg=None: zeros, block_appended False."""
    results = [{"text": "Alice met Bob"}]
    stats: dict = {}
    out = asyncio.run(retrieval._append_graph_expansion(results, None, max_tokens=100, graph_expansion_stats=stats))
    assert out == results
    assert stats["seeds_tried"] == 0
    assert stats["entities_matched"] == 0
    assert stats["triples_returned"] == 0
    assert stats["block_appended"] is False


def test_append_graph_expansion_stats_none_unchanged():
    """stats=None: return value equals call without the parameter."""
    kg = _FakeKG({"Alice": [{"subject_id": "Alice", "predicate": "knows", "object_id": "Bob", "object_name": "Bob", "confidence": 0.9, "current": True, "direction": "outgoing"}]})
    results = [{"text": "Alice met Bob"}]
    out_with = asyncio.run(retrieval._append_graph_expansion(results, kg, max_tokens=100, graph_expansion_stats=None))
    out_without = asyncio.run(retrieval._append_graph_expansion(results, kg, max_tokens=100))
    assert out_with == out_without
