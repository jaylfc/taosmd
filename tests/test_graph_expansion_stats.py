"""Tests for graph expansion stats tracking (issue #197)."""
from __future__ import annotations

import asyncio

from taosmd.graph_expansion import expand_from_results
from taosmd.retrieval import _append_graph_expansion


class _FakeKG:
    def __init__(self, triples_by_entity):
        self.triples_by_entity = triples_by_entity
        self.queries = []

    async def query_entity(self, entity: str, as_of=None, track_access=False):
        self.queries.append(entity)
        return list(self.triples_by_entity.get(entity, []))


def test_expand_from_results_stats_empty():
    kg = _FakeKG({})
    results = [{"text": "Alice met Bob"}]
    stats = {}
    out = asyncio.run(expand_from_results(kg, results, max_seeds=2, stats=stats))
    assert stats == {"seeds_tried": 2, "entities_matched": 0, "triples_returned": 0}
    assert out == []


def test_expand_from_results_stats_has_match():
    kg = _FakeKG({"Alice": [{"subject_id": "alice", "predicate": "knows", "object_id": "bob", "object_name": "Bob", "confidence": 1.0, "direction": "outgoing"}]})
    results = [{"text": "Alice met Bob"}]
    stats = {}
    out = asyncio.run(expand_from_results(kg, results, max_seeds=2, stats=stats))
    assert stats["seeds_tried"] == 2
    assert stats["entities_matched"] >= 1
    assert stats["triples_returned"] >= 1
    assert len(out) >= 1


def test_expand_from_results_stats_none():
    kg = _FakeKG({})
    results = [{"text": "Alice met Bob"}]
    out_no_stats = asyncio.run(expand_from_results(kg, results, max_seeds=2))
    out_with_none = asyncio.run(expand_from_results(kg, results, max_seeds=2, stats=None))
    assert out_no_stats == out_with_none


def test_append_graph_expansion_empty():
    kg = _FakeKG({})
    results = [{"text": "Alice met Bob"}]
    stats = {}
    out = asyncio.run(_append_graph_expansion(results, kg, max_tokens=100, graph_expansion_stats=stats))
    assert stats["seeds_tried"] == 2
    assert stats["entities_matched"] == 0
    assert stats["triples_returned"] == 0
    assert stats["block_appended"] is False
    # return value unchanged
    assert out == results


def test_append_graph_expansion_has_match():
    kg = _FakeKG({"Alice": [{"subject_id": "alice", "predicate": "knows", "object_id": "bob", "object_name": "Bob", "confidence": 1.0, "direction": "outgoing"}]})
    results = [{"text": "Alice met Bob"}]
    stats = {}
    out = asyncio.run(_append_graph_expansion(results, kg, max_tokens=100, graph_expansion_stats=stats))
    assert stats["seeds_tried"] == 2
    assert stats["entities_matched"] >= 1
    assert stats["triples_returned"] >= 1
    assert stats["block_appended"] is True
    assert len(out) == len(results) + 1


def test_append_graph_expansion_kg_none():
    results = [{"text": "Alice met Bob"}]
    stats = {}
    out = asyncio.run(_append_graph_expansion(results, None, max_tokens=100, graph_expansion_stats=stats))
    assert stats == {"seeds_tried": 0, "entities_matched": 0, "triples_returned": 0, "block_appended": False}
    assert out == results


def test_append_graph_expansion_no_stats():
    kg = _FakeKG({})
    results = [{"text": "Alice met Bob"}]
    out = asyncio.run(_append_graph_expansion(results, kg, max_tokens=100))
    assert out == results
