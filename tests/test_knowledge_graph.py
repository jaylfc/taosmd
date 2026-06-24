"""Tests for the knowledge-graph helpers used by the dashboard Explorer."""
from __future__ import annotations

import pytest

from taosmd.knowledge_graph import TemporalKnowledgeGraph


@pytest.mark.asyncio
async def test_graph_nodes_and_edges(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_triple("Jay", "prefers", "dark mode")
    await kg.add_triple("Jay", "works on", "taosmd")

    g = await kg.graph(limit=300)
    assert set(g) == {"nodes", "edges", "capped", "total_nodes", "total_edges", "t_min", "t_max"}
    names = {n["name"] for n in g["nodes"]}
    assert "Jay" in names
    jay = next(n for n in g["nodes"] if n["name"] == "Jay")
    assert jay["degree"] == 2
    assert len(g["edges"]) == 2
    assert all(e["active"] for e in g["edges"])
    assert all(e["source"] and e["target"] and e["predicate"] for e in g["edges"])
    assert g["capped"] is False


@pytest.mark.asyncio
async def test_graph_empty(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    g = await kg.graph()
    assert g["nodes"] == [] and g["edges"] == []
    assert g["total_nodes"] == 0 and g["capped"] is False
    assert g["t_min"] is None and g["t_max"] is None


def _names_by_id(g):
    return {n["id"]: n["name"] for n in g["nodes"]}


def _edge_objects(g):
    """Object-side entity names of the edges in a graph result."""
    name = _names_by_id(g)
    return {name.get(e["target"], e["target"]) for e in g["edges"]}


@pytest.mark.asyncio
async def test_graph_as_of_reconstructs_snapshot(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    # Jay lived in London over [100, 200), then Paris over [200, now).
    tid = await kg.add_triple("Jay", "lives in", "London", valid_from=100)
    await kg.invalidate(tid, ended_at=200)
    await kg.add_triple("Jay", "lives in", "Paris", valid_from=200)

    # As of 150: only the London fact was live.
    g150 = await kg.graph(as_of=150)
    name150 = _names_by_id(g150)
    assert "London" in _edge_objects(g150) and "Paris" not in _edge_objects(g150)
    # London was live at 150 but has since been replaced, so it reads inactive.
    london = next(e for e in g150["edges"] if name150.get(e["target"]) == "London")
    assert london["active"] is False

    # As of 250: only the Paris fact is live, and it is still current.
    g250 = await kg.graph(as_of=250)
    name250 = _names_by_id(g250)
    assert "Paris" in _edge_objects(g250) and "London" not in _edge_objects(g250)
    paris = next(e for e in g250["edges"] if name250.get(e["target"]) == "Paris")
    assert paris["active"] is True

    # As of 50 (before anything was known): empty snapshot.
    g50 = await kg.graph(as_of=50)
    assert g50["edges"] == [] and g50["total_edges"] == 0


@pytest.mark.asyncio
async def test_graph_time_bounds(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    tid = await kg.add_triple("Jay", "lives in", "London", valid_from=100)
    await kg.invalidate(tid, ended_at=200)
    await kg.add_triple("Jay", "lives in", "Paris", valid_from=200)
    g = await kg.graph()
    # Earliest valid_from and the latest temporal point across the graph.
    assert g["t_min"] == 100
    assert g["t_max"] == 200


@pytest.mark.asyncio
async def test_activations_after_query(tmp_path):
    import time
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_triple("Jay", "prefers", "dark mode")
    assert await kg.activations(since=time.time() - 60) == []
    await kg.query_entity("Jay")  # track_access defaults True -> bumps last_accessed_at
    acts = await kg.activations(since=time.time() - 60)
    assert len(acts) >= 1
    assert all(set(a) == {"id", "last_accessed_at"} for a in acts)
    g = await kg.graph()
    node_ids = {n["id"] for n in g["nodes"]}
    assert {a["id"] for a in acts} <= node_ids
