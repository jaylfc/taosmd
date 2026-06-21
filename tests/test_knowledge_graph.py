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
    assert set(g) == {"nodes", "edges", "capped", "total_nodes", "total_edges"}
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
