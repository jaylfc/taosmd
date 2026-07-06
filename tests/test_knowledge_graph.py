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


# ----------------------------------------------------------------------
# add_entity zero-loss: re-adding an entity must never silently drop a
# recorded type/property. add_entity is called once per subject/object on
# every add_triple, so the same entity is re-inserted constantly with
# whatever type the current extraction guessed (often the "unknown"
# placeholder). The old UPSERT clobbered name/type/properties last-writer-
# wins; these tests pin the non-destructive merge.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_entity_concrete_type_not_downgraded_to_unknown(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Jay", "person")
    # A later triple re-adds the same entity with the default placeholder.
    await kg.add_entity("Jay", "unknown")
    ent = await kg.get_entity("Jay")
    assert ent["type"] == "person"  # concrete classification preserved
    await kg.close()


@pytest.mark.asyncio
async def test_add_entity_placeholder_type_upgraded(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Jay")  # defaults to "unknown"
    await kg.add_entity("Jay", "person")  # enrichment fills the placeholder
    ent = await kg.get_entity("Jay")
    assert ent["type"] == "person"
    await kg.close()


@pytest.mark.asyncio
async def test_add_entity_first_seen_concrete_type_wins(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Acme", "person")
    await kg.add_entity("Acme", "organization")  # conflicting reclassification
    ent = await kg.get_entity("Acme")
    # First-seen concrete type is kept; nothing is silently lost.
    assert ent["type"] == "person"
    await kg.close()


@pytest.mark.asyncio
async def test_add_entity_merges_properties_additively(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Jay", "person", '{"city": "London"}')
    await kg.add_entity("Jay", "person", '{"role": "founder"}')
    ent = await kg.get_entity("Jay")
    import json
    props = json.loads(ent["properties_json"])
    assert props == {"city": "London", "role": "founder"}  # no key dropped
    await kg.close()


@pytest.mark.asyncio
async def test_add_entity_existing_property_value_wins_on_conflict(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Jay", "person", '{"city": "London"}')
    await kg.add_entity("Jay", "person", '{"city": "Paris"}')
    ent = await kg.get_entity("Jay")
    import json
    props = json.loads(ent["properties_json"])
    # The recorded value is preserved rather than overwritten.
    assert props == {"city": "London"}
    await kg.close()


@pytest.mark.asyncio
async def test_add_entity_keeps_first_seen_name(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("taOSmd", "project")
    # Same normalised id, different display casing on a later mention.
    await kg.add_entity("TAOSMD", "project")
    ent = await kg.get_entity("taosmd")
    assert ent["name"] == "taOSmd"  # first-seen display name preserved
    await kg.close()


@pytest.mark.asyncio
async def test_add_entity_preserves_created_at(tmp_path):
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Jay", "person")
    first = (await kg.get_entity("Jay"))["created_at"]
    await kg.add_entity("Jay", "unknown")
    assert (await kg.get_entity("Jay"))["created_at"] == first
    await kg.close()


@pytest.mark.asyncio
async def test_add_entity_survives_malformed_properties_in_column(tmp_path):
    """A malformed properties_json already in the row must not crash a re-add;
    the existing (malformed) value is preserved untouched (crash-safety)."""
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Widget", "thing")
    eid = kg._entity_id("Widget")
    kg._conn.execute(
        "UPDATE kg_entities SET properties_json = ? WHERE id = ?",
        ("not valid json", eid),
    )
    kg._conn.commit()
    # Re-add must not raise and must leave the existing value in place.
    await kg.add_entity("Widget", "thing", properties='{"color": "blue"}')
    ent = await kg.get_entity("Widget")
    assert ent["properties_json"] == "not valid json"
    await kg.close()


@pytest.mark.asyncio
async def test_add_entity_conflicting_concrete_type_keeps_first_and_logs(tmp_path, caplog):
    """Two different concrete types keep the first and emit a debug drift signal."""
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Acme", "organization")
    with caplog.at_level("DEBUG", logger="taosmd.knowledge_graph"):
        await kg.add_entity("Acme", "person")
    ent = await kg.get_entity("Acme")
    assert ent["type"] == "organization"  # first-seen concrete type wins
    assert any("dropping conflicting" in r.message for r in caplog.records)
    await kg.close()
