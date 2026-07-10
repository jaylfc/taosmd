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


# ---------------------------------------------------------------------------
# Atomicity: the read-merge-write must run inside one transaction so two
# interleaved add_entity for the same id cannot lose a merged property
# (audit nit). Single-connection SQLite makes this low-risk today, but the
# SELECT-then-UPDATE is hardened with BEGIN IMMEDIATE.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_entity_read_modify_write_wrapped_in_immediate_transaction(tmp_path):
    """The SELECT and the UPDATE run inside a single BEGIN IMMEDIATE tx."""
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    await kg.init()
    await kg.add_entity("Jay", "person", '{"city": "London"}')

    stmts: list[str] = []
    real_conn = kg._conn

    class _RecordingConn:
        """Delegates to the real connection, logging each executed statement."""

        def execute(self, sql, *args, **kwargs):
            stmts.append(" ".join(sql.split()).upper())
            return real_conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_conn, name)

    kg._conn = _RecordingConn()  # type: ignore[assignment]
    try:
        # An enriching re-add: a new property key forces the UPDATE branch.
        await kg.add_entity("Jay", "person", '{"role": "founder"}')
    finally:
        kg._conn = real_conn

    begins = [i for i, s in enumerate(stmts) if s.startswith("BEGIN IMMEDIATE")]
    selects = [i for i, s in enumerate(stmts) if s.startswith("SELECT")]
    updates = [i for i, s in enumerate(stmts) if s.startswith("UPDATE")]
    assert begins, f"no BEGIN IMMEDIATE issued: {stmts}"
    assert selects, f"no SELECT issued: {stmts}"
    assert updates, f"no UPDATE issued (merge should have changed props): {stmts}"
    # The write lock is taken before the read, and the read + write are in the
    # same transaction: BEGIN IMMEDIATE < SELECT < UPDATE.
    assert begins[0] < selects[0] < updates[0], stmts

    import json
    ent = await kg.get_entity("Jay")
    props = json.loads(ent["properties_json"])
    assert props == {"city": "London", "role": "founder"}
    await kg.close()


def test_add_entity_concurrent_property_merges_lose_nothing(tmp_path):
    """Concurrent enrichments from separate connections must all survive.

    Each thread opens its own graph on the same db file and merges a
    distinct property key onto the same entity. A non-atomic read-merge-write
    would drop keys under interleave; BEGIN IMMEDIATE + the busy timeout make
    the writers serialise so every key lands.
    """
    import asyncio
    import threading

    db_path = str(tmp_path / "kg.db")

    async def _seed():
        kg = TemporalKnowledgeGraph(db_path=db_path)
        await kg.init()
        await kg.add_entity("Jay", "person")
        await kg.close()

    asyncio.run(_seed())

    n = 24
    errors: list[BaseException] = []

    def worker(i: int):
        async def _run():
            kg = TemporalKnowledgeGraph(db_path=db_path)
            await kg.init()
            try:
                await kg.add_entity("Jay", "person", f'{{"k{i}": {i}}}')
            finally:
                await kg.close()
        try:
            asyncio.run(_run())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors

    async def _read():
        kg = TemporalKnowledgeGraph(db_path=db_path)
        await kg.init()
        ent = await kg.get_entity("Jay")
        await kg.close()
        return ent

    import json
    ent = asyncio.run(_read())
    props = json.loads(ent["properties_json"])
    missing = {f"k{i}" for i in range(n)} - set(props)
    assert not missing, f"lost merged properties: {sorted(missing)}"
