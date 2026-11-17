"""Tests for merge_kg (account-device-sync S4 KG half).

Content-keyed merge by (subject_id, normalised predicate, object_id), never by
row id. Peer opened read-only via file:...?mode=ro. One local transaction,
rollback and re-raise on error. Never deletes; peer rows keep their ids; new
facts get fresh ids from the same formula as KnowledgeGraph._triple_id.
"""

from __future__ import annotations

import json
import shutil
import sqlite3

import pytest

from taosmd import KnowledgeGraph
from taosmd.predicate_vocab import SINGULAR_PREDICATES


def _import_merge_kg():
    """Import merge_kg; returns None before the module exists (honest red)."""
    try:
        from taosmd.sync_merge import merge_kg
        return merge_kg
    except ImportError:
        return None


async def _kg(db_path: str) -> KnowledgeGraph:
    kg = KnowledgeGraph(db_path=db_path)
    await kg.init()
    return kg


def _table_counts(db_path: str) -> tuple[int, int]:
    con = sqlite3.connect(db_path)
    try:
        entities = con.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        triples = con.execute("SELECT COUNT(*) FROM kg_triples").fetchone()[0]
    finally:
        con.close()
    return entities, triples


def _triple_keys(db_path: str) -> set[tuple]:
    """(subject, predicate, object, valid_from, valid_to) for every triple."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("""
            SELECT s.name, t.predicate, o.name, t.valid_from, t.valid_to
            FROM kg_triples t
            JOIN kg_entities s ON s.id = t.subject_id
            JOIN kg_entities o ON o.id = t.object_id
        """).fetchall()
    finally:
        con.close()
    return {(r[0], r[1], r[2], r[3], r[4]) for r in rows}


async def _current_outgoing(kg: KnowledgeGraph, name: str) -> list[dict]:
    rows = await kg.query_entity(name)
    return [r for r in rows if r.get("direction") == "outgoing" and r.get("current")]


def _merge_kg_guard():
    merge_kg = _import_merge_kg()
    assert merge_kg is not None, (
        "merge_kg is not implemented: `from taosmd.sync_merge import merge_kg` "
        "raises ImportError before the fix"
    )
    return merge_kg


@pytest.mark.asyncio
async def test_case_1_same_fact_both_open_merges_to_one_row(tmp_path):
    merge_kg = _merge_kg_guard()

    local_db = str(tmp_path / "local.db")
    peer_db = str(tmp_path / "peer.db")

    kg_local = await _kg(local_db)
    tid_local = await kg_local.add_triple(
        "User", "lives_in", "Paris", valid_from=100.0, source="local"
    )
    kg_local._conn.execute(
        "UPDATE kg_triples SET confidence = ?, source_ids = ? WHERE id = ?",
        (0.8, json.dumps(["local-1"]), tid_local),
    )
    kg_local._conn.commit()
    await kg_local.close()

    kg_peer = await _kg(peer_db)
    tid_peer = await kg_peer.add_triple(
        "User", "lives_in", "Paris", valid_from=50.0, source="peer"
    )
    kg_peer._conn.execute(
        "UPDATE kg_triples SET confidence = ?, source_ids = ? WHERE id = ?",
        (0.9, json.dumps(["peer-2"]), tid_peer),
    )
    kg_peer._conn.commit()
    await kg_peer.close()

    before = _table_counts(local_db)
    result = merge_kg(local_db, peer_db)
    after = _table_counts(local_db)

    assert result == {
        "entities_added": 0,
        "entities_typed": 0,
        "triples_added": 0,
        "triples_merged": 1,
        "triples_closed": 0,
        "triples_superseded": 0,
    }
    assert after[0] >= before[0] and after[1] >= before[1]

    con = sqlite3.connect(local_db)
    try:
        row = con.execute("""
            SELECT valid_from, confidence, source_ids, appeared_count
            FROM kg_triples
            WHERE subject_id = 'user' AND predicate = 'lives_in'
              AND object_id = 'paris' AND valid_to IS NULL
        """).fetchone()
        count = con.execute("""
            SELECT COUNT(*) FROM kg_triples
            WHERE subject_id = 'user' AND predicate = 'lives_in'
              AND object_id = 'paris'
        """).fetchone()[0]
    finally:
        con.close()
    assert count == 1, "exactly one open row for the shared fact"
    assert row[0] == 50.0          # valid_from = min(100, 50)
    assert row[1] == 0.9           # confidence = max(0.8, 0.9)
    assert json.loads(row[2]) == ["local-1", "peer-2"]   # sorted union
    assert row[3] == 1             # appeared_count = max


@pytest.mark.asyncio
async def test_case_2_resurrection_peer_older_open_copy_does_not_resurrect(tmp_path):
    merge_kg = _merge_kg_guard()

    local_db = str(tmp_path / "local.db")
    peer_db = str(tmp_path / "peer.db")

    kg_local = await _kg(local_db)
    paris_id = await kg_local.add_triple(
        "User", "lives_in", "Paris", valid_from=100.0
    )
    await kg_local.invalidate(paris_id, ended_at=200.0)
    london_id = await kg_local.add_triple(
        "User", "lives_in", "London", valid_from=250.0
    )
    kg_local._conn.execute(
        "UPDATE kg_triples SET superseded_by = ? WHERE id = ?", (london_id, paris_id)
    )
    kg_local._conn.commit()
    await kg_local.close()

    kg_peer = await _kg(peer_db)
    await kg_peer.add_triple("User", "lives_in", "Paris", valid_from=100.0)
    await kg_peer.close()

    before = _table_counts(local_db)

    # ---- counterfactual: naive INSERT OR REPLACE by row id ----
    def _naive_merge(local_path: str, peer_path: str) -> None:
        con = sqlite3.connect(local_path)
        peer = sqlite3.connect(f"file:{peer_path}?mode=ro", uri=True)
        try:
            for row in peer.execute(
                "SELECT id, name, type, properties_json, created_at FROM kg_entities"
            ):
                con.execute(
                    "INSERT OR REPLACE INTO kg_entities "
                    "(id, name, type, properties_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    row,
                )
            for row in peer.execute(
                "SELECT id, subject_id, predicate, object_id, valid_from, valid_to, "
                "confidence, source, source_ids, superseded_by, appeared_count, created_at "
                "FROM kg_triples"
            ):
                con.execute(
                    "INSERT OR REPLACE INTO kg_triples "
                    "(id, subject_id, predicate, object_id, valid_from, valid_to, "
                    "confidence, source, source_ids, superseded_by, appeared_count, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
            con.commit()
        finally:
            peer.close()
            con.close()

    local_naive = tmp_path / "local_naive.db"
    shutil.copy2(local_db, local_naive)
    _naive_merge(str(local_naive), peer_db)
    kg_naive = await _kg(str(local_naive))
    naive_current = await _current_outgoing(kg_naive, "User")
    await kg_naive.close()
    assert any(r["object_name"] == "Paris" for r in naive_current), (
        "naive INSERT OR REPLACE resurrects closed Paris: case 2 fails under the "
        "INSERT OR REPLACE counterfactual (this is the silent-loss defect)"
    )

    # ---- real merge ----
    result = merge_kg(local_db, peer_db)
    assert result["triples_added"] == 0
    assert result["triples_merged"] == 0
    assert result["triples_closed"] == 0
    assert result["triples_superseded"] == 0

    kg = await _kg(local_db)
    current = await _current_outgoing(kg, "User")
    await kg.close()
    assert all(r["object_name"] != "Paris" for r in current), (
        "Paris must not be current: local closed at 200, peer's older open copy "
        "must not resurrect it"
    )
    assert any(r["object_name"] == "London" for r in current)

    after = _table_counts(local_db)
    assert after[0] >= before[0] and after[1] >= before[1]


@pytest.mark.asyncio
async def test_case_3_closure_wins(tmp_path):
    merge_kg = _merge_kg_guard()

    local_db = str(tmp_path / "local.db")
    peer_db = str(tmp_path / "peer.db")

    kg_local = await _kg(local_db)
    await kg_local.add_triple("User", "lives_in", "Paris", valid_from=100.0)
    await kg_local.close()

    kg_peer = await _kg(peer_db)
    tid_peer = await kg_peer.add_triple("User", "lives_in", "Paris", valid_from=60.0)
    await kg_peer.invalidate(tid_peer, ended_at=150.0)
    await kg_peer.close()

    before = _table_counts(local_db)
    result = merge_kg(local_db, peer_db)

    assert result["triples_closed"] == 1
    assert result["triples_added"] == 0
    assert result["triples_merged"] == 0

    con = sqlite3.connect(local_db)
    try:
        row = con.execute("""
            SELECT valid_to FROM kg_triples
            WHERE subject_id = 'user' AND predicate = 'lives_in'
              AND object_id = 'paris'
        """).fetchone()
    finally:
        con.close()
    assert row is not None and row[0] == 150.0

    after = _table_counts(local_db)
    assert after[0] >= before[0] and after[1] >= before[1]


@pytest.mark.asyncio
async def test_case_4_re_assert_after_closure_keeps_both_intervals(tmp_path):
    merge_kg = _merge_kg_guard()

    local_db = str(tmp_path / "local.db")
    peer_db = str(tmp_path / "peer.db")

    kg_local = await _kg(local_db)
    paris_id = await kg_local.add_triple(
        "User", "lives_in", "Paris", valid_from=100.0
    )
    await kg_local.invalidate(paris_id, ended_at=200.0)
    await kg_local.close()

    kg_peer = await _kg(peer_db)
    await kg_peer.add_triple("User", "lives_in", "Paris", valid_from=300.0)
    await kg_peer.close()

    before = _table_counts(local_db)
    result = merge_kg(local_db, peer_db)

    assert result["triples_added"] == 1
    assert result["triples_merged"] == 0

    con = sqlite3.connect(local_db)
    try:
        rows = con.execute("""
            SELECT valid_from, valid_to FROM kg_triples
            WHERE subject_id = 'user' AND predicate = 'lives_in'
              AND object_id = 'paris'
            ORDER BY valid_from
        """).fetchall()
    finally:
        con.close()
    assert rows == [(100.0, 200.0), (300.0, None)], "one closed, one open interval"

    after = _table_counts(local_db)
    assert after[0] >= before[0] and after[1] >= before[1]


@pytest.mark.asyncio
async def test_case_5_singular_predicate_latest_valid_from_wins(tmp_path):
    merge_kg = _merge_kg_guard()
    assert "lives_in" in SINGULAR_PREDICATES

    local_db = str(tmp_path / "local.db")
    peer_db = str(tmp_path / "peer.db")

    kg_local = await _kg(local_db)
    await kg_local.add_triple("User", "lives_in", "Paris", valid_from=100.0)
    await kg_local.close()

    kg_peer = await _kg(peer_db)
    await kg_peer.add_triple("User", "lives_in", "Berlin", valid_from=200.0)
    await kg_peer.close()

    before = _table_counts(local_db)
    result = merge_kg(local_db, peer_db)

    assert result["triples_added"] == 1
    assert result["triples_superseded"] == 1

    con = sqlite3.connect(local_db)
    try:
        berlin_id = con.execute("""
            SELECT id FROM kg_triples
            WHERE subject_id = 'user' AND predicate = 'lives_in'
              AND object_id = 'berlin' AND valid_to IS NULL
        """).fetchone()
        paris_row = con.execute("""
            SELECT valid_to, superseded_by FROM kg_triples
            WHERE subject_id = 'user' AND predicate = 'lives_in'
              AND object_id = 'paris'
        """).fetchone()
    finally:
        con.close()
    assert berlin_id is not None, "Berlin must be the open winner"
    assert paris_row is not None
    assert paris_row[0] == 200.0, "Paris closed at winner's valid_from"
    assert paris_row[1] == berlin_id[0], "Paris superseded_by Berlin's id"

    after = _table_counts(local_db)
    assert after[0] >= before[0] and after[1] >= before[1]


@pytest.mark.asyncio
async def test_case_6_entity_typing_unknown_beats_typed_never_downgraded(tmp_path):
    merge_kg = _merge_kg_guard()

    # direction 1: local 'unknown', peer 'person' -> 'person'
    local_db = str(tmp_path / "local.db")
    peer_db = str(tmp_path / "peer.db")

    kg_local = await _kg(local_db)
    await kg_local.add_triple("User", "knows", "Jay", subject_type="unknown")
    await kg_local.close()

    kg_peer = await _kg(peer_db)
    await kg_peer.add_triple("User", "knows", "Jay", subject_type="person")
    await kg_peer.close()

    result = merge_kg(local_db, peer_db)

    assert result["entities_typed"] == 1
    assert result["entities_added"] == 0

    con = sqlite3.connect(local_db)
    try:
        row = con.execute("SELECT type FROM kg_entities WHERE id = 'user'").fetchone()
    finally:
        con.close()
    assert row is not None and row[0] == "person"

    # direction 2: local 'person', peer 'unknown' -> stays 'person' (no downgrade)
    local_db2 = str(tmp_path / "local2.db")
    peer_db2 = str(tmp_path / "peer2.db")

    kg_local2 = await _kg(local_db2)
    await kg_local2.add_triple("User", "knows", "Jay", subject_type="person")
    await kg_local2.close()

    kg_peer2 = await _kg(peer_db2)
    await kg_peer2.add_triple("User", "knows", "Jay", subject_type="unknown")
    await kg_peer2.close()

    result2 = merge_kg(local_db2, peer_db2)
    assert result2["entities_typed"] == 0
    con = sqlite3.connect(local_db2)
    try:
        row = con.execute("SELECT type FROM kg_entities WHERE id = 'user'").fetchone()
    finally:
        con.close()
    assert row is not None and row[0] == "person"


@pytest.mark.asyncio
async def test_case_7_row_count_never_decreases(tmp_path):
    merge_kg = _merge_kg_guard()

    local_db = str(tmp_path / "local.db")
    peer_db = str(tmp_path / "peer.db")

    kg_local = await _kg(local_db)
    await kg_local.add_triple("Alice", "works_at", "Acme", valid_from=100.0)
    await kg_local.add_triple("Bob", "lives_in", "Paris", valid_from=200.0)
    await kg_local.close()

    kg_peer = await _kg(peer_db)
    await kg_peer.add_triple("Carol", "prefers", "tea", valid_from=50.0)
    await kg_peer.add_triple("Carol", "prefers", "coffee", valid_from=150.0)
    await kg_peer.close()

    before_entities, before_triples = _table_counts(local_db)
    result = merge_kg(local_db, peer_db)

    after_entities, after_triples = _table_counts(local_db)
    assert after_entities >= before_entities
    assert after_triples >= before_triples
    assert after_triples == before_triples + result["triples_added"]


@pytest.mark.asyncio
async def test_case_8_determinism_across_devices(tmp_path):
    merge_kg = _merge_kg_guard()

    async def _make_pair(db_path: str, order: list[str]) -> None:
        kg = await _kg(db_path)
        for name in order:
            await kg.add_triple("User", "lives_in", name, valid_from=1000.0)
        await kg.close()

    base_a = tmp_path / "base_a.db"
    base_b = tmp_path / "base_b.db"
    await _make_pair(str(base_a), ["Paris", "Berlin"])
    await _make_pair(str(base_b), ["Berlin", "Paris"])

    def _open_object_names(db_path: str) -> set[str]:
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute("""
                SELECT o.name FROM kg_triples t
                JOIN kg_entities o ON o.id = t.object_id
                WHERE t.subject_id = 'user' AND t.predicate = 'lives_in'
                  AND t.valid_to IS NULL
            """).fetchall()
        finally:
            con.close()
        return {r[0] for r in rows}

    # merge A <- B
    copy_a = tmp_path / "copy_a.db"
    copy_b = tmp_path / "copy_b.db"
    shutil.copy2(str(base_a), str(copy_a))
    shutil.copy2(str(base_b), str(copy_b))
    merge_kg(str(copy_a), str(copy_b))
    open_a = _open_object_names(str(copy_a))
    keys_a = _triple_keys(str(copy_a))

    # merge B <- A on independent copies
    copy_a2 = tmp_path / "copy_a2.db"
    copy_b2 = tmp_path / "copy_b2.db"
    shutil.copy2(str(base_a), str(copy_a2))
    shutil.copy2(str(base_b), str(copy_b2))
    merge_kg(str(copy_b2), str(copy_a2))
    open_b = _open_object_names(str(copy_b2))
    keys_b = _triple_keys(str(copy_b2))

    assert open_a == open_b == {"Paris"}
    assert keys_a == keys_b, (
        "both merge directions must converge to the same (subject, predicate, "
        "object, valid_from, valid_to) set regardless of insert order"
    )


@pytest.mark.asyncio
async def test_case_9_superseded_by_remap_peer_id_to_local_id(tmp_path):
    """Peer has (User, lives_in, Paris) closed + superseded_by its Berlin row;
    local is empty. After merge_kg the local Paris row's superseded_by equals
    the LOCAL Berlin row's id (and that id exists in local kg_triples).

    The peer's superseded_by is a device-local peer id that does not exist in
    the local db (bus 7401: never reuse peer ids). The fix remaps it in a
    second pass after all inserts by looking up the peer's superseding row
    in the PEER db by that id, taking its content key and valid_from, and
    finding the LOCAL row with the same content key and valid_from.
    """
    merge_kg = _merge_kg_guard()

    local_db = str(tmp_path / "local.db")
    peer_db = str(tmp_path / "peer.db")

    # Local is empty
    kg_local = await _kg(local_db)
    await kg_local.close()

    # Peer: Paris closed, superseded_by -> Berlin row
    kg_peer = await _kg(peer_db)
    paris_id = await kg_peer.add_triple("User", "lives_in", "Paris", valid_from=100.0)
    await kg_peer.invalidate(paris_id, ended_at=200.0)
    berlin_id = await kg_peer.add_triple("User", "lives_in", "Berlin", valid_from=200.0)
    kg_peer._conn.execute(
        "UPDATE kg_triples SET superseded_by = ? WHERE id = ?", (berlin_id, paris_id)
    )
    kg_peer._conn.commit()
    await kg_peer.close()

    result = merge_kg(local_db, peer_db)

    # Both triples should be added (local was empty)
    assert result["triples_added"] == 2

    # Verify: local Paris row's superseded_by points to LOCAL Berlin row's id
    con = sqlite3.connect(local_db)
    try:
        paris_row = con.execute("""
            SELECT id, superseded_by, valid_from, valid_to FROM kg_triples
            WHERE subject_id = 'user' AND predicate = 'lives_in' AND object_id = 'paris'
        """).fetchone()
        berlin_row = con.execute("""
            SELECT id, valid_from, valid_to FROM kg_triples
            WHERE subject_id = 'user' AND predicate = 'lives_in' AND object_id = 'berlin'
        """).fetchone()
    finally:
        con.close()

    assert paris_row is not None, "Paris row must exist"
    assert berlin_row is not None, "Berlin row must exist"
    assert paris_row[3] == 200.0, "Paris must be closed at 200.0"
    assert berlin_row[2] is None, "Berlin must be open"
    assert paris_row[1] == berlin_row[0], (
        "Paris superseded_by must equal LOCAL Berlin row's id"
    )
    # The local Berlin id must exist in kg_triples
    assert berlin_row[0] is not None
