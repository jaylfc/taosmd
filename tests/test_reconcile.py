"""Tests for the archive→vector reconcile path (issue #102).

Verifies that a turn present in the archive but absent from the vector store
(simulating a crash between the two sequential writes in ingest()) is detected
by reconcile() and, when repair=True, re-added so search() can find it again.

All tests are hermetic: each uses its own tmp_path data dir and a
deterministic fake embedder so ONNX/QMD are never required.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

import taosmd
from taosmd import api as taosmd_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    """Replace embed() with a stable hash-based 8-dim vector (no ONNX/QMD)."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolated data dir + clean stores cache; embedder patched after first init."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    # Release SQLite handles before pytest removes tmp_path.
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _setup(data_dir):
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


def _drop_vector_row_by_text(data_dir, text: str) -> int:
    """Directly delete one vector_memory row matching ``text`` (simulate crash gap).

    Returns the number of rows deleted. Uses a subquery to delete exactly one
    row by rowid — SQLite's ``DELETE ... LIMIT`` requires a special compile flag
    and is not available in the stdlib distribution.
    """
    db = data_dir / "vector-memory.db"
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute(
            "DELETE FROM vector_memory WHERE rowid IN "
            "(SELECT rowid FROM vector_memory WHERE text = ? LIMIT 1)",
            (text,),
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def _drop_all_vector_rows_by_text(data_dir, text: str) -> int:
    """Delete ALL vector_memory rows matching ``text``."""
    db = data_dir / "vector-memory.db"
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute("DELETE FROM vector_memory WHERE text = ?", (text,))
        con.commit()
        return cur.rowcount
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Core repair test
# ---------------------------------------------------------------------------

def test_reconcile_repairs_crash_gap(isolated):
    """ingest 3 turns, drop one vector row, reconcile repairs it."""
    data_dir = isolated
    stores = _setup(data_dir)
    agent = "reconcile-agent"

    turns = [
        "the librarian keeps every word",
        "even if the power goes out",
        "that is the whole point",
    ]
    for t in turns:
        asyncio.run(taosmd.ingest(t, agent=agent, data_dir=str(data_dir)))

    # Simulate crash: delete one vector row directly.
    missing_text = turns[1]
    dropped = _drop_vector_row_by_text(data_dir, missing_text)
    assert dropped == 1, "should have dropped exactly one vector row"

    # Reconcile.
    result = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))

    assert result["agent"] == agent
    assert result["archive_turns"] == 3
    assert result["missing"] == 1
    assert result["readded"] == 1
    assert result["checked_ok"] is False  # was False before repair; result reflects pre-repair state

    # After reconcile, search must find the previously-missing turn.
    hits = asyncio.run(taosmd.search(missing_text, agent=agent, data_dir=str(data_dir), limit=5))
    texts = {h["text"] for h in hits}
    assert missing_text in texts, "reconciled entry must be findable via search"


# ---------------------------------------------------------------------------
# Dry-run (--check / repair=False)
# ---------------------------------------------------------------------------

def test_reconcile_dryrun_reports_but_does_not_repair(isolated):
    """repair=False reports missing count without modifying the vector store."""
    data_dir = isolated
    _setup(data_dir)
    agent = "dryrun-agent"

    asyncio.run(taosmd.ingest("alpha turn", agent=agent, data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest("beta turn", agent=agent, data_dir=str(data_dir)))

    _drop_vector_row_by_text(data_dir, "alpha turn")

    # Dry-run.
    result = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=False))
    assert result["missing"] == 1
    assert result["readded"] == 0
    assert result["checked_ok"] is False

    # Real reconcile now fixes it.
    result2 = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result2["readded"] == 1
    assert result2["checked_ok"] is False  # missing was 1 going in

    # A third call on the now-consistent store.
    result3 = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result3["missing"] == 0
    assert result3["readded"] == 0
    assert result3["checked_ok"] is True


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_reconcile_idempotent_on_consistent_store(isolated):
    """Reconciling an already-consistent store reports missing=0, readded=0."""
    data_dir = isolated
    _setup(data_dir)
    agent = "idempotent-agent"

    for t in ("one fish", "two fish", "red fish"):
        asyncio.run(taosmd.ingest(t, agent=agent, data_dir=str(data_dir)))

    result = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result["missing"] == 0
    assert result["readded"] == 0
    assert result["checked_ok"] is True

    # Second call — still consistent.
    result2 = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result2["missing"] == 0
    assert result2["readded"] == 0
    assert result2["checked_ok"] is True


# ---------------------------------------------------------------------------
# Supersede guard
# ---------------------------------------------------------------------------

def test_reconcile_does_not_resurrect_superseded_entry(isolated):
    """A superseded vector entry counts as present; reconcile must NOT re-add it."""
    data_dir = isolated
    stores = _setup(data_dir)
    agent = "supersede-guard-agent"

    asyncio.run(taosmd.ingest("old fact about jay", agent=agent, data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest("another fact", agent=agent, data_dir=str(data_dir)))

    # Supersede the vector copy of the first turn (correction path).
    superseded = asyncio.run(
        taosmd_api.supersede_vectors("old fact about jay", data_dir=str(data_dir))
    )
    assert superseded == 1

    # Reconcile — the superseded entry must NOT be re-added.
    result = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result["missing"] == 0, (
        "superseded entry should count as present; reconcile must not resurrect it"
    )
    assert result["readded"] == 0
    assert result["checked_ok"] is True

    # The vector row must remain superseded (no new active row was inserted).
    # Search through the underlying vector store directly: the superseded row
    # must not appear in an active-only (valid_to IS NULL) count.
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    vmem = stores["vector"]
    active_rows = vmem._conn.execute(
        "SELECT text FROM vector_memory WHERE valid_to IS NULL AND text = ?",
        ("old fact about jay",),
    ).fetchall()
    assert len(active_rows) == 0, (
        "superseded entry must stay superseded after reconcile; no active row should exist"
    )


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------

def test_reconcile_duplicate_text_repairs_exactly_one_copy(isolated):
    """When the same text is ingested twice and one vector copy is dropped,
    reconcile re-adds exactly one copy — not both.
    """
    data_dir = isolated
    _setup(data_dir)
    agent = "dup-agent"

    repeated = "this message was sent twice"
    asyncio.run(taosmd.ingest(repeated, agent=agent, data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest(repeated, agent=agent, data_dir=str(data_dir)))

    # Drop one of the two vector copies.
    dropped = _drop_vector_row_by_text(data_dir, repeated)
    assert dropped == 1

    result = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result["archive_turns"] == 2
    assert result["missing"] == 1
    assert result["readded"] == 1
    assert result["checked_ok"] is False  # was 1 missing going in

    # Consistent now.
    result2 = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result2["missing"] == 0
    assert result2["checked_ok"] is True


# ---------------------------------------------------------------------------
# Repair metadata parity with the hot ingest path
# ---------------------------------------------------------------------------

def test_reconcile_repair_preserves_project_and_archive_span(isolated):
    """A repaired row must carry the same provenance the hot path writes:
    the project scope and the archive_span_id of the archive row it backs.
    Without them the re-added row falls out of project-scoped search and the
    claims gate can no longer look up its verification status.
    """
    import json as _json

    data_dir = isolated
    stores = _setup(data_dir)
    agent = "provenance-agent"
    project = "proj-fingerprint-123"

    lost_text = "the turn that fell in the crash gap"
    asyncio.run(taosmd.ingest(
        lost_text, agent=agent, project=project, data_dir=str(data_dir)))

    # The archive row id is the span the repaired vector row must point at.
    from taosmd.archive import EVENT_CONVERSATION
    archive_rows = asyncio.run(stores["archive"].query(
        event_type=EVENT_CONVERSATION, agent_name=agent))
    assert len(archive_rows) == 1
    span_id = archive_rows[0]["id"]

    dropped = _drop_vector_row_by_text(data_dir, lost_text)
    assert dropped == 1

    result = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result["readded"] == 1

    vmem = stores["vector"]
    rows = vmem._conn.execute(
        "SELECT metadata_json FROM vector_memory WHERE text = ?", (lost_text,)
    ).fetchall()
    assert len(rows) == 1
    meta = _json.loads(rows[0][0])
    assert meta.get("project") == project, "repair must preserve the project scope"
    assert meta.get("archive_span_id") == span_id, (
        "repair must re-link the vector row to its archive span")


# ---------------------------------------------------------------------------
# Empty store edge case
# ---------------------------------------------------------------------------

def test_reconcile_empty_agent_is_consistent(isolated):
    """An agent with no ingested turns reports consistent (missing=0)."""
    data_dir = isolated
    _setup(data_dir)
    agent = "empty-agent"

    result = asyncio.run(taosmd_api.reconcile(agent=agent, data_dir=str(data_dir), repair=True))
    assert result["archive_turns"] == 0
    assert result["vector_entries"] == 0
    assert result["missing"] == 0
    assert result["checked_ok"] is True
