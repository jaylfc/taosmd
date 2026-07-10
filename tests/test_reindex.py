"""Tests for the archive->vector reindex path (taOS embedder cutover).

taOS is migrating live agents from MiniLM to arctic-embed-s. A MiniLM-embedded
vector store and an arctic query are incompatible vector spaces, so switching
embedders requires RE-EMBEDDING the store. The zero-loss archive is the source
of truth, so the safe procedure is: clear the agent's vector rows, then re-add
every archive turn (which re-embeds under the currently configured embedder).

These tests cover:
- VectorMemory.clear(agent=...) is agent-scoped; clear() with no arg clears all.
- api.reindex(check=True) reports counts and modifies nothing.
- api.reindex(check=False) ends with vector entries == archive turns, ok=True.
- CLI smoke: ``taosmd reindex --check`` runs and prints without error.

All tests are hermetic: each uses its own tmp_path data dir and a deterministic
fake embedder so ONNX/QMD are never required.
"""

from __future__ import annotations

import asyncio

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


async def _active_count(vmem, agent: str) -> int:
    """Count active (non-superseded) vector rows for an agent via iter_entries."""
    n = 0
    async for _text, _meta in vmem.iter_entries(agent=agent, include_superseded=False):
        n += 1
    return n


# ---------------------------------------------------------------------------
# VectorMemory.clear agent scoping
# ---------------------------------------------------------------------------

def test_clear_agent_scoped_leaves_other_agents(isolated):
    """clear(agent='agent-a') deletes only agent A's rows; agent B's rows remain."""
    data_dir = isolated
    stores = _setup(data_dir)
    vmem = stores["vector"]

    asyncio.run(taosmd.ingest("alpha one", agent="agent-a", data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest("alpha two", agent="agent-a", data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest("beta one", agent="agent-b", data_dir=str(data_dir)))

    total_before = asyncio.run(vmem.count())
    assert total_before == 3

    deleted = asyncio.run(vmem.clear(agent="agent-a"))
    assert deleted == 2, "should delete exactly agent A's two rows"

    # Agent A gone, agent B intact.
    assert asyncio.run(_active_count(vmem, "agent-a")) == 0
    assert asyncio.run(_active_count(vmem, "agent-b")) == 1
    assert asyncio.run(vmem.count()) == 1


def test_clear_no_arg_deletes_all(isolated):
    """clear() with no arg still deletes every row (back-compat)."""
    data_dir = isolated
    stores = _setup(data_dir)
    vmem = stores["vector"]

    asyncio.run(taosmd.ingest("alpha one", agent="agent-a", data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest("beta one", agent="agent-b", data_dir=str(data_dir)))
    assert asyncio.run(vmem.count()) == 2

    deleted = asyncio.run(vmem.clear())
    assert deleted == 2
    assert asyncio.run(vmem.count()) == 0


# ---------------------------------------------------------------------------
# api.reindex check (dry-run)
# ---------------------------------------------------------------------------

def test_reindex_check_reports_and_does_not_modify(isolated):
    """reindex(check=True) reports archive_turns and changes nothing."""
    data_dir = isolated
    stores = _setup(data_dir)
    vmem = stores["vector"]
    agent = "check-agent"

    for t in ("one fish", "two fish", "red fish"):
        asyncio.run(taosmd.ingest(t, agent=agent, data_dir=str(data_dir)))

    count_before = asyncio.run(vmem.count())
    assert count_before == 3

    result = asyncio.run(
        taosmd_api.reindex(agent=agent, data_dir=str(data_dir), check=True)
    )
    assert result["agent"] == agent
    assert result["archive_turns"] == 3
    assert result["vector_before"] == 3
    assert result["cleared"] == 0
    assert result["readded"] == 0

    # Nothing modified.
    assert asyncio.run(vmem.count()) == 3


# ---------------------------------------------------------------------------
# api.reindex full run
# ---------------------------------------------------------------------------

def test_reindex_rebuilds_to_archive_count(isolated):
    """reindex(check=False) clears and rebuilds; vector entries == archive turns."""
    data_dir = isolated
    stores = _setup(data_dir)
    vmem = stores["vector"]
    agent = "reindex-agent"

    turns = ["the librarian keeps every word", "even if the power goes out", "that is the point"]
    for t in turns:
        asyncio.run(taosmd.ingest(t, agent=agent, data_dir=str(data_dir)))

    result = asyncio.run(
        taosmd_api.reindex(agent=agent, data_dir=str(data_dir), check=False)
    )
    assert result["agent"] == agent
    assert result["archive_turns"] == 3
    assert result["vector_before"] == 3
    assert result["cleared"] == 3
    assert result["readded"] == 3
    assert result["reindexed_ok"] is True

    # Active vector entries for the agent now equal the archive turn count.
    assert asyncio.run(_active_count(vmem, agent)) == 3

    # Re-embedded entries are still findable.
    hits = asyncio.run(taosmd.search(turns[0], agent=agent, data_dir=str(data_dir), limit=5))
    assert turns[0] in {h["text"] for h in hits}


def test_reindex_repairs_partially_stale_store(isolated):
    """A partially-stale store (some vector rows missing) is fully rebuilt."""
    data_dir = isolated
    stores = _setup(data_dir)
    vmem = stores["vector"]
    agent = "stale-agent"

    turns = ["first turn here", "second turn here", "third turn here"]
    for t in turns:
        asyncio.run(taosmd.ingest(t, agent=agent, data_dir=str(data_dir)))

    # Simulate a partially-stale vector store: drop one row directly.
    vmem._conn.execute("DELETE FROM vector_memory WHERE text = ?", (turns[1],))
    vmem._conn.commit()
    assert asyncio.run(_active_count(vmem, agent)) == 2

    result = asyncio.run(
        taosmd_api.reindex(agent=agent, data_dir=str(data_dir), check=False)
    )
    assert result["archive_turns"] == 3
    assert result["vector_before"] == 2
    assert result["cleared"] == 2
    assert result["readded"] == 3
    assert result["reindexed_ok"] is True
    assert asyncio.run(_active_count(vmem, agent)) == 3


def test_reindex_does_not_touch_other_agents(isolated):
    """Reindexing agent A leaves agent B's vector rows untouched."""
    data_dir = isolated
    stores = _setup(data_dir)
    vmem = stores["vector"]

    asyncio.run(taosmd.ingest("alpha one", agent="agent-a", data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest("beta one", agent="agent-b", data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest("beta two", agent="agent-b", data_dir=str(data_dir)))

    result = asyncio.run(taosmd_api.reindex(agent="agent-a", data_dir=str(data_dir), check=False))
    assert result["reindexed_ok"] is True
    assert asyncio.run(_active_count(vmem, "agent-a")) == 1
    assert asyncio.run(_active_count(vmem, "agent-b")) == 2


def test_reindex_empty_agent_is_consistent(isolated):
    """An agent with no archive turns reindexes to a consistent empty state."""
    data_dir = isolated
    _setup(data_dir)
    agent = "empty-agent"

    result = asyncio.run(taosmd_api.reindex(agent=agent, data_dir=str(data_dir), check=False))
    assert result["archive_turns"] == 0
    assert result["vector_before"] == 0
    assert result["cleared"] == 0
    assert result["readded"] == 0
    assert result["reindexed_ok"] is True


def test_reindex_preserves_project_and_provenance(isolated):
    """A reindexed row must carry the same scope + provenance the hot path
    wrote: the project scope, the archive_span_id of the row it backs, and the
    nested user metadata (source_id/forget_after). Without the project tag the
    rebuilt row falls out of project-scoped search and, worse, becomes visible
    to a DIFFERENT project's search for the same agent -- a cross-project scope
    leak. Mirrors reconcile()'s metadata reconstruction.
    """
    import json as _json

    data_dir = isolated
    stores = _setup(data_dir)
    agent = "scope-agent"
    project = "proj-scope-abc"
    other_project = "proj-other-xyz"
    source_id = "content-hash-42"

    scoped_text = "the project scoped turn that must not leak"
    asyncio.run(taosmd.ingest_batch(
        [{"text": scoped_text, "id": source_id, "metadata": {"forget_after": 999.0}}],
        agent=agent,
        project=project,
        data_dir=str(data_dir),
    ))

    # The archive row id is the span the rebuilt vector row must point at.
    from taosmd.archive import EVENT_CONVERSATION
    archive_rows = asyncio.run(stores["archive"].query(
        event_type=EVENT_CONVERSATION, agent_name=agent))
    assert len(archive_rows) == 1
    span_id = archive_rows[0]["id"]

    result = asyncio.run(
        taosmd_api.reindex(agent=agent, data_dir=str(data_dir), check=False)
    )
    assert result["readded"] == 1
    assert result["reindexed_ok"] is True

    # The rebuilt row must keep project, archive_span_id and nested metadata.
    vmem = stores["vector"]
    rows = vmem._conn.execute(
        "SELECT metadata_json FROM vector_memory WHERE text = ?", (scoped_text,)
    ).fetchall()
    assert len(rows) == 1
    meta = _json.loads(rows[0][0])
    assert meta.get("project") == project, "reindex must preserve the project scope"
    assert meta.get("archive_span_id") == span_id, (
        "reindex must re-link the rebuilt row to its archive span")
    assert isinstance(meta.get("metadata"), dict), "reindex must carry nested metadata"
    assert meta["metadata"].get("source_id") == source_id
    assert meta["metadata"].get("forget_after") == 999.0

    # Scope behaviour: findable under project P ...
    hits = asyncio.run(taosmd.search(
        scoped_text, agent=agent, project=project, data_dir=str(data_dir), limit=5))
    assert scoped_text in {h["text"] for h in hits}, (
        "reindexed row must stay visible to its own project's search")

    # ... and INVISIBLE to a different project's search for the same agent.
    leak = asyncio.run(taosmd.search(
        scoped_text, agent=agent, project=other_project, data_dir=str(data_dir), limit=5))
    assert scoped_text not in {h["text"] for h in leak}, (
        "reindexed row must not leak into a different project's search")


# ---------------------------------------------------------------------------
# service wrapper
# ---------------------------------------------------------------------------

def test_service_reindex_wrapper(isolated):
    """service.reindex forwards to api.reindex and returns the same shape."""
    from taosmd import service

    data_dir = isolated
    stores = _setup(data_dir)
    vmem = stores["vector"]
    agent = "svc-agent"

    asyncio.run(taosmd.ingest("svc turn one", agent=agent, data_dir=str(data_dir)))
    asyncio.run(taosmd.ingest("svc turn two", agent=agent, data_dir=str(data_dir)))

    result = asyncio.run(service.reindex(agent=agent, data_dir=str(data_dir), check=False))
    assert result["archive_turns"] == 2
    assert result["reindexed_ok"] is True
    assert asyncio.run(_active_count(vmem, agent)) == 2


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_reindex_check_runs(isolated, monkeypatch, capsys):
    """`taosmd reindex --check` runs and prints without error."""
    from taosmd import cli

    data_dir = isolated
    _setup(data_dir)
    agent = "cli-agent"
    asyncio.run(taosmd.ingest("cli turn one", agent=agent, data_dir=str(data_dir)))

    rc = cli.main(["--data-dir", str(data_dir), "reindex", "--agent", agent, "--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert agent in out
    assert "archive=" in out
