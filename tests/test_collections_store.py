"""Tests for the collections store: CRUD, links, grants, archive.

Collections are first-class rows (not metadata tags): a named, typed
container for content indexed from a folder. The store enforces:

- ``kind`` in {docs, codebase, mixed}
- ``source_path`` must resolve inside a configured allowed root
  (``collections.allowed_roots``); empty roots means creation is refused
- links are typed rows {type: taos|git, id}, unique per collection
- grants are (canonical_id, scope='collection', collection_id) unique rows
- delete is archive (status='archived', reversible), never destruction
"""
from __future__ import annotations

import pytest

from taosmd import config
from taosmd.collections import (
    CollectionNotFoundError,
    CollectionStore,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TAOSMD_COLLECTIONS_ALLOWED_ROOTS", raising=False)
    d = tmp_path / "taosmd-data"
    d.mkdir()
    return str(d)


@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "docs-src"
    d.mkdir()
    (d / "readme.md").write_text("# Hello\n\nSome docs.")
    return str(d)


@pytest.fixture
def store(data_dir, source_dir):
    config.set_collections_allowed_roots([source_dir], data_dir=data_dir)
    return CollectionStore(data_dir)


# ---------------------------------------------------------------------------
# create / get / list
# ---------------------------------------------------------------------------

def test_create_returns_row_with_defaults(store, source_dir):
    col = store.create(name="repo docs", kind="docs", source_path=source_dir)
    assert col["id"].startswith("col-")
    assert len(col["id"]) == len("col-") + 12
    assert col["name"] == "repo docs"
    assert col["kind"] == "docs"
    assert col["status"] == "created"
    assert col["embedder"] is None  # default = global embedder
    assert col["last_indexed"] is None
    assert col["stats"] == {}
    assert col["links"] == []
    assert col["grants"] == []


def test_create_rejects_bad_kind(store, source_dir):
    with pytest.raises(ValueError):
        store.create(name="x", kind="movies", source_path=source_dir)


def test_create_requires_allowed_root(data_dir, source_dir):
    # No allowed_roots configured: collections are off.
    s = CollectionStore(data_dir)
    with pytest.raises(ValueError, match="allowed_roots"):
        s.create(name="x", kind="docs", source_path=source_dir)


def test_create_rejects_path_outside_roots(store, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ValueError):
        store.create(name="x", kind="docs", source_path=str(outside))


def test_create_rejects_missing_dir(store, source_dir):
    with pytest.raises(ValueError):
        store.create(name="x", kind="docs", source_path=source_dir + "/nope")


def test_create_stores_embedder(store, source_dir):
    col = store.create(
        name="x", kind="docs", source_path=source_dir, embedder="arctic-embed-s"
    )
    assert store.get(col["id"])["embedder"] == "arctic-embed-s"


def test_get_unknown_raises(store):
    with pytest.raises(CollectionNotFoundError):
        store.get("col-000000000000")


def test_list_and_get_round_trip(store, source_dir):
    a = store.create(name="a", kind="docs", source_path=source_dir)
    b = store.create(name="b", kind="mixed", source_path=source_dir)
    ids = {c["id"] for c in store.list()}
    assert ids == {a["id"], b["id"]}
    assert store.get(a["id"])["name"] == "a"


# ---------------------------------------------------------------------------
# links
# ---------------------------------------------------------------------------

def test_link_unlink_round_trip(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    store.link(col["id"], "taos", "prj-123")
    store.link(col["id"], "git", "abc123def456")
    links = store.get(col["id"])["links"]
    assert {"type": "taos", "id": "prj-123"} in links
    assert {"type": "git", "id": "abc123def456"} in links
    store.unlink(col["id"], "taos", "prj-123")
    links = store.get(col["id"])["links"]
    assert {"type": "taos", "id": "prj-123"} not in links
    assert {"type": "git", "id": "abc123def456"} in links


def test_link_is_idempotent(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    store.link(col["id"], "taos", "prj-123")
    store.link(col["id"], "taos", "prj-123")
    assert len(store.get(col["id"])["links"]) == 1


def test_link_rejects_bad_type(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    with pytest.raises(ValueError):
        store.link(col["id"], "jira", "PROJ-1")


def test_list_project_filter_matches_either_link_type(store, source_dir):
    a = store.create(name="a", kind="docs", source_path=source_dir)
    b = store.create(name="b", kind="docs", source_path=source_dir)
    store.link(a["id"], "taos", "prj-123")
    store.link(b["id"], "git", "abc123def456")
    assert [c["id"] for c in store.list(project="prj-123")] == [a["id"]]
    assert [c["id"] for c in store.list(project="abc123def456")] == [b["id"]]
    assert store.list(project="nothing") == []


# ---------------------------------------------------------------------------
# grants
# ---------------------------------------------------------------------------

def test_grant_revoke_round_trip(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    assert not store.has_grant("agent-a", col["id"])
    store.grant(col["id"], "agent-a")
    assert store.has_grant("agent-a", col["id"])
    assert "agent-a" in store.get(col["id"])["grants"]
    store.revoke(col["id"], "agent-a")
    assert not store.has_grant("agent-a", col["id"])


def test_grant_is_idempotent(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    store.grant(col["id"], "agent-a")
    store.grant(col["id"], "agent-a")
    assert store.get(col["id"])["grants"] == ["agent-a"]


def test_grant_unknown_collection_raises(store):
    with pytest.raises(CollectionNotFoundError):
        store.grant("col-000000000000", "agent-a")


# ---------------------------------------------------------------------------
# archive (delete alias) + status
# ---------------------------------------------------------------------------

def test_archive_is_reversible_status_change(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    out = store.archive(col["id"])
    assert out["status"] == "archived"
    # Row still exists: nothing destroyed.
    assert store.get(col["id"])["status"] == "archived"
    # Archived collections are hidden from the default listing…
    assert store.list() == []
    # …but visible when explicitly asked for.
    assert [c["id"] for c in store.list(include_archived=True)] == [col["id"]]


def test_set_status_and_stats(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    store.set_status(col["id"], "indexing")
    assert store.get(col["id"])["status"] == "indexing"
    store.set_status(col["id"], "error", error="boom")
    got = store.get(col["id"])
    assert got["status"] == "error"
    assert got["error"] == "boom"
    store.set_stats(col["id"], {"files_indexed": 3})
    store.set_status(col["id"], "ready", last_indexed=123.0)
    got = store.get(col["id"])
    assert got["stats"] == {"files_indexed": 3}
    assert got["last_indexed"] == 123.0
    assert got["error"] is None  # cleared on non-error status


def test_set_status_rejects_unknown(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    with pytest.raises(ValueError):
        store.set_status(col["id"], "exploded")


# ---------------------------------------------------------------------------
# file state (incremental re-index support)
# ---------------------------------------------------------------------------

def test_file_state_round_trip(store, source_dir):
    col = store.create(name="a", kind="docs", source_path=source_dir)
    assert store.file_states(col["id"]) == {}
    store.set_file_state(col["id"], "readme.md", "hash1")
    assert store.file_states(col["id"]) == {"readme.md": "hash1"}
    store.set_file_state(col["id"], "readme.md", "hash2")
    assert store.file_states(col["id"]) == {"readme.md": "hash2"}
    store.remove_file_state(col["id"], "readme.md")
    assert store.file_states(col["id"]) == {}
