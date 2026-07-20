"""Tests for the collections folder walker, chunker, and ingest pipeline.

The walker is gitignore-aware (simplified stdlib fnmatch rules), skips
version-control/binary/oversized files, and only ingests files an existing
loader explicitly claims (DocLoader md/txt/markdown/rst plus the other
registered loaders). ingest_folder routes chunks through api.ingest_batch
under the collection's own agent namespace, so re-index dedups on chunk
content hashes and changed/deleted files are superseded, never destroyed.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from taosmd import api as taosmd_api
from taosmd import config
from taosmd.collections import (
    CollectionStore,
    chunk_text,
    collect_files,
    ingest_folder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "readme.md").write_text("# Widget\n\nThe widget frobnicates the sprocket.")
    (d / "guide.txt").write_text("Install with pip. Configure the flux capacitor.")
    (d / "notes.rst").write_text("Deployment notes\n----------------\n\nUse systemd.")
    sub = d / "deep"
    sub.mkdir()
    (sub / "inner.md").write_text("# Inner\n\nNested documentation file.")
    return d


@pytest.fixture
def data_dir(tmp_path, monkeypatch, source_dir):
    d = tmp_path / "taosmd-data"
    d.mkdir()
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(d))
    monkeypatch.delenv("TAOSMD_COLLECTIONS_ALLOWED_ROOTS", raising=False)
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    config.set_collections_allowed_roots([str(source_dir.parent)], data_dir=str(d))
    return str(d)


def _patch_embedder(data_dir) -> None:
    """Deterministic 8-dim hash embedder, same pattern as test_http_server."""
    stores = asyncio.run(taosmd_api._ensure_stores(data_dir))
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------

def test_collect_files_finds_claimed_docs(source_dir):
    files, skips = collect_files(source_dir)
    rels = sorted(rel for _, rel in files)
    assert rels == ["deep/inner.md", "guide.txt", "notes.rst", "readme.md"]


def test_collect_files_skips_unclaimed_extensions(source_dir):
    (source_dir / "main.py").write_text("print('hi')")
    (source_dir / "data.json").write_text("{}")
    files, skips = collect_files(source_dir)
    rels = {rel for _, rel in files}
    assert "main.py" not in rels
    assert "data.json" not in rels
    assert skips["skipped_unclaimed"] == 2


def test_collect_files_skips_vcs_and_dep_dirs(source_dir):
    git = source_dir / ".git"
    git.mkdir()
    (git / "config.txt").write_text("secret")
    nm = source_dir / "node_modules"
    nm.mkdir()
    (nm / "pkg.md").write_text("# dep readme")
    files, _ = collect_files(source_dir)
    rels = {rel for _, rel in files}
    assert not any(r.startswith(".git") or r.startswith("node_modules") for r in rels)


def test_collect_files_respects_gitignore(source_dir):
    (source_dir / ".gitignore").write_text("*.log\nbuild/\n/secret.md\n")
    (source_dir / "debug.log").write_text("log log log")
    (source_dir / "secret.md").write_text("# do not index")
    build = source_dir / "build"
    build.mkdir()
    (build / "out.md").write_text("# generated")
    # A nested .gitignore applies within its own directory.
    (source_dir / "deep" / ".gitignore").write_text("inner.md\n")
    files, skips = collect_files(source_dir)
    rels = {rel for _, rel in files}
    assert "debug.log" not in rels
    assert "secret.md" not in rels
    assert "build/out.md" not in rels
    assert "deep/inner.md" not in rels
    assert "readme.md" in rels
    assert skips["skipped_ignored"] >= 3


def test_collect_files_skips_binary_and_oversized(source_dir):
    (source_dir / "logo.png").write_bytes(b"\x89PNG\r\n" + b"\x00" * 32)
    # A claimed extension whose content is binary is sniffed out.
    (source_dir / "fake.md").write_bytes(b"\x00\x01\x02 binary sneaking as md")
    (source_dir / "huge.md").write_text("x" * 4096)
    files, skips = collect_files(source_dir, max_file_bytes=1024)
    rels = {rel for _, rel in files}
    assert "logo.png" not in rels
    assert "fake.md" not in rels
    assert "huge.md" not in rels
    assert skips["skipped_binary"] >= 2
    assert skips["skipped_size"] == 1


def test_collect_files_rejects_trees_over_the_file_cap(source_dir):
    """The walker has a per-file size cap but also needs a tree cap: pointing
    a collection at a huge tree must fail fast with a clear message instead
    of grinding through it."""
    with pytest.raises(ValueError, match="file cap"):
        collect_files(source_dir, max_files=2)
    # At or under the cap is fine (the fixture tree has 4 claimed files).
    files, _ = collect_files(source_dir, max_files=4)
    assert len(files) == 4


def test_ingest_folder_errors_cleanly_over_file_cap(data_dir, source_dir):
    store, col = _make_collection(data_dir, source_dir)
    with pytest.raises(ValueError, match="file cap"):
        asyncio.run(ingest_folder(col["id"], data_dir=data_dir, max_files=2))
    got = store.get(col["id"])
    assert got["status"] == "error"
    assert "file cap" in (got["error"] or "")


def test_collect_files_skips_symlink_escape(source_dir, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("# outside the root")
    os.symlink(outside, source_dir / "escape.md")
    files, skips = collect_files(source_dir)
    rels = {rel for _, rel in files}
    assert "escape.md" not in rels


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def test_chunk_text_short_is_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_text_packs_paragraphs():
    paras = [f"Paragraph {i} " + "word " * 30 for i in range(10)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, max_chars=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # Nothing lost: every paragraph's marker appears in exactly one chunk.
    joined = "\n\n".join(chunks)
    for i in range(10):
        assert f"Paragraph {i} " in joined


def test_chunk_text_hard_splits_long_paragraph():
    text = "word " * 1000  # one giant paragraph, no blank lines
    chunks = chunk_text(text, max_chars=800)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


# ---------------------------------------------------------------------------
# ingest_folder
# ---------------------------------------------------------------------------

def _make_collection(data_dir, source_dir):
    store = CollectionStore(data_dir)
    return store, store.create(name="docs", kind="docs", source_path=str(source_dir))


def test_ingest_folder_indexes_docs(data_dir, source_dir):
    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)
    stats = asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    assert stats["files_indexed"] == 4
    assert stats["chunks_ingested"] >= 4
    got = store.get(col["id"])
    assert got["status"] == "ready"
    assert got["last_indexed"] is not None
    assert got["stats"]["files_indexed"] == 4
    # Chunk rows carry collection metadata for provenance.
    hits = asyncio.run(
        taosmd_api.search(
            "widget frobnicates", agent="dev", mode="bm25",
            collections=[col["id"]], collections_only=True, data_dir=data_dir,
        )
    )
    # No grant yet: nothing visible.
    assert hits == []
    store.grant(col["id"], "dev")
    hits = asyncio.run(
        taosmd_api.search(
            "widget frobnicates", agent="dev", mode="bm25",
            collections=[col["id"]], collections_only=True, data_dir=data_dir,
        )
    )
    assert hits
    top = hits[0]
    assert top["metadata"]["collection_id"] == col["id"]
    assert top["metadata"]["file_path"] == "readme.md"
    assert top["metadata"]["source"] == "collection"


def test_reindex_unchanged_skips_everything(data_dir, source_dir):
    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    stats2 = asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    assert stats2["files_indexed"] == 0
    assert stats2["files_unchanged"] == 4
    assert stats2["chunks_ingested"] == 0


def test_reindex_changed_file_supersedes_old_rows(data_dir, source_dir):
    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)
    store.grant(col["id"], "dev")
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    (source_dir / "readme.md").write_text("# Widget\n\nThe widget now defenestrates.")
    stats2 = asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    assert stats2["files_indexed"] == 1
    assert stats2["files_unchanged"] == 3
    assert stats2["chunks_superseded"] >= 1

    def _search(q):
        return asyncio.run(
            taosmd_api.search(
                q, agent="dev", mode="bm25",
                collections=[col["id"]], collections_only=True, data_dir=data_dir,
            )
        )

    # The old content is out of active recall; the new content is in.
    assert not any("frobnicates" in h["text"] for h in _search("frobnicates sprocket"))
    assert any("defenestrates" in h["text"] for h in _search("defenestrates"))
    # Zero-loss: the superseded row still exists physically (valid_to set).
    stores = asyncio.run(taosmd_api._ensure_stores(data_dir))
    row = stores["vector"]._conn.execute(
        "SELECT COUNT(*) AS n FROM vector_memory WHERE valid_to IS NOT NULL"
    ).fetchone()
    assert row["n"] >= 1


def test_reindex_deleted_file_supersedes_rows(data_dir, source_dir):
    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)
    store.grant(col["id"], "dev")
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    (source_dir / "guide.txt").unlink()
    stats2 = asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    assert stats2["files_deleted"] == 1
    hits = asyncio.run(
        taosmd_api.search(
            "flux capacitor", agent="dev", mode="bm25",
            collections=[col["id"]], collections_only=True, data_dir=data_dir,
        )
    )
    assert not any("flux capacitor" in h["text"] for h in hits)
    assert "guide.txt" not in store.file_states(col["id"])


def test_reindex_emptied_file_supersedes_rows(data_dir, source_dir):
    """A previously-indexed file whose content is emptied must lose its rows.

    Zero-loss cuts both ways: the old text has to leave active recall (it no
    longer exists in the source) while the physical rows survive as
    superseded history. Skipping blank files outright left the stale content
    searchable forever and the hash state pointing at content that is gone.
    """
    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)
    store.grant(col["id"], "dev")
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))

    def _search(q):
        return asyncio.run(
            taosmd_api.search(
                q, agent="dev", mode="bm25",
                collections=[col["id"]], collections_only=True, data_dir=data_dir,
            )
        )

    assert any("frobnicates" in h["text"] for h in _search("frobnicates sprocket"))

    # Truncate to whitespace: the file still exists, but has no content.
    (source_dir / "readme.md").write_text("   \n\n  \n")
    stats2 = asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    assert stats2["chunks_superseded"] >= 1

    # The stale content is out of active recall.
    assert not any("frobnicates" in h["text"] for h in _search("frobnicates sprocket"))
    # And its hash state is cleared, so a later refill re-indexes cleanly.
    assert "readme.md" not in store.file_states(col["id"])
    # Zero-loss: superseded, not hard-deleted.
    stores = asyncio.run(taosmd_api._ensure_stores(data_dir))
    rows = stores["vector"]._conn.execute(
        "SELECT metadata_json FROM vector_memory WHERE valid_to IS NOT NULL"
    ).fetchall()
    assert rows
    assert any("collection-reindex:" in (r["metadata_json"] or "") for r in rows)


def test_reindex_already_empty_file_is_a_no_op(data_dir, source_dir):
    """A file that was blank and is still blank must not churn on re-index."""
    _patch_embedder(data_dir)
    (source_dir / "blank.md").write_text("\n\n   \n")
    store, col = _make_collection(data_dir, source_dir)
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    stats2 = asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    assert stats2["chunks_superseded"] == 0
    assert stats2["files_deleted"] == 0
    assert stats2["files_indexed"] == 0


def _closing_store_spy(monkeypatch):
    """Patch CollectionStore so every instance records whether it was closed."""
    from taosmd import collections as collections_mod

    real = collections_mod.CollectionStore
    made: list = []

    class _Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed = False
            made.append(self)

        def close(self) -> None:
            super().close()
            self.closed = True

    monkeypatch.setattr(collections_mod, "CollectionStore", _Spy)
    return made


def test_ingest_folder_closes_the_store_on_success(data_dir, source_dir, monkeypatch):
    """ingest_folder opens its own CollectionStore; a server that re-indexes
    on a timer would leak one sqlite connection per run if it never closed."""
    _patch_embedder(data_dir)
    _store, col = _make_collection(data_dir, source_dir)
    made = _closing_store_spy(monkeypatch)

    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))

    assert made, "ingest_folder did not open a CollectionStore"
    assert all(s.closed for s in made)
    for s in made:
        with pytest.raises(Exception):
            s._conn.execute("SELECT 1")


def test_ingest_folder_closes_the_store_on_error(data_dir, source_dir, monkeypatch):
    """The failure path (status=error) must close the connection too."""
    from taosmd import collections as collections_mod

    _patch_embedder(data_dir)
    _store, col = _make_collection(data_dir, source_dir)
    made = _closing_store_spy(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("walk exploded")

    monkeypatch.setattr(collections_mod, "collect_files", _boom)

    with pytest.raises(RuntimeError, match="walk exploded"):
        asyncio.run(ingest_folder(col["id"], data_dir=data_dir))

    assert made, "ingest_folder did not open a CollectionStore"
    assert all(s.closed for s in made)


def test_ingest_folder_walks_off_the_event_loop(data_dir, source_dir, monkeypatch):
    """The 202+poll contract promises a responsive server during an index:
    the filesystem walk (os.walk + per-file stat + null-byte sniff) must run
    on a worker thread, not on the single service loop thread where it would
    block /search and /ingest for the duration."""
    import threading

    from taosmd import collections as collections_mod

    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)

    seen: dict = {}
    real_collect = collections_mod.collect_files

    def _spy(*args, **kwargs):
        seen["thread"] = threading.current_thread()
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(collections_mod, "collect_files", _spy)

    async def _run():
        seen["loop_thread"] = threading.current_thread()
        return await ingest_folder(col["id"], data_dir=data_dir)

    stats = asyncio.run(_run())
    # The pipeline still completes correctly end to end.
    assert stats["files_indexed"] == 4
    assert store.get(col["id"])["status"] == "ready"
    # And the walk ran off the loop thread.
    assert "thread" in seen
    assert seen["thread"] is not seen["loop_thread"], (
        "collect_files ran on the event loop thread; the async index blocks the server"
    )


def test_ingest_folder_archived_collection_refused(data_dir, source_dir):
    store, col = _make_collection(data_dir, source_dir)
    store.archive(col["id"])
    with pytest.raises(ValueError):
        asyncio.run(ingest_folder(col["id"], data_dir=data_dir))


def test_ingest_folder_root_removed_from_config_refused(data_dir, source_dir):
    store, col = _make_collection(data_dir, source_dir)
    config.set_collections_allowed_roots([], clear=True, data_dir=data_dir)
    with pytest.raises(ValueError):
        asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    assert store.get(col["id"])["status"] == "error"


def test_index_start_rejects_concurrent_index(data_dir, source_dir):
    """collections_index_start must refuse to double-start: a second start
    while the collection is already ``indexing`` raises CollectionBusyError
    (409 over HTTP); ready and error states re-arm it."""
    from taosmd import service
    from taosmd.collections import CollectionBusyError

    store, col = _make_collection(data_dir, source_dir)
    receipt = asyncio.run(service.collections_index_start(col["id"], data_dir=data_dir))
    assert receipt["status"] == "indexing"
    with pytest.raises(CollectionBusyError):
        asyncio.run(service.collections_index_start(col["id"], data_dir=data_dir))

    # A finished index (ready) re-arms the start.
    store.set_status(col["id"], "ready")
    receipt = asyncio.run(service.collections_index_start(col["id"], data_dir=data_dir))
    assert receipt["status"] == "indexing"

    # So does a failed one (error): retries stay possible.
    store.set_status(col["id"], "error", error="boom")
    receipt = asyncio.run(service.collections_index_start(col["id"], data_dir=data_dir))
    assert receipt["status"] == "indexing"


# ---------------------------------------------------------------------------
# Search integration
# ---------------------------------------------------------------------------

def test_search_without_collections_excludes_collection_rows(data_dir, source_dir):
    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)
    store.grant(col["id"], "dev")
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    hits = asyncio.run(
        taosmd_api.search("widget frobnicates", agent="dev", mode="bm25", data_dir=data_dir)
    )
    assert not any(h["metadata"].get("collection_id") for h in hits)


def test_search_with_collections_merges_own_memory(data_dir, source_dir):
    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)
    store.grant(col["id"], "dev")
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    asyncio.run(
        taosmd_api.ingest("The widget owner is Jay.", agent="dev", data_dir=data_dir)
    )
    hits = asyncio.run(
        taosmd_api.search(
            "widget", agent="dev", mode="bm25", limit=10,
            collections=[col["id"]], data_dir=data_dir,
        )
    )
    sources = {h["metadata"].get("collection_id") for h in hits}
    assert col["id"] in sources       # collection hit present
    assert None in sources            # own conversational memory present too


def test_search_archived_collection_hidden(data_dir, source_dir):
    _patch_embedder(data_dir)
    store, col = _make_collection(data_dir, source_dir)
    store.grant(col["id"], "dev")
    asyncio.run(ingest_folder(col["id"], data_dir=data_dir))
    store.archive(col["id"])
    hits = asyncio.run(
        taosmd_api.search(
            "widget frobnicates", agent="dev", mode="bm25",
            collections=[col["id"]], collections_only=True, data_dir=data_dir,
        )
    )
    assert hits == []
