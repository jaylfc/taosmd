"""Tests for retrieval-time TTL filter (forget_after / forget_reason).

The forget_after field lives in user metadata; no schema change is needed.
Behaviour contract:
  - Expired row (forget_after < now): invisible to search() and search_bm25().
  - Future-dated row (forget_after > now): visible.
  - Non-numeric forget_after: ignored, row stays visible.
  - Row without forget_after: always visible.
  - Zero-loss: the raw row is never deleted; it still exists in the DB.
  - Batch ingest with forget_after round-trips through metadata.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

import taosmd
import taosmd.api as taosmd_api
from taosmd.vector_memory import VectorMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedder(vmem: VectorMemory) -> None:
    """Patch embed() with a deterministic 16-dim hash vector (no ONNX/QMD)."""

    async def _embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        return [((h >> (i * 3)) & 0xFF) / 255.0 - 0.5 for i in range(16)]

    vmem.embed = _embed  # type: ignore[assignment]


def _make_store(tmp_path) -> VectorMemory:
    vmem = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx",
    )
    asyncio.run(vmem.init())
    _fake_embedder(vmem)
    return vmem


def _row_count(db_path) -> int:
    con = sqlite3.connect(db_path)
    try:
        return con.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_expired_row_invisible_to_search(tmp_path):
    """An expired row is excluded from semantic search results."""
    vmem = _make_store(tmp_path)
    try:
        past = time.time() - 3600  # one hour ago
        rid = asyncio.run(
            vmem.add("expired fact about cats", metadata={"forget_after": past})
        )
        asyncio.run(vmem.add("active fact about dogs"))

        now = time.time()
        # Pass explicit now to exercise the clock override path.
        rows = vmem._load_active_rows(now=now)
        ids = {r["id"] for r in rows}
        assert rid not in ids, "expired row must be hidden from _load_active_rows"

        hits = asyncio.run(vmem.search("cats", limit=10, hybrid=False, fusion="none"))
        assert all(h["id"] != rid for h in hits), "expired row must be absent from search()"
    finally:
        asyncio.run(vmem.close())

    # Zero-loss: the row is still in the database.
    assert _row_count(tmp_path / "vec.db") == 2


def test_expired_row_invisible_to_search_bm25(tmp_path):
    """An expired row is excluded from BM25 search results."""
    vmem = _make_store(tmp_path)
    try:
        past = time.time() - 3600
        rid = asyncio.run(
            vmem.add("expired keyword text here", metadata={"forget_after": past})
        )
        asyncio.run(vmem.add("active keyword text here"))

        hits = asyncio.run(vmem.search_bm25("keyword", limit=10))
        assert all(h["id"] != rid for h in hits), "expired row must be absent from search_bm25()"
    finally:
        asyncio.run(vmem.close())


def test_future_dated_row_visible(tmp_path):
    """A row with forget_after in the future is visible in search results."""
    vmem = _make_store(tmp_path)
    try:
        future = time.time() + 86400  # tomorrow
        rid = asyncio.run(
            vmem.add("future fact about trains", metadata={"forget_after": future})
        )

        rows = vmem._load_active_rows()
        ids = {r["id"] for r in rows}
        assert rid in ids, "future-dated row must remain active"

        hits = asyncio.run(vmem.search("trains", limit=10, hybrid=False, fusion="none"))
        assert any(h["id"] == rid for h in hits), "future-dated row must appear in search()"
    finally:
        asyncio.run(vmem.close())


def test_non_numeric_forget_after_is_ignored(tmp_path):
    """A non-numeric forget_after value is silently ignored; row stays visible."""
    vmem = _make_store(tmp_path)
    try:
        rid = asyncio.run(
            vmem.add("row with bad ttl", metadata={"forget_after": "not-a-number"})
        )

        rows = vmem._load_active_rows()
        ids = {r["id"] for r in rows}
        assert rid in ids, "non-numeric forget_after must not hide the row"

        hits = asyncio.run(vmem.search("bad ttl", limit=10, hybrid=False, fusion="none"))
        assert any(h["id"] == rid for h in hits)
    finally:
        asyncio.run(vmem.close())


def test_row_without_forget_after_always_visible(tmp_path):
    """Rows with no forget_after field are unaffected by the TTL filter."""
    vmem = _make_store(tmp_path)
    try:
        rid = asyncio.run(vmem.add("plain row no ttl"))

        rows = vmem._load_active_rows()
        assert any(r["id"] == rid for r in rows)

        hits = asyncio.run(vmem.search("plain row", limit=5, hybrid=False, fusion="none"))
        assert any(h["id"] == rid for h in hits)
    finally:
        asyncio.run(vmem.close())


def test_forget_reason_stored_in_metadata(tmp_path):
    """forget_reason is stored in metadata and round-trips correctly."""
    vmem = _make_store(tmp_path)
    try:
        future = time.time() + 86400
        rid = asyncio.run(
            vmem.add(
                "fact with reason",
                metadata={"forget_after": future, "forget_reason": "test cleanup"},
            )
        )

        hits = asyncio.run(vmem.search("fact with reason", limit=5, hybrid=False, fusion="none"))
        matching = [h for h in hits if h["id"] == rid]
        assert matching, "row must be visible (future-dated)"
        meta = matching[0]["metadata"]
        assert meta.get("forget_reason") == "test cleanup"
        assert meta.get("forget_after") == future
    finally:
        asyncio.run(vmem.close())


def test_batch_items_with_forget_after_round_trip(tmp_path):
    """Batch-ingested items with forget_after respect the TTL at retrieval.

    This mirrors the REAL shape ``api.ingest_batch`` writes: the caller's
    per-item metadata (with ``source_id`` / ``forget_after``) is nested under
    ``meta["metadata"]`` alongside the top-level ``agent`` tag, not flattened.
    Encoding the flat shape here previously masked the batch TTL bug.
    """
    vmem = _make_store(tmp_path)
    try:
        past = time.time() - 60
        future = time.time() + 86400

        rid_expired = asyncio.run(
            vmem.add(
                "batch item expired",
                metadata={"agent": "a", "metadata": {"source_id": "b1", "forget_after": past}},
            )
        )
        rid_active = asyncio.run(
            vmem.add(
                "batch item active",
                metadata={"agent": "a", "metadata": {"source_id": "b2", "forget_after": future}},
            )
        )
        rid_plain = asyncio.run(
            vmem.add(
                "batch item no ttl",
                metadata={"agent": "a", "metadata": {"source_id": "b3"}},
            )
        )

        rows = vmem._load_active_rows()
        ids = {r["id"] for r in rows}
        assert rid_expired not in ids, "expired batch item must be hidden"
        assert rid_active in ids, "future-dated batch item must be visible"
        assert rid_plain in ids, "plain batch item must be visible"
    finally:
        asyncio.run(vmem.close())

    # All three raw rows must still exist (zero-loss).
    assert _row_count(tmp_path / "vec.db") == 3


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolated data dir + clean stores cache, mirroring the integrity suite."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"),
                      stores.get("kg"), stores.get("claims")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def test_ingest_batch_forget_after_expires_via_api(isolated):
    """End-to-end: forget_after supplied via api.ingest_batch actually expires.

    Exercises the true producer (not a hand-built row): ingest_batch nests the
    caller's per-item metadata under ``meta["metadata"]``, so a flat TTL filter
    silently never expires batch-supplied forget_after. This proves the fix.
    """
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _patch_embedder(stores)
    agent = "ttl-batch-agent"

    past = time.time() - 3600
    future = time.time() + 86400
    result = asyncio.run(taosmd.ingest_batch(
        [
            {"text": "expired batch fact about cats", "id": "e1",
             "metadata": {"forget_after": past}},
            {"text": "active batch fact about dogs", "id": "a1",
             "metadata": {"forget_after": future}},
            {"text": "plain batch fact about birds", "id": "p1"},
        ],
        agent=agent,
        data_dir=str(isolated),
    ))
    assert result["ingested"] == 3

    vmem = stores["vector"]
    rows = vmem._load_active_rows()
    active_texts = {r["text"] for r in rows}
    assert "expired batch fact about cats" not in active_texts, (
        "forget_after supplied via ingest_batch must hide the expired row")
    assert "active batch fact about dogs" in active_texts
    assert "plain batch fact about birds" in active_texts

    # Zero-loss: all three raw rows still exist in the DB.
    total = vmem._conn.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]
    assert total == 3


def test_ingest_batch_expired_item_still_dedupes(isolated):
    """A batch item whose forget_after has passed must still dedupe on re-POST.

    An expired row is only hidden from recall; it is physically present, so a
    re-import of the same id must be skipped (idempotency) rather than writing
    a second archive/vector row. Regression guard for the #188 TTL filter
    leaking into existing_source_ids() and breaking batch dedupe (zero-loss:
    unbounded archive duplication on every re-POST of an expired id)."""
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _patch_embedder(stores)
    agent = "ttl-dedupe-agent"
    vmem = stores["vector"]

    past = time.time() - 3600
    item = [{"text": "stale note", "id": "stale-1",
             "metadata": {"forget_after": past}}]

    r1 = asyncio.run(taosmd.ingest_batch(item, agent=agent, data_dir=str(isolated)))
    assert r1["ingested"] == 1 and r1["skipped"] == 0, r1

    r2 = asyncio.run(taosmd.ingest_batch(item, agent=agent, data_dir=str(isolated)))
    assert r2["ingested"] == 0, "expired item must not re-ingest"
    assert r2["skipped"] == 1, "expired item id must be deduped on re-POST"

    # Zero-loss / no duplication: exactly one physical row for the id.
    total = vmem._conn.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]
    assert total == 1, "re-POST of an expired id must not write a second row"


def test_existing_source_ids_includes_expired_rows(tmp_path):
    """existing_source_ids() must report source_ids of expired rows too.

    The set backs batch dedupe; an expired row still exists on disk, so its
    source_id must dedupe. A non-expired row is reported as before."""
    vmem = _make_store(tmp_path)
    try:
        past = time.time() - 3600
        future = time.time() + 86400
        asyncio.run(vmem.add(
            "expired sourced row",
            metadata={"agent": "a", "metadata": {"source_id": "expired-sid",
                                                 "forget_after": past}}))
        asyncio.run(vmem.add(
            "active sourced row",
            metadata={"agent": "a", "metadata": {"source_id": "active-sid",
                                                 "forget_after": future}}))

        sids = vmem.existing_source_ids(agent="a")
        assert "active-sid" in sids
        assert "expired-sid" in sids, (
            "expired-but-present row must still dedupe in existing_source_ids()")
    finally:
        asyncio.run(vmem.close())


def test_ttl_filter_uses_now_override(tmp_path):
    """Passing an explicit now value controls which rows are expired."""
    vmem = _make_store(tmp_path)
    try:
        fixed_ts = 1_000_000.0
        rid = asyncio.run(
            vmem.add("time-controlled row", metadata={"forget_after": fixed_ts + 1})
        )

        # Before expiry: row visible.
        rows_before = vmem._load_active_rows(now=fixed_ts)
        assert any(r["id"] == rid for r in rows_before)

        # After expiry: row hidden.
        rows_after = vmem._load_active_rows(now=fixed_ts + 2)
        assert all(r["id"] != rid for r in rows_after)
    finally:
        asyncio.run(vmem.close())
