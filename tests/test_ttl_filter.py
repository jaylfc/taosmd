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
    """Batch-ingested items with forget_after respect the TTL at retrieval."""
    # Simulate the batch ingest path by adding items individually (same code
    # path as api.ingest_batch which calls vmem.add per item with its metadata).
    vmem = _make_store(tmp_path)
    try:
        past = time.time() - 60
        future = time.time() + 86400

        rid_expired = asyncio.run(
            vmem.add("batch item expired", metadata={"source_id": "b1", "forget_after": past})
        )
        rid_active = asyncio.run(
            vmem.add("batch item active", metadata={"source_id": "b2", "forget_after": future})
        )
        rid_plain = asyncio.run(vmem.add("batch item no ttl", metadata={"source_id": "b3"}))

        rows = vmem._load_active_rows()
        ids = {r["id"] for r in rows}
        assert rid_expired not in ids, "expired batch item must be hidden"
        assert rid_active in ids, "future-dated batch item must be visible"
        assert rid_plain in ids, "plain batch item must be visible"
    finally:
        asyncio.run(vmem.close())

    # All three raw rows must still exist (zero-loss).
    assert _row_count(tmp_path / "vec.db") == 3


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
