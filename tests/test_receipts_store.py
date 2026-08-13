"""Tests for taosmd.receipts store WAL and busy-timeout wiring.

The ReceiptStore must use _db.connect so it shares the same WAL and
busy-timeout behaviour as every other store in the package.
"""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from taosmd.receipts import ReceiptStore


@pytest.mark.asyncio
async def test_receipts_store_uses_wal_mode(tmp_path):
    """ReceiptStore.init enables WAL mode via _db.connect."""
    db_path = str(tmp_path / "receipts.db")
    store = ReceiptStore(db_path)
    await store.init()
    try:
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        mode = (row[0] if row else "") or ""
        assert mode.lower() == "wal", f"expected WAL, got {mode!r}"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_receipts_store_sets_busy_timeout(tmp_path):
    """ReceiptStore.init sets a busy_timeout via _db.connect."""
    db_path = str(tmp_path / "receipts-busy.db")
    store = ReceiptStore(db_path)
    await store.init()
    try:
        row = store._conn.execute("PRAGMA busy_timeout").fetchone()
        assert row is not None
        timeout = row[0] if row else 0
        assert timeout >= 5000, f"expected busy_timeout >= 5000, got {timeout}"
    finally:
        await store.close()
