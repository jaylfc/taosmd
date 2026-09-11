"""Thread-affinity pin for ReceiptStore.

ReceiptStore is opened through ``_db.connect`` with the default
``check_same_thread=True`` because all production callers touch it only from
the ``_ServiceLoop`` service-loop thread. These tests verify that a genuine
cross-thread use raises ``sqlite3.ProgrammingError`` rather than silently
corrupting data, and that same-thread use continues to work.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading

from taosmd import receipts


def test_receipt_store_cross_thread_raises(tmp_path):
    """A connection created on one thread must raise when used on another.

    ReceiptStore uses ``check_same_thread=True`` (the default) so SQLite
    enforces single-thread access. A worker thread that tries to write
    through a connection created on the main thread must raise
    ``ProgrammingError``, not silently corrupt data.
    """
    db_path = str(tmp_path / "receipts.db")
    store = receipts.ReceiptStore(db_path=db_path)
    asyncio.run(store.init())  # connection created on the main thread
    outcome: dict = {}

    def worker() -> None:
        try:
            asyncio.run(store.record_delivered(1, "alice", 100.0))
            outcome["ok"] = True
        except sqlite3.ProgrammingError as exc:
            outcome["error"] = exc
        except Exception as exc:  # noqa: BLE001
            outcome["other"] = repr(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    asyncio.run(store.close())

    assert "error" in outcome, f"expected ProgrammingError, got {outcome}"
    assert outcome.get("ok") is not True
