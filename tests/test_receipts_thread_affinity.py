"""Thread-affinity pin for ReceiptStore.

ReceiptStore is opened through ``_db.connect(..., check_same_thread=False)``
because its async methods may be driven from any thread (the event loop is not
guaranteed to run on the creating thread). These tests fail with
``sqlite3.ProgrammingError: SQLite objects created in a thread can only be
used in that same thread`` if ``check_same_thread=False`` is dropped from the
connect call -- i.e. they are RED without that flag.
"""

from __future__ import annotations

import asyncio
import threading

from taosmd import receipts


def test_receipt_store_usable_from_non_creating_thread(tmp_path):
    """A connection created on one thread must work on another.

    RED without ``check_same_thread=False`` on the connect call: the worker
    thread's write raises ``ProgrammingError``. With the flag the write
    succeeds and the row is readable from the same worker thread.
    """
    db_path = str(tmp_path / "receipts.db")
    store = receipts.ReceiptStore(db_path=db_path)
    asyncio.run(store.init())  # connection created on the main thread
    outcome: dict = {}

    def worker() -> None:
        try:
            asyncio.run(store.record_delivered(1, "alice", 100.0))
            outcome["receipt"] = asyncio.run(store.get_receipt(1, "alice"))
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = repr(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    asyncio.run(store.close())

    assert "error" not in outcome, outcome.get("error")
    assert outcome["receipt"]["delivered_at"] == 100.0
