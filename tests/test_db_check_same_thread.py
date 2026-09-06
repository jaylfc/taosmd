"""Tests for the ``check_same_thread`` parameter on :func:`_db.connect`.

Under the ThreadingHTTPServer + _ServiceLoop design every store connection is
created and used on the single service-loop thread, so the default
``check_same_thread=True`` is correct for all current call sites. But the
parameter must be a real, behavioural opt-in -- not a no-op -- so callers that
genuinely need cross-thread access (e.g. ReceiptStore today, future store
refactors tomorrow) can request it.

These tests prove the parameter actually changes sqlite3's behaviour: the
default raises ``sqlite3.ProgrammingError`` when a connection is touched from a
thread other than the one that opened it, while ``check_same_thread=False``
permits it.
"""

from __future__ import annotations

import sqlite3
import threading

from taosmd import _db


def test_default_check_same_thread_raises_on_foreign_thread(tmp_path):
    """Without the opt-in, sqlite3 enforces thread affinity.

    On master this raises ``sqlite3.ProgrammingError``; after the fix it still
    does, because the default stays ``True`` and no caller was changed.
    """
    db_path = str(tmp_path / "thread_bound.db")
    conn = _db.connect(db_path)
    result: dict = {}

    def worker():
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            result["count"] = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        except sqlite3.ProgrammingError as exc:
            result["error"] = exc

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)
    conn.close()

    assert "error" in result, "expected sqlite3.ProgrammingError on cross-thread use"
    assert "count" not in result, "connection should not have succeeded"


def test_check_same_thread_false_allows_cross_thread(tmp_path):
    """The explicit opt-in disables the thread-affinity check.

    Fails on master with ``TypeError`` (the parameter does not exist yet);
    passes after the fix.
    """
    db_path = str(tmp_path / "thread_free.db")
    conn = _db.connect(db_path, check_same_thread=False)
    result: dict = {}

    def worker():
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            result["count"] = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        except sqlite3.ProgrammingError as exc:
            result["error"] = exc

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)
    conn.close()

    assert "error" not in result, result.get("error")
    assert result.get("count") == 1
