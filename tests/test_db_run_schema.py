"""Deterministic tests for :func:`taosmd._db.run_schema`.

``run_schema`` retries idempotent schema DDL on transient ``SQLITE_BUSY`` /
``database is locked`` errors and propagates every other error. The three code
branches (retry-then-succeed, raise-on-non-lock, exhaustion) are pinned with a
fake connection so behaviour is reproducible instead of depending on a
timing-dependent multiprocess race.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3

import pytest

from taosmd import _db
from taosmd.mentions import MentionStore

WORKER_COUNT = 4


def _locked() -> sqlite3.OperationalError:
    return sqlite3.OperationalError("database is locked")


class _FakeConn:
    """Connection double that records calls and raises on demand.

    ``failures`` is consumed left-to-right: the i-th ``executescript`` call
    raises ``failures[i]`` when it is not ``None`` and otherwise succeeds,
    modelling a lock that clears on retry. The REAL ``sqlite3.OperationalError``
    type is used so ``run_schema``'s ``except sqlite3.OperationalError`` matches.
    """

    def __init__(self, failures) -> None:
        self._failures = list(failures)
        self.executed: list[str] = []
        self._i = 0

    def executescript(self, schema: str) -> None:
        self.executed.append(schema)
        if self._i < len(self._failures):
            failure = self._failures[self._i]
            self._i += 1
            if failure is not None:
                raise failure


def test_run_schema_succeeds_first_try(monkeypatch):
    """A clean run does one executescript and no sleeping."""
    slept: list[float] = []
    monkeypatch.setattr(_db.time, "sleep", lambda s: slept.append(s))
    conn = _FakeConn([None])
    _db.run_schema(conn, "CREATE TABLE t(x)")
    assert len(conn.executed) == 1
    assert slept == []


def test_run_schema_retries_on_busy_then_succeeds(monkeypatch):
    """Transient 'database is locked' is retried; the script re-runs when it clears."""
    monkeypatch.setattr(_db.time, "sleep", lambda _s: None)
    conn = _FakeConn([_locked(), None])
    _db.run_schema(conn, "CREATE TABLE t(x)")
    assert len(conn.executed) == 2


def test_run_schema_raises_on_non_lock_error():
    """A non-lock OperationalError propagates immediately, with no retry."""
    conn = _FakeConn([sqlite3.OperationalError("no such table: foo")])
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        _db.run_schema(conn, "CREATE TABLE t(x)")
    assert len(conn.executed) == 1


def test_run_schema_exhausts_on_persistent_lock(monkeypatch):
    """A persistently locked DB raises after SCHEMA_RETRY_ATTEMPTS attempts."""
    monkeypatch.setattr(_db.time, "sleep", lambda _s: None)
    conn = _FakeConn([_locked()] * _db.SCHEMA_RETRY_ATTEMPTS)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _db.run_schema(conn, "CREATE TABLE t(x)")
    assert len(conn.executed) == _db.SCHEMA_RETRY_ATTEMPTS


def test_run_schema_exhaustion_tracks_loop_bound(monkeypatch):
    """The exhaustion check is derived from the loop bound, not a literal.

    The earlier landmine compared ``attempt == 4`` while looping over
    ``range(5)``: shrinking the range to ``range(1)`` let the loop exit without
    raising and silently swallow a persistent lock error. With the bound
    shrunk to 1 the lock error must still raise on that single attempt, which
    would fail against the literal form.
    """
    monkeypatch.setattr(_db.time, "sleep", lambda _s: None)
    original = _db.SCHEMA_RETRY_ATTEMPTS
    _db.SCHEMA_RETRY_ATTEMPTS = 1
    try:
        conn = _FakeConn([_locked()])
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            _db.run_schema(conn, "CREATE TABLE t(x)")
        assert len(conn.executed) == 1
    finally:
        _db.SCHEMA_RETRY_ATTEMPTS = original


def _init_only(db_path: str, worker_id: int) -> None:
    """Init then close a MentionStore; exits non-zero on any failure."""
    store = MentionStore(db_path)
    try:
        asyncio.run(store.init())
        asyncio.run(store.close())
    except BaseException as exc:  # noqa: BLE001
        print(f"worker {worker_id}: {type(exc).__name__}: {exc}", flush=True)
        raise


def test_concurrent_first_init_is_deterministic(tmp_path):
    """Concurrent first-time init across processes must not raise.

    Each worker races ``MentionStore.init()`` -> ``_db.run_schema`` on a fresh
    DB. run_schema retries the transient ``database is locked`` produced by the
    CREATE/CREATE-INDEX DDL race, so every writer exits cleanly and the table
    is created with zero rows.

    The INSERT-phase row-count acceptance from the original (flaky) test is
    deliberately out of scope here: ``record_mentions`` performs unretried
    cross-process writes, and that row loss is a separate defect not addressed
    by the schema-DDL retry.
    """
    db_path = str(tmp_path / "mentions.db")
    ctx = multiprocessing.get_context("fork")
    procs = [
        ctx.Process(target=_init_only, args=(db_path, i))
        for i in range(WORKER_COUNT)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode is not None, f"worker {p.name} did not finish"

    failed = [p for p in procs if p.exitcode != 0]
    assert not failed, (
        f"{len(failed)} workers failed: {[(p.name, p.exitcode) for p in failed]}"
    )

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
        table = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='mentions'"
        ).fetchone()
    finally:
        conn.close()
    assert table is not None, "mentions table was not created"
    assert count == 0
