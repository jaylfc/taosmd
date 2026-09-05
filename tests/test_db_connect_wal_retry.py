"""Deterministic tests for the WAL-pragma retry and ordering in ``_db.connect``.

The init race in this package is real but probabilistic (~1% at load): multiple
processes call ``_db.connect`` on the same fresh database and contend on the
brief exclusive lock ``PRAGMA journal_mode=WAL`` needs. That lock is taken
*before* ``busy_timeout`` is armed on the connection, and SQLite does not drive
the busy handler for the journal-mode switch itself, so the switch raises
``sqlite3.OperationalError: database is locked`` immediately.

The fix arms ``busy_timeout`` first (so the handler covers everything that
follows) and then retries the WAL pragma itself on transient lock errors,
re-raising anything else immediately, and never raising merely because the
connection fell back to another journal mode (it warns instead).

These tests drive ``connect`` with a configurable in-memory fake connection so
the retry branches are exercised deterministically rather than waiting on a
flaky ~1% race. The genuine concurrency reproduction lives in
``scripts/probe_wal_dual_arm.py``.
"""
from __future__ import annotations

import sqlite3
import warnings
from unittest.mock import patch

import pytest

from taosmd import _db


class _FakeResult:
    """Row-set stand-in for the value returned by ``Connection.execute``."""

    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConn:
    """Configurable connection used in place of ``sqlite3.connect``.

    ``wal_outcomes`` is a list of per-call outcomes for ``PRAGMA journal_mode``
    in call order. Each item is either a row tuple (e.g. ``("wal",)``) or a
    ``sqlite3.OperationalError`` instance to raise. After the list is consumed
    the last outcome repeats, so a single-element ``[locked_exc]`` models
    "always fails". Every SQL string passed to ``execute`` is recorded in
    ``self.executed`` in order, which lets the tests assert on call sequence and
    attempt counts.
    """

    def __init__(self, wal_outcomes):
        self.wal_outcomes = list(wal_outcomes)
        self.executed: list[str] = []
        self.wal_attempts = 0
        self.check_same_thread = True

    def execute(self, sql, *args, **kwargs):
        self.executed.append(sql)
        lowered = sql.strip().lower()
        if lowered.startswith("pragma busy_timeout"):
            return _FakeResult(None)
        if lowered.startswith("pragma journal_mode"):
            self.wal_attempts += 1
            idx = min(self.wal_attempts - 1, len(self.wal_outcomes) - 1)
            outcome = self.wal_outcomes[idx]
            if isinstance(outcome, BaseException):
                raise outcome
            return _FakeResult(outcome)
        return _FakeResult(None)

    def close(self):
        pass


def _install_fake(monkeypatch, fake):
    """Patch the specific function ``_db`` calls: ``sqlite3.connect``.

    Only the ``connect`` function is replaced, so ``sqlite3.OperationalError``
    stays the real class and ``except sqlite3.OperationalError`` in the code
    under test keeps working. The fake is returned for every argument shape
    ``_db.connect``'s call may use.
    """
    factory = lambda *args, **kwargs: fake  # noqa: E731
    monkeypatch.setattr(_db.sqlite3, "connect", factory)
    return fake


# ----------------------------------------------------------------------
# ordering: busy_timeout must be armed before the WAL pragma
# ----------------------------------------------------------------------

def test_busy_timeout_armed_before_wal_pragma(monkeypatch, tmp_path):
    fake = _install_fake(monkeypatch, FakeConn([("wal",)]))
    _db.connect(str(tmp_path / "o.db"))
    busy_idx = _first_index(fake.executed, "pragma busy_timeout")
    wal_idx = _first_index(fake.executed, "pragma journal_mode=w")
    # Both must have run ...
    assert busy_idx is not None, "busy_timeout was never set"
    assert wal_idx is not None, "WAL pragma was never set"
    # ... and busy_timeout strictly precedes the WAL pragma so the busy handler
    # is armed before the contended journal-mode switch.
    assert busy_idx < wal_idx, (
        "busy_timeout ran AFTER journal_mode=WAL; the switch is unguarded"
    )


def _first_index(items, needle):
    lowered = needle.lower()
    for i, s in enumerate(items):
        if lowered in s.lower():
            return i
    return None


# ----------------------------------------------------------------------
# retry: lock-busy errors are retried, other errors are not
# ----------------------------------------------------------------------

def test_wal_pragma_retried_on_locked_then_succeeds(monkeypatch, tmp_path):
    locked = sqlite3.OperationalError("database is locked")
    fake = _install_fake(
        monkeypatch, FakeConn([locked, locked, ("wal",)])
    )
    conn = _db.connect(str(tmp_path / "r.db"))
    try:
        assert fake.wal_attempts == 3, (
            f"expected 3 WAL attempts (2 locked + 1 success), got {fake.wal_attempts}"
        )
    finally:
        conn.close()


def test_wal_pragma_uses_busy_keyword_for_retry(monkeypatch, tmp_path):
    """Errors mentioning 'busy' (not just 'locked') are retried too."""
    busy_exc = sqlite3.OperationalError("database is busy")
    fake = _install_fake(monkeypatch, FakeConn([busy_exc, ("wal",)]))
    conn = _db.connect(str(tmp_path / "b.db"))
    try:
        assert fake.wal_attempts == 2
    finally:
        conn.close()


def test_non_lock_error_is_not_retried(monkeypatch, tmp_path):
    """A non-lock OperationalError propagates on the first attempt, no retry."""
    other = sqlite3.OperationalError("no such table: phantom")
    fake = _install_fake(monkeypatch, FakeConn([other]))
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        _db.connect(str(tmp_path / "n.db"))
    assert fake.wal_attempts == 1, "non-lock error must not be retried"


def test_wal_pragma_exhaustion_raises_after_attempts(monkeypatch, tmp_path):
    """A persistent lock error surfaces the exception after bounded attempts."""
    locked = sqlite3.OperationalError("database is locked")
    fake = _install_fake(monkeypatch, FakeConn([locked]))  # repeats forever
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _db.connect(str(tmp_path / "e.db"))
    assert fake.wal_attempts == _db.WAL_RETRY_ATTEMPTS, (
        f"expected exactly {_db.WAL_RETRY_ATTEMPTS} attempts before raising, "
        f"got {fake.wal_attempts}"
    )


# ----------------------------------------------------------------------
# fallback: a non-WAL mode warns rather than raises; the conn is usable
# ----------------------------------------------------------------------

def test_wal_fallback_mode_warns_not_raises(monkeypatch, tmp_path):
    fake = _install_fake(monkeypatch, FakeConn([("delete",)]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conn = _db.connect(str(tmp_path / "f.db"))
    try:
        assert len(conn.executed) > 0
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert runtime, "a non-WAL fallback must emit a RuntimeWarning"
        assert "journal_mode" in str(runtime[0].message)
    finally:
        conn.close()


def test_wal_mode_no_warning(monkeypatch, tmp_path):
    fake = _install_fake(monkeypatch, FakeConn([("wal",)]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conn = _db.connect(str(tmp_path / "w.db"))
    try:
        assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    finally:
        conn.close()


def test_memory_mode_no_warning(monkeypatch, tmp_path):
    fake = _install_fake(monkeypatch, FakeConn([("memory",)]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conn = _db.connect(str(tmp_path / "m.db"))
    try:
        assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    finally:
        conn.close()
