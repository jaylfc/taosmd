"""Read-only store-format probe (Card C of the f32 upgrade-safety audit).

The probe inspects a data directory's databases WITHOUT writing anything:
it opens every database with ``sqlite3.connect("file:<path>?mode=ro&immutable=1",
uri=True)`` so no ``-wal``/``-shm`` sidecars are created and no PRAGMA writes
run. ``migrations.status_all`` is the thing that fails today: it routes
through ``_db.connect`` (``migrations.py:696``), which runs
``PRAGMA journal_mode=WAL`` (``_db.py:64``), so a read-only status check on a
WAL database creates sidecars.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from taosmd import migrations
from taosmd.archive import INDEX_SCHEMA


def _sidecars(d):
    return sorted(f for f in os.listdir(d) if f.endswith(("-wal", "-shm")))


def test_status_does_not_create_wal_files(tmp_path):
    """A read-only status check must not create -wal/-shm sidecars.

    Fails today: ``migrations.status_all`` opens each database through
    ``_db.connect`` (``migrations.py:696``), which runs
    ``PRAGMA journal_mode=WAL`` (``_db.py:64``). Opening a WAL database in
    WAL mode creates sidecars even for a read-only check.
    """
    path = tmp_path / "archive-index.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(INDEX_SCHEMA)
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    for f in os.listdir(tmp_path):
        if f.endswith(("-wal", "-shm")):
            (tmp_path / f).unlink()

    before = set(os.listdir(tmp_path))

    rows = migrations.status_all(tmp_path)
    existing = [r for r in rows if r["exists"]]
    assert len(existing) == 1
    assert existing[0]["db"] == "archive_index"

    after = set(os.listdir(tmp_path))
    assert before == after, (
        f"status_all created sidecars: {sorted(after - before)}"
    )
    assert _sidecars(tmp_path) == [], "no -wal/-shm sidecars may exist after a read-only status check"


def test_status_all_does_not_create_sidecars_on_wal_db(tmp_path):
    """Same contract, stated positively: sidecar set is empty after the call."""
    path = tmp_path / "archive-index.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(INDEX_SCHEMA)
    conn.commit()
    conn.close()

    migrations.status_all(tmp_path)

    assert _sidecars(tmp_path) == [], (
        "status_all must not create -wal/-shm sidecars on a WAL database"
    )