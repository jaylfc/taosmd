"""Shared SQLite connection helper.

Every persistent store in the package opens its own SQLite database. By
default ``sqlite3.connect`` uses the rollback-journal mode, which takes an
exclusive lock for the duration of a write and lets only a single writer
touch the file at a time. When several agents (or several processes on the
same machine) share a memory store, that serialisation surfaces as
``SQLITE_BUSY`` errors.

``connect`` centralises the fix: it enables write-ahead logging (WAL), which
lets readers proceed concurrently with a writer, and sets a busy timeout so a
contended writer waits and retries rather than failing immediately. Both are
plain PRAGMAs with no extra dependencies; standalone behaviour is unchanged
apart from the journal mode.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

# Allow a contended connection to block-and-retry for this many milliseconds
# before raising ``sqlite3.OperationalError: database is locked``.
BUSY_TIMEOUT_MS = 5000


def connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Open a SQLite connection in WAL mode with a busy timeout.

    Drop-in replacement for ``sqlite3.connect(db_path)``. Callers that need a
    ``row_factory`` or other connection attributes should set them on the
    returned connection as before.
    """
    conn = sqlite3.connect(db_path)
    # WAL is harmless (silently stays "memory") on ``:memory:`` databases.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return conn
