"""Mention index for the A2A bus.

Stores @handle mentions extracted from A2A message bodies and the optional
``recipient`` field, enabling the /a2a/mentions feed and thread-scoped
visibility (anti-bypass rule from taOSmd #211).
"""

from __future__ import annotations

import re
import sqlite3

from taosmd import _db


def _normalise_handle(handle: str) -> str:
    return handle.lstrip("@").casefold()


_MENTION_RE = re.compile(r'(?<![\w/])@([a-zA-Z0-9_-]+)')


class MentionStore:
    """Append-only mention index keyed by (mentioned_handle, message_id, ts, thread).

    All methods are async so callers can ``await`` them uniformly; the body
    runs synchronously against a single SQLite connection. The connection is
    opened through ``taosmd._db.connect`` (WAL journal mode, 5000 ms busy
    timeout) with ``check_same_thread=False`` so the connection stays usable
    from whichever thread drives the event loop -- this differs from the
    thread-affine stores and is required because the async methods are not
    guaranteed to run on the creating thread.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        self._conn = _db.connect(self._db_path, check_same_thread=False)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mentioned_handle TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                thread TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mentions_handle_ts
                ON mentions(mentioned_handle, ts);
        """)
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def record_mentions(
        self,
        message_id: int,
        body: str,
        thread: str,
        ts: float,
        recipient: str | None = None,
    ) -> None:
        handles: set[str] = set()
        for m in _MENTION_RE.finditer(body or ''):
            handles.add(_normalise_handle(m.group(1)))
        if recipient:
            handles.add(_normalise_handle(recipient))

        for handle in handles:
            self._conn.execute(
                "INSERT INTO mentions (mentioned_handle, message_id, ts, thread) VALUES (?, ?, ?, ?)",
                (handle, message_id, ts, thread),
            )
        if handles:
            self._conn.commit()

    async def get_mentioned_message_ids(
        self, reader: str, since: float | None = None, limit: int = 50
    ) -> list[dict]:
        norm = _normalise_handle(reader)
        query = "SELECT message_id, ts FROM mentions WHERE mentioned_handle = ?"
        params: list = [norm]
        if since is not None:
            query += " AND ts > ?"
            params.append(since)
        query += " ORDER BY ts ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [{"message_id": r[0], "ts": r[1]} for r in rows]

    async def get_mention_recipients(self, message_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT mentioned_handle FROM mentions WHERE message_id = ?",
            (message_id,),
        ).fetchall()
        return [r[0] for r in rows]
