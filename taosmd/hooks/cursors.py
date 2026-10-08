import logging
import sqlite3
import time

from taosmd._db import connect

logger = logging.getLogger(__name__)

_CURSOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS capture_cursors (
    session_id TEXT PRIMARY KEY,
    transcript_path TEXT NOT NULL,
    byte_offset INTEGER NOT NULL DEFAULT 0,
    last_entry_id TEXT,
    updated_at REAL NOT NULL
)
"""


class CursorStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        self._conn = connect(self.db_path)
        self._conn.execute(_CURSOR_SCHEMA)
        self._conn.commit()

    async def get(self, session_id: str):
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT session_id, transcript_path, byte_offset, last_entry_id, updated_at "
            "FROM capture_cursors WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "transcript_path": row[1],
            "byte_offset": row[2],
            "last_entry_id": row[3],
            "updated_at": row[4],
        }

    async def update(
        self,
        session_id: str,
        transcript_path: str,
        byte_offset: int,
        last_entry_id: str | None = None,
    ) -> None:
        if self._conn is None:
            return
        now = time.time()
        self._conn.execute(
            "INSERT INTO capture_cursors (session_id, transcript_path, byte_offset, last_entry_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (session_id) DO UPDATE SET "
            "transcript_path = excluded.transcript_path, "
            "byte_offset = excluded.byte_offset, "
            "last_entry_id = excluded.last_entry_id, "
            "updated_at = excluded.updated_at",
            (session_id, transcript_path, byte_offset, last_entry_id, now),
        )
        self._conn.commit()

    async def all_sessions(self) -> list[dict]:
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT session_id, transcript_path, byte_offset FROM capture_cursors"
        ).fetchall()
        return [
            {
                "session_id": r[0],
                "transcript_path": r[1],
                "byte_offset": r[2],
            }
            for r in rows
        ]

    async def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
