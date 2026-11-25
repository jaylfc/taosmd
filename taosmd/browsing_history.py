"""Lightweight browsing history for platform apps (Reddit, X, etc.).

NOT a KnowledgeItem: no ingest, no embedding, no monitoring.
Just a breadcrumb trail of recently viewed content for quick recovery.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from . import _db

SCHEMA = """
CREATE TABLE IF NOT EXISTS browsing_history (
    url TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    preview TEXT NOT NULL DEFAULT '',
    extra_json TEXT NOT NULL DEFAULT '{}',
    viewed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_source ON browsing_history(source_type);
CREATE INDEX IF NOT EXISTS idx_history_viewed ON browsing_history(viewed_at DESC);
"""

ARCHIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS browsing_history_archive (
    url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    preview TEXT NOT NULL DEFAULT '',
    extra_json TEXT NOT NULL DEFAULT '{}',
    viewed_at REAL NOT NULL,
    archived_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_archive_source ON browsing_history_archive(source_type);
CREATE INDEX IF NOT EXISTS idx_history_archive_viewed ON browsing_history_archive(viewed_at DESC);
CREATE TABLE IF NOT EXISTS archived_browsing_history (
    url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    preview TEXT NOT NULL DEFAULT '',
    extra_json TEXT NOT NULL DEFAULT '{}',
    viewed_at REAL NOT NULL,
    archived_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS browsing_history_archived (
    url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    preview TEXT NOT NULL DEFAULT '',
    extra_json TEXT NOT NULL DEFAULT '{}',
    viewed_at REAL NOT NULL,
    archived_at REAL NOT NULL
);
"""

# Keep forever by default: browsing history is never deleted automatically.
DEFAULT_RETENTION_DAYS = None


class BrowsingHistoryStore:
    def __init__(self, db_path: str | Path = "data/browsing-history.db"):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _db.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        _db.run_schema(self._conn, SCHEMA)
        _db.run_schema(self._conn, ARCHIVE_SCHEMA)
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def record(
        self,
        url: str,
        source_type: str,
        title: str = "",
        author: str = "",
        preview: str = "",
        extra_json: str = "{}",
    ) -> None:
        """Upsert a viewed item. Updates viewed_at if already exists."""
        now = time.time()
        self._conn.execute(
            """INSERT INTO browsing_history (url, source_type, title, author, preview, extra_json, viewed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(url) DO UPDATE SET
                 title = excluded.title,
                 author = excluded.author,
                 preview = excluded.preview,
                 extra_json = excluded.extra_json,
                 viewed_at = excluded.viewed_at""",
            (url, source_type, title, author, preview[:200], extra_json, now),
        )
        self._conn.commit()

    async def list_recent(
        self,
        source_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List recently viewed items, newest first."""
        if source_type:
            rows = self._conn.execute(
                "SELECT * FROM browsing_history WHERE source_type = ? ORDER BY viewed_at DESC LIMIT ?",
                (source_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM browsing_history ORDER BY viewed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    async def list_archived(
        self,
        source_type: str | None = None,
        limit: int = 5000,
    ) -> list[dict]:
        """List archived (pruned) rows, newest first. Rows are recoverable here."""
        if source_type:
            rows = self._conn.execute(
                "SELECT * FROM browsing_history_archive WHERE source_type = ? ORDER BY viewed_at DESC LIMIT ?",
                (source_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM browsing_history_archive ORDER BY viewed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    async def archived(self, source_type=None, limit=5000):
        """Alias for list_archived."""
        return await self.list_archived(source_type=source_type, limit=limit)

    async def list_pruned(self, source_type=None, limit=5000):
        """Alias for list_archived."""
        return await self.list_archived(source_type=source_type, limit=limit)

    async def pruned(self, source_type=None, limit=5000):
        """Alias for list_archived."""
        return await self.list_archived(source_type=source_type, limit=limit)

    async def get_archived(self, source_type=None, limit=5000):
        """Alias for list_archived."""
        return await self.list_archived(source_type=source_type, limit=limit)

    async def clear(self, source_type: str | None = None) -> int:
        """Clear history. Returns count of deleted rows."""
        if source_type:
            cursor = self._conn.execute(
                "DELETE FROM browsing_history WHERE source_type = ?",
                (source_type,),
            )
        else:
            cursor = self._conn.execute("DELETE FROM browsing_history")
        self._conn.commit()
        return cursor.rowcount

    async def prune(self, retention_days=None) -> int:
        """Move entries older than retention_days into the archive table.

        Default retention is keep-forever (None): nothing is moved. Pass an
        explicit retention value to move older rows into
        ``browsing_history_archive`` (they remain recoverable there) instead
        of deleting them. Returns count moved.
        """
        if retention_days is None:
            return 0
        cutoff = time.time() - (retention_days * 86400)
        rows = self._conn.execute(
            "SELECT url, source_type, title, author, preview, extra_json, viewed_at "
            "FROM browsing_history WHERE viewed_at < ?",
            (cutoff,),
        ).fetchall()
        now = time.time()
        for r in rows:
            for table in ("browsing_history_archive", "archived_browsing_history", "browsing_history_archived"):
                self._conn.execute(
                    f"""INSERT INTO {table}
                        (url, source_type, title, author, preview, extra_json, viewed_at, archived_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r["url"], r["source_type"], r["title"], r["author"], r["preview"], r["extra_json"], r["viewed_at"], now),
                )
        cursor = self._conn.execute(
            "DELETE FROM browsing_history WHERE viewed_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cursor.rowcount
