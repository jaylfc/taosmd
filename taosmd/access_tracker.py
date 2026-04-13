"""Access Tracker — tracks query diversity and consolidation for memories (taOSmd).

Records which queries accessed which memories, enabling the composite scoring
system to compute query_diversity (how many DIFFERENT queries retrieved a memory)
and consolidation_count (how many processing phases reinforced it).

Separate from the KG/vector stores so it doesn't require schema migrations.
Works across all memory layers — any store can call track_access().
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS access_log (
    memory_key TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    accessed_at REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'search'
);
CREATE INDEX IF NOT EXISTS idx_access_key ON access_log(memory_key);
CREATE INDEX IF NOT EXISTS idx_access_query ON access_log(query_hash);

CREATE TABLE IF NOT EXISTS consolidation_log (
    memory_key TEXT NOT NULL,
    phase TEXT NOT NULL,
    processed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consolidation_key ON consolidation_log(memory_key);
"""


def _query_hash(query: str) -> str:
    """Hash a query string for diversity tracking."""
    normalised = " ".join(query.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:12]


class AccessTracker:
    """Tracks memory access patterns for composite scoring."""

    def __init__(self, db_path: str | Path = "data/access-tracker.db"):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    async def track_access(
        self,
        memory_key: str,
        query: str,
        source: str = "search",
    ) -> None:
        """Record that a query accessed a memory.

        Args:
            memory_key: Unique key for the memory (e.g., "kg:triple_id", "vector:42")
            query: The search query that retrieved this memory
            source: Which retrieval path ("vector", "kg", "archive", "catalog", "crystals")
        """
        self._conn.execute(
            "INSERT INTO access_log (memory_key, query_hash, accessed_at, source) VALUES (?, ?, ?, ?)",
            (memory_key, _query_hash(query), time.time(), source),
        )
        self._conn.commit()

    async def track_consolidation(
        self,
        memory_key: str,
        phase: str,
    ) -> None:
        """Record that a processing phase reinforced a memory.

        Args:
            memory_key: Unique key for the memory
            phase: Which phase ("enrich", "crystallize", "reflect", "split")
        """
        self._conn.execute(
            "INSERT INTO consolidation_log (memory_key, phase, processed_at) VALUES (?, ?, ?)",
            (memory_key, phase, time.time()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Scoring inputs
    # ------------------------------------------------------------------

    async def unique_query_count(self, memory_key: str) -> int:
        """How many different queries have accessed this memory."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT query_hash) as n FROM access_log WHERE memory_key = ?",
            (memory_key,),
        ).fetchone()
        return row["n"]

    async def access_count(self, memory_key: str) -> int:
        """Total access count for a memory."""
        row = self._conn.execute(
            "SELECT COUNT(*) as n FROM access_log WHERE memory_key = ?",
            (memory_key,),
        ).fetchone()
        return row["n"]

    async def consolidation_count(self, memory_key: str) -> int:
        """How many processing phases have reinforced this memory."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT phase) as n FROM consolidation_log WHERE memory_key = ?",
            (memory_key,),
        ).fetchone()
        return row["n"]

    async def scoring_inputs(self, memory_key: str) -> dict:
        """Get all scoring inputs for a memory in one call.

        Returns {unique_queries, total_accesses, consolidation_count, last_accessed_at}.
        """
        unique = await self.unique_query_count(memory_key)
        total = await self.access_count(memory_key)
        consolidation = await self.consolidation_count(memory_key)

        last_row = self._conn.execute(
            "SELECT MAX(accessed_at) as ts FROM access_log WHERE memory_key = ?",
            (memory_key,),
        ).fetchone()
        last_accessed = last_row["ts"] if last_row["ts"] else 0

        return {
            "unique_queries": unique,
            "total_accesses": total,
            "consolidation_count": consolidation,
            "last_accessed_at": last_accessed,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def cleanup(self, older_than_days: int = 90) -> int:
        """Remove old access logs. Keeps consolidation logs permanently."""
        cutoff = time.time() - (older_than_days * 86400)
        cursor = self._conn.execute(
            "DELETE FROM access_log WHERE accessed_at < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    async def stats(self) -> dict:
        accesses = self._conn.execute("SELECT COUNT(*) as n FROM access_log").fetchone()["n"]
        unique_memories = self._conn.execute("SELECT COUNT(DISTINCT memory_key) as n FROM access_log").fetchone()["n"]
        consolidations = self._conn.execute("SELECT COUNT(*) as n FROM consolidation_log").fetchone()["n"]
        return {
            "total_accesses": accesses,
            "unique_memories_tracked": unique_memories,
            "total_consolidations": consolidations,
        }
