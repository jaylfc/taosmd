"""A2A read receipts store.

Receipts are keyed by (message_id, agent_id) and track:

* ``delivered_at`` -- epoch float, set once on first delivery, never moves.
* ``seen_at`` -- epoch float, nullable, moves from null to a value exactly once.

Design rules (from taOS 1472):

* ``INSERT OR IGNORE`` is used for the delivered mark: a watcher redelivering
  the same message to the same agent must NOT duplicate or move ``delivered_at``
  earlier. After a prune removes the row, a subsequent delivery inserts a fresh
  ``delivered_at`` because the row no longer exists.
* ``seen_at`` is guarded by a ``WHERE seen_at IS NULL`` update: it only ever
  moves from null to a value and is never cleared or moved back.
* Raw-bus subscribers that do not present an identifiable agent produce no
  delivered mark.
"""

from __future__ import annotations

import logging
import sqlite3

from taosmd import _db

__all__ = ["SCHEMA", "ReceiptStore"]

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS a2a_receipts (
    message_id INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    delivered_at REAL NOT NULL,
    seen_at REAL,
    PRIMARY KEY (message_id, agent_id)
);
CREATE INDEX IF NOT EXISTS idx_receipts_message ON a2a_receipts(message_id);
CREATE INDEX IF NOT EXISTS idx_receipts_agent ON a2a_receipts(agent_id);
CREATE INDEX IF NOT EXISTS idx_receipts_delivered ON a2a_receipts(delivered_at);
"""


class ReceiptStore:
    """Store for A2A read receipts.

    All methods are async so callers can ``await`` them uniformly; the body
    runs synchronously against a single SQLite connection. The connection is
    opened through ``taosmd._db.connect`` (WAL journal mode, 5000 ms busy
    timeout) with ``check_same_thread=False`` so the connection stays usable
    from whichever thread drives the event loop. This differs from the
    thread-affine stores: ReceiptStore's async methods are not guaranteed to
    run on the creating thread, so ``check_same_thread=False`` is required to
    avoid a thread-affinity crash, and routing through ``_db.connect`` is
    required to get WAL and the busy timeout.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        self._conn = _db.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def record_delivered(
        self, message_id: int, agent_id: str, ts: float
    ) -> None:
        """Record that a message was delivered to an agent.

        Idempotent: when a row already exists for ``(message_id, agent_id)``
        the existing ``delivered_at`` is left unchanged.  After a prune
        removes the row a subsequent call inserts a fresh ``delivered_at``
        because ``INSERT OR IGNORE`` fires only when the row is absent.
        """
        if self._conn is None:
            raise RuntimeError("ReceiptStore not initialised")
        self._conn.execute(
            "INSERT OR IGNORE INTO a2a_receipts (message_id, agent_id, delivered_at) "
            "VALUES (?, ?, ?)",
            (message_id, agent_id, ts),
        )
        self._conn.commit()

    async def record_seen(
        self, message_id: int, agent_id: str, ts: float
    ) -> None:
        """Record that an agent has seen a message.

        * When no row exists for ``(message_id, agent_id)`` a new row is
          created with both ``delivered_at`` and ``seen_at`` set to ``ts``.
        * When ``seen_at`` is already set it is left unchanged (monotonic).
        """
        if self._conn is None:
            raise RuntimeError("ReceiptStore not initialised")
        self._conn.execute(
            "INSERT OR IGNORE INTO a2a_receipts "
            "(message_id, agent_id, delivered_at, seen_at) "
            "VALUES (?, ?, ?, ?)",
            (message_id, agent_id, ts, ts),
        )
        self._conn.execute(
            "UPDATE a2a_receipts SET seen_at = ? "
            "WHERE message_id = ? AND agent_id = ? AND seen_at IS NULL",
            (ts, message_id, agent_id),
        )
        self._conn.commit()

    async def get_receipts_for_message(self, message_id: int) -> dict:
        """Return ``{"delivered": [...], "read": [...]}`` for ``message_id``."""
        if self._conn is None:
            raise RuntimeError("ReceiptStore not initialised")
        rows = self._conn.execute(
            "SELECT agent_id, delivered_at, seen_at FROM a2a_receipts "
            "WHERE message_id = ?",
            (message_id,),
        ).fetchall()
        delivered = []
        read = []
        for row in rows:
            delivered.append(
                {"agent_id": row["agent_id"], "delivered_at": row["delivered_at"]}
            )
            if row["seen_at"] is not None:
                read.append(
                    {"agent_id": row["agent_id"], "seen_at": row["seen_at"]}
                )
        return {"delivered": delivered, "read": read}

    async def get_receipt(
        self, message_id: int, agent_id: str
    ) -> dict | None:
        """Return ``{"delivered_at", "seen_at"}`` for one receipt, or ``None``."""
        if self._conn is None:
            raise RuntimeError("ReceiptStore not initialised")
        row = self._conn.execute(
            "SELECT delivered_at, seen_at FROM a2a_receipts "
            "WHERE message_id = ? AND agent_id = ?",
            (message_id, agent_id),
        ).fetchone()
        if row is None:
            return None
        return {"delivered_at": row["delivered_at"], "seen_at": row["seen_at"]}

    async def prune(self, older_than_ts: float) -> int:
        """Delete receipts whose ``delivered_at`` predates ``older_than_ts``.

        Returns the number of rows removed.
        """
        if self._conn is None:
            raise RuntimeError("ReceiptStore not initialised")
        cur = self._conn.execute(
            "DELETE FROM a2a_receipts WHERE delivered_at < ?",
            (older_than_ts,),
        )
        self._conn.commit()
        return cur.rowcount
