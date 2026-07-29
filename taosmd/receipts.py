"""Receipt store module for A2A bus message delivery receipts.

Stores delivered and seen marks for A2A messages per (message_id, agent_id).
Key: (message_id, agent_id) -> delivered_at (nullable), seen_at (nullable)
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Receipt:
    """A2A message receipt record."""
    message_id: int
    agent_id: str
    delivered_at: Optional[float] = None
    seen_at: Optional[float] = None

    @classmethod
    def from_db_row(cls, row: tuple) -> "Receipt":
        """Create a Receipt from a database row."""
        return cls(
            message_id=row[0],
            agent_id=row[1],
            delivered_at=row[2],
            seen_at=row[3],
        )

    def to_db_values(self) -> tuple:
        """Convert to database values tuple."""
        return (self.message_id, self.agent_id, self.delivered_at, self.seen_at)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this receipt has expired (TTL-based pruning)."""
        if now is None:
            now = time.time()
        if self.delivered_at is None:
            return False
        # Default TTL of 30 days
        ttl = 30 * 24 * 60 * 60
        return now - self.delivered_at > ttl

    def has_been_delivered(self) -> bool:
        """Check if the message has been delivered to this agent."""
        return self.delivered_at is not None

    def has_been_seen(self) -> bool:
        """Check if the message has been seen by this agent."""
        return self.seen_at is not None

    def is_delivered_to_agent(self, agent_id: str) -> bool:
        """Check if the message has been delivered to the specified agent."""
        return self.agent_id == agent_id and self.delivered_at is not None

    def is_seen_by_agent(self, agent_id: str) -> bool:
        """Check if the message has been seen by the specified agent."""
        return self.agent_id == agent_id and self.seen_at is not None


_RECEIPTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    message_id INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    delivered_at REAL,
    seen_at REAL,
    PRIMARY KEY (message_id, agent_id)
)
"""


class ReceiptStore:
    """Store for A2A message delivery receipts."""

    def __init__(
        self,
        data_dir: str | Path = "data",
    ):
        self._data_dir = Path(data_dir)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            path = self._data_dir / "receipts.db"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS receipts (
                message_id INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                delivered_at REAL,
                seen_at REAL,
                PRIMARY KEY (message_id, agent_id)
            )
            """
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def record_delivered(
        self,
        message_id: int,
        agent_id: str,
        delivered_at: Optional[float] = None,
        upsert: bool = True,
    ) -> bool:
        """Record that a message was delivered to an agent.

        Args:
            message_id: The ID of the message.
            agent_id: The ID of the agent that received the message.
            delivered_at: When the message was delivered (defaults to now).
            upsert: If True, update existing record. If False, skip if exists.

        Returns:
            True if a new row was inserted or updated, False if skip-if-exists.
        """
        if delivered_at is None:
            delivered_at = time.time()

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT delivered_at FROM receipts WHERE message_id = ? AND agent_id = ?",
            (message_id, agent_id),
        )
        exists = cursor.fetchone() is not None

        if exists and not upsert:
            return False

        conn.execute(
            """
            INSERT INTO receipts (message_id, agent_id, delivered_at, seen_at)
            VALUES (?, ?, ?, NULL)
            ON CONFLICT(message_id, agent_id) DO UPDATE SET
                delivered_at = excluded.delivered_at,
                seen_at = receipts.seen_at
            """,
            (message_id, agent_id, delivered_at),
        )
        conn.commit()
        return True

    def record_seen(
        self,
        message_id: int,
        agent_id: str,
        seen_at: Optional[float] = None,
    ) -> bool:
        """Record that an agent has seen a message.

        Args:
            message_id: The ID of the message.
            agent_id: The ID of the agent that saw the message.
            seen_at: When the message was seen (defaults to now).

        Returns:
            True if the seen_at was updated, False if the record doesn't exist.
        """
        if seen_at is None:
            seen_at = time.time()

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT seen_at FROM receipts WHERE message_id = ? AND agent_id = ?",
            (message_id, agent_id),
        )
        row = cursor.fetchone()
        if row is None:
            return False

        existing_seen = row["seen_at"]
        # Only update if moving from null to a value
        if existing_seen is not None and existing_seen >= seen_at:
            return False

        conn.execute(
            "UPDATE receipts SET seen_at = ? WHERE message_id = ? AND agent_id = ?",
            (seen_at, message_id, agent_id),
        )
        conn.commit()
        return True

    def get_receipt(
        self,
        message_id: int,
        agent_id: str,
    ) -> Optional[Receipt]:
        """Get a receipt for a specific message and agent."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT message_id, agent_id, delivered_at, seen_at "
            "FROM receipts WHERE message_id = ? AND agent_id = ?",
            (message_id, agent_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return Receipt.from_db_row(row)

    def get_delivered_by_message(self, message_id: int) -> List[Receipt]:
        """Get all receipts for a message, including delivered info."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT message_id, agent_id, delivered_at, seen_at "
            "FROM receipts WHERE message_id = ?",
            (message_id,),
        )
        rows = cursor.fetchall()
        return [Receipt.from_db_row(row) for row in rows]

    def get_read_by_message(self, message_id: int) -> List[Receipt]:
        """Get all receipts where the message has been seen, including seen info."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT message_id, agent_id, delivered_at, seen_at "
            "FROM receipts WHERE message_id = ? AND seen_at IS NOT NULL",
            (message_id,),
        )
        rows = cursor.fetchall()
        return [Receipt.from_db_row(row) for row in rows]

    def prune_expired(self, ttl_days: int = 30) -> int:
        """Remove expired receipts.

        Args:
            ttl_days: Time-to-live in days for receipts.

        Returns:
            Number of rows pruned.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT message_id, agent_id, delivered_at FROM receipts WHERE delivered_at IS NOT NULL"
        )
        rows = cursor.fetchall()

        now = time.time()
        ttl_seconds = ttl_days * 24 * 60 * 60

        expired_message_ids = []
        for row in rows:
            if now - row["delivered_at"] > ttl_seconds:
                expired_message_ids.append(row["message_id"])

        if not expired_message_ids:
            return 0

        placeholders = ",".join("?" for _ in expired_message_ids)
        cursor = conn.execute(
            f"DELETE FROM receipts WHERE message_id IN ({placeholders})",
            expired_message_ids,
        )
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count

    def get_all_by_agent(self, agent_id: str) -> List[Receipt]:
        """Get all receipts for a specific agent."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT message_id, agent_id, delivered_at, seen_at "
            "FROM receipts WHERE agent_id = ?",
            (agent_id,),
        )
        rows = cursor.fetchall()
        return [Receipt.from_db_row(row) for row in rows]
