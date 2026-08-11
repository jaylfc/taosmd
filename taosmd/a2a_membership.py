"""A2A thread membership store.

Tracks which principals (agents) belong to which threads and their roles.
Zero-loss: removal marks membership inactive; it does NOT delete the row.
Backward compatible: threads with no membership rows are open to all.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from taosmd import _db, migrations

Role = Literal["owner", "member"]


@dataclass(frozen=True)
class Membership:
    """Thread membership record."""
    thread: str
    principal_id: str
    role: Role
    created_at: float
    removed_at: float | None = None


class MembershipStore:
    """Persistent store for A2A thread membership."""

    def __init__(self, data_dir: str | Path = "~/.taosmd") -> None:
        self._data_dir = Path(data_dir)
        self._path = self._data_dir / "a2a-membership.db"
        self._conn: sqlite3.Connection | None = None
        self._conn = _db.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if missing."""
        schema = """
        CREATE TABLE IF NOT EXISTS a2a_membership (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at REAL NOT NULL,
            removed_at REAL,
            UNIQUE(thread, principal_id)
        );
        CREATE INDEX IF NOT EXISTS idx_a2a_membership_thread ON a2a_membership(thread);
        CREATE INDEX IF NOT EXISTS idx_a2a_membership_principal ON a2a_membership(principal_id);
        CREATE INDEX IF NOT EXISTS idx_a2a_membership_active ON a2a_membership(thread, removed_at)
            WHERE removed_at IS NULL;
        """
        self._conn.executescript(schema)

    async def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- CRUD operations ---

    async def add_membership(
        self,
        thread: str,
        principal_id: str,
        role: Role = "member",
        created_at: float | None = None,
    ) -> int:
        """Add a membership record (owner by default for thread creation)."""
        ts = created_at or time.time()
        cursor = self._conn.execute(
            """
            INSERT INTO a2a_membership (thread, principal_id, role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (thread, principal_id, role, ts),
        )
        return cursor.lastrowid

    async def remove_membership(
        self,
        thread: str,
        principal_id: str,
        removed_at: float | None = None,
    ) -> bool:
        """Mark a membership inactive (zero-loss)."""
        ts = removed_at or time.time()
        cursor = self._conn.execute(
            """
            UPDATE a2a_membership
            SET removed_at = ?
            WHERE thread = ? AND principal_id = ? AND removed_at IS NULL
            """,
            (ts, thread, principal_id),
        )
        return cursor.rowcount > 0

    async def get_membership(
        self,
        thread: str,
        principal_id: str,
    ) -> Membership | None:
        """Get current (active) membership for a principal in a thread."""
        row = self._conn.execute(
            """
            SELECT thread, principal_id, role, created_at, removed_at
            FROM a2a_membership
            WHERE thread = ? AND principal_id = ? AND removed_at IS NULL
            """,
            (thread, principal_id),
        ).fetchone()
        if not row:
            return None
        return Membership(
            thread=row["thread"],
            principal_id=row["principal_id"],
            role=row["role"],
            created_at=row["created_at"],
            removed_at=row["removed_at"],
        )

    async def list_active_members(self, thread: str) -> list[Membership]:
        """List all active members (owners and members) of a thread."""
        rows = self._conn.execute(
            """
            SELECT thread, principal_id, role, created_at, removed_at
            FROM a2a_membership
            WHERE thread = ? AND removed_at IS NULL
            ORDER BY role DESC, principal_id
            """,
            (thread,),
        ).fetchall()
        return [
            Membership(
                thread=row["thread"],
                principal_id=row["principal_id"],
                role=row["role"],
                created_at=row["created_at"],
                removed_at=row["removed_at"],
            )
            for row in rows
        ]

    async def get_thread_owners(self, thread: str) -> list[Membership]:
        """Get all active owners of a thread."""
        rows = self._conn.execute(
            """
            SELECT thread, principal_id, role, created_at, removed_at
            FROM a2a_membership
            WHERE thread = ? AND role = 'owner' AND removed_at IS NULL
            """,
            (thread,),
        ).fetchall()
        return [
            Membership(
                thread=row["thread"],
                principal_id=row["principal_id"],
                role=row["role"],
                created_at=row["created_at"],
                removed_at=row["removed_at"],
            )
            for row in rows
        ]

    async def is_principal_owner(self, thread: str, principal_id: str) -> bool:
        """Check if a principal is an active owner of a thread."""
        row = self._conn.execute(
            """
            SELECT 1
            FROM a2a_membership
            WHERE thread = ? AND principal_id = ? AND role = 'owner' AND removed_at IS NULL
            LIMIT 1
            """,
            (thread, principal_id),
        ).fetchone()
        return row is not None

    async def is_principal_member(self, thread: str, principal_id: str) -> bool:
        """Check if a principal is an active member of a thread."""
        row = self._conn.execute(
            """
            SELECT 1
            FROM a2a_membership
            WHERE thread = ? AND principal_id = ? AND removed_at IS NULL
            LIMIT 1
            """,
            (thread, principal_id),
        ).fetchone()
        return row is not None

    async def has_ownership(self, thread: str) -> bool:
        """Check if a thread has at least one owner."""
        row = self._conn.execute(
            """
            SELECT 1
            FROM a2a_membership
            WHERE thread = ? AND role = 'owner' AND removed_at IS NULL
            LIMIT 1
            """,
            (thread,),
        ).fetchone()
        return row is not None

    async def count_active_members(self, thread: str) -> int:
        """Count active members (owners + members) of a thread."""
        row = self._conn.execute(
            """
            SELECT COUNT(*) as n
            FROM a2a_membership
            WHERE thread = ? AND removed_at IS NULL
            """,
            (thread,),
        ).fetchone()
        return row["n"] if row else 0

    async def has_any_membership(self, thread: str) -> bool:
        """Check if a thread has any membership rows at all (active or inactive)."""
        row = self._conn.execute(
            """
            SELECT 1
            FROM a2a_membership
            WHERE thread = ?
            LIMIT 1
            """,
            (thread,),
        ).fetchone()
        return row is not None

    # --- Archive integration ---

    async def archive_membership_created(
        self,
        thread: str,
        principal_id: str,
        role: Role,
        ts: float,
        data_dir: str | Path,
    ) -> None:
        """Record a membership creation as an archive event."""
        from taosmd.archive import ArchiveStore, EVENT_A2A

        archive = ArchiveStore(data_dir=str(data_dir))
        await archive.init()
        await archive.record(
            event_type=EVENT_A2A,
            data={
                "admin_action": "membership_created",
                "thread": thread,
                "principal_id": principal_id,
                "role": role,
                "timestamp": ts,
            },
            agent_name=principal_id,
            app_id=thread,
            summary=f"Added {principal_id} as {role} to thread {thread}",
        )
        await archive.close()

    async def archive_membership_removed(
        self,
        thread: str,
        principal_id: str,
        ts: float,
        data_dir: str | Path,
    ) -> None:
        """Record a membership removal as an archive event."""
        from taosmd.archive import ArchiveStore, EVENT_A2A

        archive = ArchiveStore(data_dir=str(data_dir))
        await archive.init()
        await archive.record(
            event_type=EVENT_A2A,
            data={
                "admin_action": "membership_removed",
                "thread": thread,
                "principal_id": principal_id,
                "timestamp": ts,
            },
            agent_name=principal_id,
            app_id=thread,
            summary=f"Removed {principal_id} from thread {thread}",
        )
        await archive.close()