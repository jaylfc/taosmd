"""Admin surface: shelf lifecycle and A2A channel admin operations.

This module provides the service-layer logic and sidecar state for:

- Shelf lifecycle (POST /shelves, POST /shelves/{id}/archive,
  POST /shelves/{id}/unarchive) per the taOS#774 contract.
- A2A channel admin (delete-channel, rename-channel, supersede-message)
  that hide or redirect content without mutating the zero-loss archive.

Shelf lifecycle
---------------
A shelf IS an agent registration (per agents.py). Creating a shelf calls
ensure_agent with optional project_id and display_name stored in the agent
record's metadata field. Archiving a shelf soft-hides its vector rows using
the same valid_to supersede machinery as the correction path, but tags each
hidden row's metadata with ``hidden_by: "shelf-archive:<ts>"`` so that
unarchive can restore exactly those rows and nothing else.

A2A admin sidecar
-----------------
The deleted-channels set, channel-alias map, and superseded-message set are
persisted in a small JSON sidecar under the data dir at
``data/a2a-admin-state.json``. Writes use atomic tmp+os.replace exactly like
agents.py and config.py.

At query time (in service.py wrappers) callers load this sidecar and filter:
- a2a_channels() skips channels in deleted_channels
- a2a_feed() skips channels in deleted_channels, resolves aliases, skips
  message ids in superseded_messages
- a2a_send() redirects sends to renamed channels via the alias map
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from .agents import AgentRegistry, InvalidAgentNameError, NAME_RE
from .archive import EVENT_A2A

logger = logging.getLogger(__name__)

# Marker prefix embedded in metadata so unarchive only un-hides rows
# that were hidden by this specific shelf-archive operation.
_SHELF_ARCHIVE_MARKER = "shelf-archive"


# ---------------------------------------------------------------------------
# A2A admin sidecar
# ---------------------------------------------------------------------------

class A2AAdminState:
    """Manages the persisted sidecar for A2A admin operations.

    State shape::

        {
            "deleted_channels": ["chan1", "chan2"],
            "channel_aliases": {"old_name": "new_name"},
            "superseded_messages": [42, 99]
        }

    All writes are atomic (tmp + os.replace). The sidecar is read fresh on
    each operation so multiple processes / threads see a consistent view.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._path = Path(data_dir) / "a2a-admin-state.json"

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"deleted_channels": [], "channel_aliases": {}, "superseded_messages": []}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raw = {}
        return {
            "deleted_channels": list(raw.get("deleted_channels") or []),
            "channel_aliases": dict(raw.get("channel_aliases") or {}),
            "superseded_messages": list(raw.get("superseded_messages") or []),
        }

    def _write(self, state: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    # ----- reads ----------------------------------------------------------

    def deleted_channels(self) -> set[str]:
        return set(self._read()["deleted_channels"])

    def channel_aliases(self) -> dict[str, str]:
        return self._read()["channel_aliases"]

    def superseded_messages(self) -> set[int]:
        return set(int(x) for x in self._read()["superseded_messages"])

    def resolve_channel(self, channel: str) -> str:
        """Return the canonical channel name after following any alias chain."""
        aliases = self.channel_aliases()
        visited: set[str] = set()
        while channel in aliases and channel not in visited:
            visited.add(channel)
            channel = aliases[channel]
        return channel

    # ----- writes ----------------------------------------------------------

    def delete_channel(self, channel: str) -> None:
        state = self._read()
        if channel not in state["deleted_channels"]:
            state["deleted_channels"].append(channel)
            self._write(state)

    def add_alias(self, from_channel: str, to_channel: str) -> None:
        """Map from_channel -> to_channel. Overwrites an existing mapping."""
        state = self._read()
        state["channel_aliases"][from_channel] = to_channel
        self._write(state)

    def supersede_message(self, msg_id: int) -> None:
        state = self._read()
        if msg_id not in state["superseded_messages"]:
            state["superseded_messages"].append(msg_id)
            self._write(state)


# ---------------------------------------------------------------------------
# Shelf operations
# ---------------------------------------------------------------------------

async def shelf_create(
    shelf_id: str,
    *,
    project_id: str | None = None,
    display_name: str | None = None,
    data_dir: Path | str,
) -> dict:
    """Create or return an existing shelf (agent registration).

    Returns ``{"shelf": {...}, "created": bool}``. shelf_id must match
    ``^[a-z][a-z0-9_-]{0,62}$``; raises ``InvalidAgentNameError`` (400)
    otherwise. Idempotent: if the shelf already exists (and is not archived),
    returns it with ``created=False``.
    """
    if not NAME_RE.match(shelf_id):
        raise InvalidAgentNameError(
            f"shelf_id must match ^[a-z][a-z0-9_-]{{0,62}}$ (got {shelf_id!r})"
        )

    registry = AgentRegistry(data_dir)
    data = registry._read()

    existing = next((a for a in data["agents"] if a["name"] == shelf_id), None)
    if existing is not None:
        # Return existing shelf (even if archived -- caller can unarchive separately)
        rec = dict(existing)
        return {"shelf": rec, "created": False}

    # Create new registration. Store project_id and display_name in the record.
    from .agents import AgentRecord, _default_librarian  # noqa: PLC0415
    record = AgentRecord(
        name=shelf_id,
        display_name=display_name or shelf_id,
        created_at=int(time.time()),
        librarian=_default_librarian(),
    )
    rec_dict = record.to_dict()
    # Store extra metadata on the record
    if project_id:
        rec_dict["project_id"] = project_id
    if display_name:
        rec_dict["display_name"] = display_name

    data["agents"].append(rec_dict)
    registry._write(data)
    registry._agent_dir(shelf_id).mkdir(parents=True, exist_ok=True)

    return {"shelf": rec_dict, "created": True}


async def shelf_archive(
    shelf_id: str,
    *,
    expect_empty: bool = False,
    data_dir: Path | str,
    stores: dict,
) -> dict:
    """Archive a shelf: mark it archived and soft-hide its vector rows.

    If ``expect_empty=True`` and the shelf has active vector rows, returns
    a ``409 Conflict`` signal by raising ``ShelfNotEmptyError``. Archiving
    an already-archived shelf is a no-op (returns ``rows_hidden=0``).

    Archive events are recorded to the zero-loss archive (event_type "shelf").
    """
    registry = AgentRegistry(data_dir)
    data = registry._read()
    agent_rec = next((a for a in data["agents"] if a["name"] == shelf_id), None)
    if agent_rec is None:
        raise ShelfNotFoundError(f"shelf {shelf_id!r} not found")

    # Already archived: no-op
    meta = agent_rec.get("metadata") or {}
    if meta.get("archived_at"):
        return {"archived": True, "rows_hidden": 0}

    vmem = stores["vector"]
    archive = stores["archive"]

    # Count active rows for this shelf
    rows = _get_active_shelf_rows(vmem, shelf_id)

    if expect_empty and rows:
        raise ShelfNotEmptyError(
            f"shelf {shelf_id!r} has {len(rows)} active row(s); "
            "use expect_empty=false to archive anyway"
        )

    ts = time.time()
    marker = f"{_SHELF_ARCHIVE_MARKER}:{ts}"

    # Soft-hide the rows by patching their metadata to include the marker
    # and stamping valid_to. We need to update metadata_json before setting
    # valid_to so the marker is visible for unarchive.
    hidden = 0
    for row in rows:
        row_id = row["id"]
        row_meta = row["metadata"]
        row_meta["hidden_by"] = marker
        vmem._conn.execute(
            "UPDATE vector_memory SET valid_to = ?, metadata_json = ? "
            "WHERE id = ? AND valid_to IS NULL",
            (ts, json.dumps(row_meta), row_id),
        )
        hidden += 1
    if hidden:
        vmem._conn.commit()
        vmem._bm25_dirty = True

    # Mark the agent record archived
    if "metadata" not in agent_rec or not isinstance(agent_rec["metadata"], dict):
        agent_rec["metadata"] = {}
    agent_rec["metadata"]["archived_at"] = ts
    registry._write(data)

    # Record archive event
    await archive.record(
        event_type="shelf",
        data={"action": "archived", "shelf_id": shelf_id, "rows_hidden": hidden},
        agent_name=shelf_id,
        summary=f"shelf {shelf_id} archived; {hidden} row(s) hidden",
    )

    return {"archived": True, "rows_hidden": hidden}


async def shelf_unarchive(
    shelf_id: str,
    *,
    data_dir: Path | str,
    stores: dict,
) -> dict:
    """Unarchive a shelf: clear archived_at and restore only shelf-archive-hidden rows.

    Rows superseded for OTHER reasons (corrections, manual supersede) are NOT
    restored. Only rows whose ``hidden_by`` metadata marker starts with
    ``"shelf-archive:"`` are restored.
    """
    registry = AgentRegistry(data_dir)
    data = registry._read()
    agent_rec = next((a for a in data["agents"] if a["name"] == shelf_id), None)
    if agent_rec is None:
        raise ShelfNotFoundError(f"shelf {shelf_id!r} not found")

    vmem = stores["vector"]
    archive = stores["archive"]

    # Find rows hidden by this shelf's archive operation(s) and restore them
    rows_restored = 0
    prefix = f"{_SHELF_ARCHIVE_MARKER}:"
    hidden_rows = vmem._conn.execute(
        "SELECT id, metadata_json FROM vector_memory WHERE valid_to IS NOT NULL"
    ).fetchall()

    for row in hidden_rows:
        try:
            meta = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            meta = {}
        hidden_by = meta.get("hidden_by", "")
        if not isinstance(hidden_by, str):
            continue
        # Check the marker belongs to this shelf
        # Format: "shelf-archive:<ts>" -- we only restore rows hidden by *this* shelf's
        # archive operations. We check that they match the agent's known archived_at
        # or any prior archived_at for this shelf.
        # Simpler: match rows where hidden_by starts with "shelf-archive:" AND the
        # row's agent metadata matches shelf_id.
        if not hidden_by.startswith(prefix):
            continue
        # Verify agent association
        row_agent = meta.get("agent")
        if row_agent is not None and row_agent != shelf_id:
            continue

        # Clear the hidden_by marker and restore the row
        del meta["hidden_by"]
        vmem._conn.execute(
            "UPDATE vector_memory SET valid_to = NULL, metadata_json = ? WHERE id = ?",
            (json.dumps(meta), row["id"]),
        )
        rows_restored += 1

    if rows_restored:
        vmem._conn.commit()
        vmem._bm25_dirty = True

    # Clear archived_at on the agent record
    if "metadata" in agent_rec and isinstance(agent_rec["metadata"], dict):
        agent_rec["metadata"].pop("archived_at", None)
    registry._write(data)

    # Record archive event
    await archive.record(
        event_type="shelf",
        data={"action": "unarchived", "shelf_id": shelf_id, "rows_restored": rows_restored},
        agent_name=shelf_id,
        summary=f"shelf {shelf_id} unarchived; {rows_restored} row(s) restored",
    )

    return {"archived": False, "rows_restored": rows_restored}


# ---------------------------------------------------------------------------
# A2A admin operations
# ---------------------------------------------------------------------------

async def a2a_admin_delete_channel(
    channel: str,
    *,
    data_dir: Path | str,
    stores: dict,
) -> dict:
    """Soft-delete a channel: record an archive event and hide it from feeds.

    Messages stay in the archive (zero-loss). The deleted-channels set in the
    sidecar is consulted at query time by a2a_channels and a2a_feed.
    """
    state = A2AAdminState(data_dir)
    archive = stores["archive"]

    state.delete_channel(channel)

    await archive.record(
        event_type=EVENT_A2A,
        data={"admin_action": "delete_channel", "channel": channel},
        app_id=channel,
        summary=f"A2A admin: channel {channel!r} deleted",
    )

    return {"deleted": True, "channel": channel}


async def a2a_admin_rename_channel(
    from_channel: str,
    to_channel: str,
    *,
    data_dir: Path | str,
    stores: dict,
) -> dict:
    """Rename a channel by adding an alias: old -> new.

    New sends to the old name are redirected to the new name. Reads of the
    new name include the old name's history (the alias map is consulted at
    query time). Stored rows are NOT mutated. Renaming again re-points the
    alias.
    """
    if not from_channel or not to_channel:
        raise ValueError("both 'from' and 'to' channel names are required")
    if from_channel == to_channel:
        raise ValueError("'from' and 'to' channel names must differ")

    state = A2AAdminState(data_dir)
    archive = stores["archive"]

    state.add_alias(from_channel, to_channel)

    await archive.record(
        event_type=EVENT_A2A,
        data={"admin_action": "rename_channel", "from": from_channel, "to": to_channel},
        app_id=to_channel,
        summary=f"A2A admin: channel {from_channel!r} renamed to {to_channel!r}",
    )

    return {"renamed": True, "from": from_channel, "to": to_channel}


async def a2a_admin_supersede_message(
    msg_id: int,
    *,
    data_dir: Path | str,
    stores: dict,
) -> dict:
    """Hide one message from feed responses. Archive row is untouched.

    The message id is added to the superseded-messages set in the sidecar;
    a2a_feed and a2a_messages filter it out at query time.
    """
    state = A2AAdminState(data_dir)
    state.supersede_message(msg_id)
    return {"superseded": True, "id": msg_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_active_shelf_rows(vmem, shelf_id: str) -> list[dict]:
    """Return active (valid_to IS NULL) rows whose metadata agent == shelf_id."""
    rows = vmem._conn.execute(
        "SELECT id, text, metadata_json FROM vector_memory WHERE valid_to IS NULL"
    ).fetchall()
    result = []
    for row in rows:
        try:
            meta = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            meta = {}
        if meta.get("agent") == shelf_id:
            result.append({"id": row["id"], "text": row["text"], "metadata": meta})
    return result


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ShelfNotFoundError(KeyError):
    """Raised when referencing a shelf that does not exist."""


class ShelfNotEmptyError(ValueError):
    """Raised when archiving with expect_empty=True and the shelf has rows."""
