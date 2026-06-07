"""Adapter-agnostic service layer over :mod:`taosmd.api`.

This is the shared core that activation surfaces sit on top of: the local
HTTP/REST server (#85) and the upcoming MCP server (#84) both call these
functions rather than reaching into :mod:`taosmd.api` directly. Keeping the
glue in one place means a single, consistent contract for every adapter and
guarantees their behaviour matches the Python API exactly.

The functions here are deliberately thin — they reuse
:func:`taosmd.api.ingest`, :func:`taosmd.api.search`,
:func:`taosmd.api.list_pending_decisions`, and
:func:`taosmd.api.resolve_pending_decision` (and therefore
``_ensure_stores`` / the stores cache / ``TAOSMD_DATA_DIR`` handling) so
the only thing they add is a uniform, transport-friendly signature:
``(positional, agent=..., data_dir=..., **opts)``.
"""

from __future__ import annotations

import json

from . import api as _api
from .archive import EVENT_A2A


async def ingest(text, *, agent: str, data_dir=None, **opts) -> dict:
    """Shelve a transcript and embed it for later search.

    Thin wrapper over :func:`taosmd.api.ingest`. ``text`` may be a string,
    a turn dict, or an iterable of either (see the underlying API for the
    accepted shapes). Returns ``{"archived", "agent", "data_dir"}``.
    """
    return await _api.ingest(text, agent=agent, data_dir=data_dir, **opts)


async def search(query: str, *, agent: str, data_dir=None, limit: int = 5, **opts) -> list[dict]:
    """Search memory for passages relevant to ``query``.

    Thin wrapper over :func:`taosmd.api.search`. Returns ranked hits in the
    agent-rules contract shape (``text``/``source``/``timestamp``/
    ``confidence``/``metadata``).
    """
    return await _api.search(query, agent=agent, data_dir=data_dir, limit=limit, **opts)


async def pending_list(*, agent: str | None = None, data_dir=None, limit: int = 20) -> list[dict]:
    """Return unresolved KG-update decisions deferred by the librarian.

    ``agent`` is accepted for adapter symmetry; the pending-decisions queue
    is keyed per data dir (per install), not per agent, so it is not used to
    filter here. Use ``subject=`` on the underlying API if subject-level
    filtering is needed.
    """
    return await _api.list_pending_decisions(limit=limit, data_dir=data_dir)


async def pending_resolve(
    decision_id: str,
    decision: str,
    *,
    note: str = "",
    data_dir=None,
) -> dict:
    """Resolve a pending decision with the user's explicit choice.

    ``decision`` is one of ``accept`` / ``reject`` / ``modify`` (the
    ``action`` argument of :func:`taosmd.api.resolve_pending_decision`).
    Returns ``{ok, applied_kg, resolution}``.
    """
    return await _api.resolve_pending_decision(
        decision_id, action=decision, note=note, data_dir=data_dir,
    )


async def supersede(match: str, *, agent: str | None = None, data_dir=None) -> dict:
    """Soft-supersede vector chunk(s) whose stored text contains ``match``.

    Thin wrapper over :func:`taosmd.api.supersede_vectors`. Used to wire a
    correction into the vector layer by content: matching chunks leave active
    recall while the raw rows + archive entries are retained (zero-loss),
    mirroring how the typed KG invalidates a corrected triple. ``agent`` is
    accepted for adapter symmetry; the vector store is keyed per data dir.
    Returns ``{"superseded": int, "match": str}``.
    """
    count = await _api.supersede_vectors(match, data_dir=data_dir)
    return {"superseded": count, "match": match}


async def stats(*, agent: str, data_dir=None) -> dict:
    """Return lightweight stats for an agent.

    Ensures the stores exist (so a freshly-pointed data dir is initialised),
    then reports the registry record for ``agent`` plus the resolved data
    dir. Shape: ``{"agent", "data_dir", "registered", "created_at",
    "last_ingest_at", "total_chunks"}``. Unknown agents report
    ``registered=False`` with zeroed counters rather than raising, so the
    surface stays forgiving for read-only probes.
    """
    if not agent:
        raise ValueError("agent name is required")
    stores = await _api._ensure_stores(data_dir)

    from .agents import AgentNotFoundError, get_agent  # noqa: PLC0415

    out = {
        "agent": agent,
        "data_dir": stores["data_dir"],
        "registered": False,
        "created_at": 0,
        "last_ingest_at": 0,
        "total_chunks": 0,
    }
    try:
        record = get_agent(agent)
    except AgentNotFoundError:
        return out
    out.update(
        registered=True,
        created_at=record.get("created_at", 0),
        last_ingest_at=record.get("last_ingest_at", 0),
        total_chunks=record.get("total_chunks", 0),
    )
    return out


async def a2a_send(
    sender: str,
    body: str,
    *,
    thread: str = "general",
    reply_to: str | None = None,
    data_dir=None,
) -> dict:
    """Post a message onto the agent-to-agent bus.

    Stores the message as an append-only archive event of type
    :data:`~taosmd.archive.EVENT_A2A` and returns a receipt with the
    assigned row ID. ``sender`` and ``body`` must be non-empty strings;
    ``thread`` defaults to ``"general"``; ``reply_to`` is optional and
    should be the string ID of the message being replied to.

    Returns ``{"id", "from", "thread", "reply_to"}``.
    """
    if not isinstance(sender, str) or not sender:
        raise ValueError("sender must be a non-empty string")
    if not isinstance(body, str) or not body:
        raise ValueError("body must be a non-empty string")
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    row_id = await archive.record(
        event_type=EVENT_A2A,
        data={"from": sender, "body": body, "thread": thread, "reply_to": reply_to},
        agent_name=sender,
        app_id=thread,
        summary=body[:200],
    )
    return {"id": row_id, "from": sender, "thread": thread, "reply_to": reply_to}


async def a2a_feed(
    *,
    thread: str | None = None,
    since: float | None = None,
    limit: int = 50,
    data_dir=None,
) -> list[dict]:
    """Return messages from the agent-to-agent bus, oldest-first.

    Filters by ``thread`` (when given) and by ``since`` (Unix timestamp,
    exclusive lower bound). ``limit`` caps the number of rows fetched from
    the archive (applied before reversing, so it limits the most-recent N
    messages when ``since`` is None). Returns chronological order (oldest
    first) suitable for chat-style display.

    Each item has shape ``{"id", "ts", "from", "body", "thread",
    "reply_to"}``.
    """
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    rows = await archive.query(
        event_type=EVENT_A2A,
        app_id=thread,
        since=since,
        limit=limit,
    )
    # archive.query returns newest-first; A2A feed is displayed oldest-first.
    rows = list(reversed(rows))
    result = []
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        result.append({
            "id": row["id"],
            "ts": row["timestamp"],
            "from": data.get("from"),
            "body": data.get("body"),
            "thread": data.get("thread"),
            "reply_to": data.get("reply_to"),
        })
    return result


async def a2a_channels(*, data_dir=None) -> list[dict]:
    """Return a summary of every channel (named thread) on the A2A bus.

    Derived entirely from existing :data:`~taosmd.archive.EVENT_A2A` archive
    events — no additional schema. Groups by ``app_id`` (which equals the
    thread name) and aggregates membership, message count, and timestamps.

    Each item has shape ``{"channel", "members", "message_count",
    "created_ts", "last_ts"}``, sorted by ``last_ts`` descending (most
    recently active channel first). ``members`` is a sorted list of unique
    sender names observed on that channel.
    """
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    rows = await archive.query(event_type=EVENT_A2A, limit=100_000)

    channels: dict[str, dict] = {}
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        thread = data.get("thread") or row.get("app_id") or "general"
        sender = data.get("from") or ""
        ts = row.get("timestamp", 0.0)

        if thread not in channels:
            channels[thread] = {
                "channel": thread,
                "_members": set(),
                "message_count": 0,
                "created_ts": ts,
                "last_ts": ts,
            }
        ch = channels[thread]
        if sender:
            ch["_members"].add(sender)
        ch["message_count"] += 1
        if ts < ch["created_ts"]:
            ch["created_ts"] = ts
        if ts > ch["last_ts"]:
            ch["last_ts"] = ts

    result = []
    for ch in channels.values():
        result.append({
            "channel": ch["channel"],
            "members": sorted(ch["_members"]),
            "message_count": ch["message_count"],
            "created_ts": ch["created_ts"],
            "last_ts": ch["last_ts"],
        })
    result.sort(key=lambda c: c["last_ts"], reverse=True)
    return result


async def a2a_members(*, channel: str, data_dir=None) -> list[str]:
    """Return distinct sender names observed on ``channel``, sorted.

    Derived from :data:`~taosmd.archive.EVENT_A2A` events whose ``app_id``
    matches ``channel``. Returns an empty list (not an error) when the
    channel has never received a message.
    """
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    rows = await archive.query(event_type=EVENT_A2A, app_id=channel, limit=100_000)
    members: set[str] = set()
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        sender = data.get("from") or ""
        if sender:
            members.add(sender)
    return sorted(members)


__all__ = ["ingest", "search", "pending_list", "pending_resolve", "stats", "supersede",
           "a2a_send", "a2a_feed", "a2a_channels", "a2a_members"]
