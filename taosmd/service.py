"""Adapter-agnostic service layer over :mod:`taosmd.api`.

This is the shared core that activation surfaces sit on top of: the local
HTTP/REST server (#85) and the upcoming MCP server (#84) both call these
functions rather than reaching into :mod:`taosmd.api` directly. Keeping the
glue in one place means a single, consistent contract for every adapter and
guarantees their behaviour matches the Python API exactly.

The functions here are deliberately thin; they reuse
:func:`taosmd.api.ingest`, :func:`taosmd.api.search`,
:func:`taosmd.api.list_pending_decisions`, and
:func:`taosmd.api.resolve_pending_decision` (and therefore
``_ensure_stores`` / the stores cache / ``TAOSMD_DATA_DIR`` handling) so
the only thing they add is a uniform, transport-friendly signature:
``(positional, agent=..., data_dir=..., **opts)``.

Remote dispatch
---------------
When a server URL is configured (via ``TAOSMD_SERVER_URL`` or
``taosmd config set-server``) each function delegates to a cached
:class:`~taosmd.remote.RemoteClient` instead of running the local store.
The caller's signature is identical in both code paths, so the CLI, MCP
server, and Python API all go remote transparently.
"""

from __future__ import annotations

import json

from . import api as _api
from . import config as _config
from .archive import EVENT_A2A

# Cache of RemoteClient instances keyed by (base_url, token) so we don't
# create a fresh object on every call.  Access from async coroutines is safe
# because Python dict operations are GIL-protected.
_remote_cache: dict[tuple[str, str | None], object] = {}


def _get_remote(data_dir=None):
    """Return a cached :class:`~taosmd.remote.RemoteClient` when a server URL
    is configured, otherwise ``None`` (use local path).

    When ``data_dir`` is explicitly provided (as the http_server always does),
    the server-URL is resolved **only from the config file** in that data dir;
    the ``TAOSMD_SERVER_URL`` env var is intentionally ignored.  This prevents
    the running HTTP server from reading the env var and proxying its own
    requests back to itself (infinite loop).  The env override is only active
    for callers that do not specify a data_dir (CLI, MCP, Python API at the
    top level).
    """
    if data_dir is not None:
        # Explicit data_dir: config-file only, skip env.
        import json as _json  # noqa: PLC0415
        import os as _os  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415
        cfg_path = _Path(_os.fspath(data_dir)) / "config.json"
        try:
            cfg = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        except (OSError, _json.JSONDecodeError):
            cfg = {}
        url = cfg.get("server_url", "")
        if not isinstance(url, str) or not url.strip():
            return None
        url = url.strip()
        token_raw = cfg.get("server_token", "")
        token: str | None = token_raw.strip() if isinstance(token_raw, str) and token_raw.strip() else None
    else:
        # No explicit data_dir: use the full resolution (env override + config file).
        url = _config.get_server_url(data_dir)
        if not url:
            return None
        token = _config.get_server_token(data_dir)

    key = (url, token)
    client = _remote_cache.get(key)
    if client is None:
        from .remote import RemoteClient  # noqa: PLC0415
        client = RemoteClient(url, token=token)
        _remote_cache[key] = client
    return client


async def ingest(text, *, agent: str, data_dir=None, **opts) -> dict:
    """Shelve a transcript and embed it for later search.

    Thin wrapper over :func:`taosmd.api.ingest`. ``text`` may be a string,
    a turn dict, or an iterable of either (see the underlying API for the
    accepted shapes). Returns ``{"archived", "agent", "data_dir"}``.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.ingest(text, agent, **opts)
    return await _api.ingest(text, agent=agent, data_dir=data_dir, **opts)


async def search(query: str, *, agent: str, data_dir=None, limit: int = 5, **opts) -> list[dict]:
    """Search memory for passages relevant to ``query``.

    Thin wrapper over :func:`taosmd.api.search`. Returns ranked hits in the
    agent-rules contract shape (``text``/``source``/``timestamp``/
    ``confidence``/``metadata``).

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.search(query, agent, limit=limit, **opts)
    return await _api.search(query, agent=agent, data_dir=data_dir, limit=limit, **opts)


async def list_projects(*, data_dir=None) -> list[dict]:
    """List projects that have stored memories.

    Thin wrapper over :func:`taosmd.api.list_projects`. Forwarded to
    :class:`~taosmd.remote.RemoteClient` when a server URL is configured.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.list_projects()
    return await _api.list_projects(data_dir=data_dir)


async def list_shelves(*, project: str, data_dir=None) -> list[dict]:
    """List the agent shelves that have memories within ``project``.

    Thin wrapper over :func:`taosmd.api.list_shelves`. Forwarded to
    :class:`~taosmd.remote.RemoteClient` when a server URL is configured.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.list_shelves(project=project)
    return await _api.list_shelves(project=project, data_dir=data_dir)


async def pending_list(*, agent: str | None = None, data_dir=None, limit: int = 20) -> list[dict]:
    """Return unresolved KG-update decisions deferred by the librarian.

    ``agent`` is accepted for adapter symmetry; the pending-decisions queue
    is keyed per data dir (per install), not per agent, so it is not used to
    filter here. Use ``subject=`` on the underlying API if subject-level
    filtering is needed.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.pending_list(agent=agent, limit=limit)
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

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.pending_resolve(decision_id, decision, note=note)
    return await _api.resolve_pending_decision(
        decision_id, action=decision, note=note, data_dir=data_dir,
    )


async def reconcile(*, agent: str, data_dir=None, repair: bool = True) -> dict:
    """Detect and (when ``repair=True``) fix archive turns missing from the vector store.

    Thin wrapper over :func:`taosmd.api.reconcile`. The archive is the source of
    truth; the vector store is a derived index. A crash between the two sequential
    writes in :func:`ingest` leaves a turn in the archive but absent from vector
    recall. Reconcile re-adds those missing entries without touching anything that
    was deliberately superseded (superseded rows count as present).

    Returns ``{"agent", "archive_turns", "vector_entries", "missing", "readded",
    "checked_ok"}``. When ``repair=False`` this is a dry-run: ``readded`` is
    always 0.
    """
    return await _api.reconcile(agent=agent, data_dir=data_dir, repair=repair)


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

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` (best-effort via ``GET /health``).
    """
    if not agent:
        raise ValueError("agent name is required")
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.stats(agent=agent)
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

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    if not isinstance(sender, str) or not sender:
        raise ValueError("sender must be a non-empty string")
    if not isinstance(body, str) or not body:
        raise ValueError("body must be a non-empty string")
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_send(sender, body, thread=thread, reply_to=reply_to)
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

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_feed(thread=thread, since=since, limit=limit)
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
    events, no additional schema. Groups by ``app_id`` (which equals the
    thread name) and aggregates membership, message count, and timestamps.

    Each item has shape ``{"channel", "members", "message_count",
    "created_ts", "last_ts"}``, sorted by ``last_ts`` descending (most
    recently active channel first). ``members`` is a sorted list of unique
    sender names observed on that channel.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_channels()
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

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_members(channel=channel)
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


__all__ = ["ingest", "search", "pending_list", "pending_resolve", "reconcile", "stats",
           "supersede", "a2a_send", "a2a_feed", "a2a_channels", "a2a_members"]
