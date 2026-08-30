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

import hashlib
import json
import logging
import math
import re
import time

from . import api as _api
from . import config as _config
from .archive import EVENT_A2A
from .mentions import MentionStore, _normalise_handle

logger = logging.getLogger(__name__)

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


async def ingest_batch(items, *, agent: str, data_dir=None, **opts) -> dict:
    """Bulk-shelve memory chunks with idempotent re-import.

    Thin wrapper over :func:`taosmd.api.ingest_batch`. ``items`` is a list of
    ``{"text", "id"?, "metadata"?}`` dicts; items whose ``id`` was already
    ingested are skipped. Returns ``{"ingested", "skipped", "agent",
    "data_dir"}``.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.ingest_batch(items, agent, **opts)
    return await _api.ingest_batch(items, agent=agent, data_dir=data_dir, **opts)


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


async def dashboard_stats(*, scope: str | None = None, data_dir=None) -> dict:
    """Aggregate dashboard stats over the stores, optionally scoped to one agent.

    Thin wrapper over :func:`taosmd.api.dashboard_stats`. Forwarded to
    :class:`~taosmd.remote.RemoteClient` when a server URL is configured.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.dashboard_stats(scope=scope)
    return await _api.dashboard_stats(scope=scope, data_dir=data_dir)


async def list_memories(*, scope: str | None = None, limit: int = 50, data_dir=None) -> list[dict]:
    """Recent archived memories for the dashboard browse view (scoped by ``scope``).

    Thin wrapper over :func:`taosmd.api.list_memories`. Forwarded to
    :class:`~taosmd.remote.RemoteClient` when a server URL is configured.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.list_memories(scope=scope, limit=limit)
    return await _api.list_memories(scope=scope, limit=limit, data_dir=data_dir)


async def graph(*, limit: int = 300, as_of: float | None = None, data_dir=None) -> dict:
    """Knowledge-graph nodes and edges for the Explorer view.

    ``as_of`` (unix seconds) reconstructs the graph as of that instant for the
    time-travel scrubber; ``None`` (default) returns the current graph.

    Thin wrapper over :func:`taosmd.api.graph`. Forwarded to
    :class:`~taosmd.remote.RemoteClient` when a server URL is configured.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.graph(limit=limit, as_of=as_of)
    return await _api.graph(limit=limit, as_of=as_of, data_dir=data_dir)


async def graph_activations(*, since: float | None = None, window: float = 60.0,
                            limit: int = 100, data_dir=None) -> dict:
    """Entities recalled recently, for the Explorer live-recall pulse.

    Thin wrapper over :func:`taosmd.api.graph_activations`. Forwarded to
    :class:`~taosmd.remote.RemoteClient` when a server URL is configured.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.graph_activations(since=since, window=window, limit=limit)
    return await _api.graph_activations(since=since, window=window, limit=limit, data_dir=data_dir)


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


async def reindex(*, agent: str, data_dir=None, check: bool = False) -> dict:
    """Re-embed an agent's vector store from the zero-loss archive.

    Thin wrapper over :func:`taosmd.api.reindex`. The append-only archive is the
    source of truth; the vector store is a derived index. Switching embedders
    (e.g. MiniLM -> arctic-embed-s) leaves the old vectors in an incompatible
    space, so reindex clears the agent's vector rows and rebuilds them by
    re-adding every archive turn, which re-embeds each one under the *currently
    configured* embedder. The archive is never touched, so reindex is safe to
    re-run and is applied per-agent (live agents cut over one at a time).

    Returns ``{"agent", "archive_turns", "vector_before", "cleared", "readded",
    "reindexed_ok"}``. When ``check=True`` this is a dry-run: ``cleared`` and
    ``readded`` are 0 and nothing is modified.
    """
    return await _api.reindex(agent=agent, data_dir=data_dir, check=check)


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


async def fetch_by_ref(ref: dict, *, agent: str, data_dir=None) -> dict:
    """Fetch and verify bytes for a taOS Files-backed ref.

    Thin wrapper over :func:`taosmd.ref_fetch.fetch_by_ref`. Resolves the
    controller base URL from config, builds a fetcher that authenticates with
    the registry token, and returns the verified bytes as a base64-encoded
    string together with its sha256 and size.

    Returns ``{"bytes": <base64-str>, "sha256": <hash>, "size": <int>}``.

    Raises :class:`ValueError` for an unresolvable uri,
    :class:`~taosmd.ref_fetch.HashMismatchError` for a hash mismatch,
    :class:`~taosmd.ref_fetch.NotFoundError` for a 404, or
    :class:`~taosmd.ref_fetch.UnauthorizedError` for a 401/403.
    """
    import base64

    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.fetch_by_ref(ref, agent=agent)

    from . import config as _config
    from .ref_fetch import NotFoundError, RefFetchError, UnauthorizedError, fetch_by_ref as _fetch_by_ref

    registry_token = _config.get_registry_token(data_dir)

    def _fetcher(url: str, agent: str) -> bytes:
        import urllib.error
        import urllib.request

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        headers = {"Accept": "application/octet-stream"}
        if registry_token:
            headers["Authorization"] = f"Bearer {registry_token}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            opener = urllib.request.build_opener(_NoRedirect)
            with opener.open(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise UnauthorizedError(f"HTTP {exc.code} from {url}") from exc
            if exc.code == 404:
                raise NotFoundError(f"HTTP 404 from {url}") from exc
            raise
        except urllib.error.URLError as exc:
            raise RefFetchError(f"fetch failed for {url}: {exc}") from exc

    raw = await _fetch_by_ref(ref, _fetcher, agent, data_dir=data_dir)
    return {
        "bytes": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }


_A2A_KINDS = frozenset({"chat", "alarm", "ack", "digest", "receipt", "review", "system"})
_A2A_ALARM_MIN_INTERVAL = 5.0


async def a2a_send(
    sender: str,
    body: str,
    *,
    thread: str = "general",
    reply_to: str | None = None,
    refs: list | None = None,
    blocks: list | None = None,
    recipient: str | None = None,
    kind: str = "chat",
    alarm_key: str | None = None,
    alarm_fingerprint: str | None = None,
    data_dir=None,
) -> dict:
    """Post a message onto the agent-to-agent bus.

    Stores the message as an append-only archive event of type
    :data:`~taosmd.archive.EVENT_A2A` and returns a receipt with the
    assigned row ID. ``sender`` and ``body`` must be non-empty strings;
    ``thread`` defaults to ``"general"``; ``reply_to`` is optional and
    should be the string ID of the message being replied to.

    ``kind`` is one of ``chat``, ``alarm``, ``ack``, ``digest``,
    ``receipt``, ``review``, ``system`` (default ``chat``). It is stored
    on the envelope and returned in every read path.

    ``alarm_key`` and ``alarm_fingerprint`` apply to ``kind="alarm"``
    messages. When ``alarm_fingerprint`` is omitted the server computes
    ``sha256(body)``. A same-(key, fingerprint) alarm within the
    module-level min interval is not stored: the send answers
    ``{"deduped": true}``. The dedup guarantee relies on the single
    service loop serialising the read and write calls.

    ``refs`` and ``blocks`` are optional first-class envelope fields
    (taOSmd #211). When provided they are stored verbatim in the archive
    payload and echoed back in the receipt and on feed/SSE reads. When
    absent they are omitted from output entirely (no null noise).

    ``recipient`` is an optional explicit mention target; when provided it
    is stored in the archive payload and indexed as a mention so the
    recipient can retrieve it via GET /a2a/mentions.

    Returns ``{"id", "from", "thread", "reply_to", "kind"}`` plus ``refs``
    and/or ``blocks`` when those were supplied. For deduped alarms returns
    ``{"deduped": true, "kind": "alarm"}``.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    if not isinstance(sender, str) or not sender:
        raise ValueError("sender must be a non-empty string")
    if not isinstance(body, str) or not body:
        raise ValueError("body must be a non-empty string")
    if kind not in _A2A_KINDS:
        raise ValueError(
            f"'kind' must be one of {sorted(_A2A_KINDS)}; got {kind!r}"
        )
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_send(
            sender, body, thread=thread, reply_to=reply_to,
            refs=refs, blocks=blocks, recipient=recipient, kind=kind,
            alarm_key=alarm_key, alarm_fingerprint=alarm_fingerprint,
        )
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    # Redirect sends to renamed channels: if the target thread has been aliased
    # to a new name, route the message to the canonical name instead.
    if data_dir is not None:
        from .admin import A2AAdminState  # noqa: PLC0415
        _admin = A2AAdminState(data_dir)
        thread = _admin.resolve_channel(thread)

    # Alarm dedup: server-enforced, store-backed. The single service loop
    # serialises the read and write below, so no two alarms race.
    deduped = False
    if kind == "alarm" and isinstance(alarm_key, str) and alarm_key:
        fp = alarm_fingerprint if isinstance(alarm_fingerprint, str) and alarm_fingerprint else None
        if fp is None:
            fp = hashlib.sha256(body.encode()).hexdigest()
        now = time.time()
        state = await archive.get_alarm_dedup(alarm_key, fp)
        if state is not None:
            last_fire = state["last_fire_ts"] or 0.0
            last_cleared = state["last_cleared_ts"] or 0.0
            if last_fire > (now - _A2A_ALARM_MIN_INTERVAL) and last_cleared < last_fire:
                deduped = True
        if not deduped:
            await archive.record_alarm_dedup(alarm_key, fp, now)

    if deduped:
        return {"deduped": True, "kind": "alarm"}

    data = {"from": sender, "body": body, "thread": thread, "reply_to": reply_to, "kind": kind}
    if recipient is not None:
        data["recipient"] = recipient
    if refs is not None:
        data["refs"] = refs
    if blocks is not None:
        data["blocks"] = blocks
    if kind == "alarm" and alarm_key is not None:
        data["alarm_key"] = alarm_key
        fp = alarm_fingerprint if isinstance(alarm_fingerprint, str) and alarm_fingerprint else None
        if fp is not None:
            data["alarm_fingerprint"] = fp
    row_id = await archive.record(
        event_type=EVENT_A2A,
        data=data,
        agent_name=sender,
        app_id=thread,
        summary=body[:200],
    )
    receipt = {"id": row_id, "from": sender, "thread": thread, "reply_to": reply_to, "kind": kind}
    if recipient is not None:
        receipt["recipient"] = recipient
    if refs is not None:
        receipt["refs"] = refs
    if blocks is not None:
        receipt["blocks"] = blocks
    # Index @handle mentions for the A2A mention feed (#211).
    stored = await archive.get_event(row_id)
    ts = stored["timestamp"] if stored else time.time()
    mentions = stores.get("mentions")
    if isinstance(mentions, MentionStore):
        await mentions.record_mentions(
            message_id=row_id,
            body=body,
            thread=thread,
            ts=ts,
            recipient=recipient,
        )
    return receipt


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
    "reply_to"}`` plus ``refs`` and/or ``blocks`` when those were
    supplied on send (taOSmd #211).

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_feed(thread=thread, since=since, limit=limit)
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]

    # Apply admin alias resolution: reads of a new channel name include history
    # from the old name. Resolve thread through the alias map so callers
    # querying the canonical name see both old and new messages.
    alias_sources: list[str] = []
    if data_dir is not None:
        from .admin import A2AAdminState  # noqa: PLC0415
        _admin_state = A2AAdminState(data_dir)
        _aliases = _admin_state.channel_aliases()
        _deleted = _admin_state.deleted_channels()
        _superseded = _admin_state.superseded_messages()
        # Find all channel names that alias to thread (so we can include
        # their history when querying the canonical name).
        if thread is not None:
            alias_sources = [k for k, v in _aliases.items() if v == thread]
    else:
        _deleted = set()
        _superseded = set()
        alias_sources = []

    # Query with no thread filter when we need to merge history from aliases
    if alias_sources and thread is not None:
        rows_all = await archive.query(event_type=EVENT_A2A, since=since, limit=limit * 10)
        rows = [
            r for r in rows_all
            if (r.get("app_id") == thread or r.get("app_id") in alias_sources)
        ]
        rows = rows[:limit]
    else:
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
        # Skip admin-suppressed items
        row_id = row["id"]
        if row_id in _superseded:
            continue
        msg_thread = data.get("thread") or row.get("app_id") or "general"
        # A deleted channel is delisted and unreadable by its own name, but if
        # it was renamed first its rows are still surfaced as alias-merged
        # history under the (live) canonical thread. Only skip a deleted-thread
        # row when it is NOT being merged in as alias history for this query,
        # so "rename then delete the old name" keeps the history under the new
        # name without mutating the zero-loss archive.
        if msg_thread in _deleted and msg_thread not in alias_sources:
            continue
        # Skip admin-action rows (they have no "from" field)
        if data.get("admin_action"):
            continue
        msg = {
            "id": row_id,
            "ts": row["timestamp"],
            "from": data.get("from"),
            "body": data.get("body"),
            "thread": msg_thread,
            "reply_to": data.get("reply_to"),
            "kind": data.get("kind") or "chat",
        }
        # First-class envelope fields (taOSmd #211): stored verbatim,
        # omitted from output when absent (no null noise).
        if "refs" in data:
            msg["refs"] = data["refs"]
        if "blocks" in data:
            msg["blocks"] = data["blocks"]
        if "acked_by" in data:
            msg["acked_by"] = data["acked_by"]
        result.append(msg)
    return result


async def a2a_alarms_clear(alarm_key: str, *, data_dir=None) -> dict:
    """Clear the dedup cooldown for an alarm key.

    The next same-key alarm will store, after which the cooldown re-applies
    to subsequent duplicates. The guarantee relies on the single service
    loop serialising reads and writes.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    if not isinstance(alarm_key, str) or not alarm_key:
        raise ValueError("alarm_key must be a non-empty string")
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_alarms_clear(alarm_key)
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    now = time.time()
    await archive.clear_alarm_key(alarm_key, now)
    return {"cleared": True, "key": alarm_key}


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

    # Load admin state once for filtering
    deleted: set[str] = set()
    aliases: dict[str, str] = {}
    superseded: set[int] = set()
    if data_dir is not None:
        from .admin import A2AAdminState  # noqa: PLC0415
        _admin = A2AAdminState(data_dir)
        deleted = _admin.deleted_channels()
        aliases = _admin.channel_aliases()
        superseded = _admin.superseded_messages()

    channels: dict[str, dict] = {}
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        # Skip admin-action rows and superseded messages
        if data.get("admin_action"):
            continue
        if row["id"] in superseded:
            continue
        thread = data.get("thread") or row.get("app_id") or "general"
        # Redirect aliased channels to their canonical name
        if thread in aliases:
            thread = aliases[thread]
        # Skip deleted channels
        if thread in deleted:
            continue
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


async def a2a_sender_census(*, data_dir=None) -> dict:
    """Return a per-sender message census across every A2A channel.

    Queries all :data:`~taosmd.archive.EVENT_A2A` events and aggregates them
    by sender.  The result maps each distinct ``from`` value to a dict with:

    * ``total`` -- total messages sent by that sender across all channels
    * ``channels`` -- mapping of channel name to message count on that channel

    Senders are returned in descending order of ``total``.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_sender_census()
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    rows = await archive.query(event_type=EVENT_A2A, limit=100_000)

    deleted: set[str] = set()
    aliases: dict[str, str] = {}
    superseded: set[int] = set()
    if data_dir is not None:
        from .admin import A2AAdminState  # noqa: PLC0415
        _admin = A2AAdminState(data_dir)
        deleted = _admin.deleted_channels()
        aliases = _admin.channel_aliases()
        superseded = _admin.superseded_messages()

    census: dict[str, dict] = {}
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        if data.get("admin_action"):
            continue
        row_id = row["id"]
        if row_id in superseded:
            continue
        sender = data.get("from") or ""
        if not sender:
            continue
        thread = data.get("thread") or row.get("app_id") or "general"
        if thread in aliases:
            thread = aliases[thread]
        if thread in deleted:
            continue
        if sender not in census:
            census[sender] = {"total": 0, "channels": {}}
        entry = census[sender]
        entry["total"] += 1
        entry["channels"][thread] = entry["channels"].get(thread, 0) + 1

    return dict(
        sorted(census.items(), key=lambda item: item[1]["total"], reverse=True)
    )


async def a2a_migrate_kinds(*, data_dir=None) -> dict:
    """One-shot migration: backfill ``kind`` on historical A2A messages.

    Messages that already have a ``kind`` field are left untouched. Messages
    without one are tagged by their body-prefix convention:
    ``[AUTOMATED`` -> ``alarm``, ``[AUTO-ACK]`` -> ``ack``,
    ``[REVIEW]`` -> ``review``, everything else -> ``chat``.

    Returns ``{"migrated": int, "alarm": int, "ack": int, "review": int,
    "chat": int}``. Idempotent: running twice yields ``migrated == 0``.
    """
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    rows = await archive.query(event_type=EVENT_A2A, limit=100_000)
    counts = {"migrated": 0, "alarm": 0, "ack": 0, "review": 0, "chat": 0}
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        if data.get("kind"):
            continue
        body = data.get("body", "") or ""
        if body.startswith("[AUTOMATED"):
            kind = "alarm"
        elif body.startswith("[AUTO-ACK]"):
            kind = "ack"
        elif body.startswith("[REVIEW]"):
            kind = "review"
        else:
            kind = "chat"
        data["kind"] = kind
        await archive.update_event_data_json(row["id"], data)
        counts[kind] += 1
        counts["migrated"] += 1
    return counts


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


async def a2a_threads(*, principal: str | None = None, data_dir=None) -> list[dict]:
    """Return a summary of every thread on the A2A bus.

    Derived from :data:`~taosmd.archive.EVENT_A2A` events. Groups by thread
    name and aggregates participants and the last message preview.

    Each item has shape ``{"thread", "kind", "participants",
    "last_message": {"id", "ts", "from", "body_preview"}}``.

    Threads are returned **most-recently-active first**, ordered by the
    timestamp of each thread's last message. This replaced an alphabetical
    ordering; callers that relied on alphabetical order must sort client-side.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_threads(principal=principal)
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    rows = await archive.query(event_type=EVENT_A2A, limit=100_000)

    deleted: set[str] = set()
    aliases: dict[str, str] = {}
    superseded: set[int] = set()
    if data_dir is not None:
        from .admin import A2AAdminState  # noqa: PLC0415
        _admin = A2AAdminState(data_dir)
        deleted = _admin.deleted_channels()
        aliases = _admin.channel_aliases()
        superseded = _admin.superseded_messages()

    threads: dict[str, dict] = {}
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        if data.get("admin_action"):
            continue
        if row["id"] in superseded:
            continue
        thread = data.get("thread") or row.get("app_id") or "general"
        if thread in aliases:
            thread = aliases[thread]
        if thread in deleted:
            continue
        sender = data.get("from") or ""
        ts = row.get("timestamp", 0.0)
        body = data.get("body") or ""
        if thread not in threads:
            threads[thread] = {
                "thread": thread,
                "kind": "channel",
                "participants": set(),
                "last_message": None,
                "_last_ts": -1.0,
            }
        t = threads[thread]
        if sender:
            t["participants"].add(sender)
        if ts > t["_last_ts"]:
            t["_last_ts"] = ts
            preview = body[:120] + ("..." if len(body) > 120 else "")
            t["last_message"] = {
                "id": row["id"],
                "ts": ts,
                "from": sender,
                "body_preview": preview,
            }

    result = []
    for t in threads.values():
        result.append({
            "thread": t["thread"],
            "kind": t["kind"],
            "participants": sorted(t["participants"]),
            "last_message": t["last_message"],
            "_last_ts": t["_last_ts"],
        })
    result.sort(key=lambda x: x["_last_ts"], reverse=True)
    return [{"thread": t["thread"], "kind": t["kind"],
             "participants": t["participants"],
             "last_message": t["last_message"]} for t in result]


async def a2a_thread_messages(
    *, thread: str, before: int | float | None = None,
    after: int | float | None = None, limit: int = 50, data_dir=None,
) -> dict:
    """Return cursor-paginated messages for a thread, oldest-first.

    ``before`` and ``after`` are message ids (int) or timestamps (float
    >= 1e9). ``limit`` defaults to 50, max 200.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_thread_messages(
            thread=thread, before=before, after=after, limit=limit,
        )
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    rows = await archive.query(event_type=EVENT_A2A, app_id=thread, limit=100_000)

    deleted: set[str] = set()
    superseded: set[int] = set()
    if data_dir is not None:
        from .admin import A2AAdminState  # noqa: PLC0415
        _admin = A2AAdminState(data_dir)
        deleted = _admin.deleted_channels()
        superseded = _admin.superseded_messages()

    messages = []
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        if data.get("admin_action"):
            continue
        if row["id"] in superseded:
            continue
        msg_thread = data.get("thread") or row.get("app_id") or "general"
        if msg_thread in deleted:
            continue
        msg = {
            "id": row["id"],
            "ts": row["timestamp"],
            "from": data.get("from"),
            "body": data.get("body"),
            "thread": msg_thread,
            "reply_to": data.get("reply_to"),
            "kind": data.get("kind") or "chat",
        }
        if "refs" in data:
            msg["refs"] = data["refs"]
        if "blocks" in data:
            msg["blocks"] = data["blocks"]
        if "acked_by" in data:
            msg["acked_by"] = data["acked_by"]
        messages.append(msg)

    def _cursor_val(cursor):
        if cursor is None:
            return None
        if isinstance(cursor, int):
            return ("id", cursor)
        if isinstance(cursor, float):
            return ("ts", cursor)
        return None

    before_type, before_val = _cursor_val(before) or (None, None)
    after_type, after_val = _cursor_val(after) or (None, None)

    if before_type == "id":
        messages = [m for m in messages if m["id"] < before_val]
    elif before_type == "ts":
        messages = [m for m in messages if m["ts"] < before_val]

    if after_type == "id":
        messages = [m for m in messages if m["id"] > after_val]
    elif after_type == "ts":
        messages = [m for m in messages if m["ts"] > after_val]

    messages.sort(key=lambda m: (m["ts"], m["id"]))
    limit_i = max(1, min(limit, 200))
    messages = messages[:limit_i]
    return {"thread": thread, "messages": messages}


async def a2a_mentions_feed(
    reader: str,
    *,
    since: float | None = None,
    limit: int = 50,
    data_dir=None,
) -> list[dict]:
    """Return messages that mention ``reader`` plus their reply_to chains.

    Each item has shape ``{"id", "ts", "from", "body", "thread",
    "reply_to", "thread_root"}``. Results are ordered oldest-first and
    capped by ``limit``.

    Thread-scoped visibility (#211 anti-bypass): a mention grants access
    to the mentioned message and the full reply_to chain rooted at it,
    but not to unrelated sibling messages in the same channel.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    if not isinstance(limit, int) or math.isnan(limit) or math.isinf(limit) or limit <= 0:
        raise ValueError("limit must be a positive finite integer")
    if since is not None and (math.isnan(since) or math.isinf(since)):
        raise ValueError("since must be a finite float timestamp or None")
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_mentions_feed(reader, since=since, limit=limit)
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    mentions_store = stores["mentions"]

    norm_reader = _normalise_handle(reader)
    mentioned_rows = await mentions_store.get_mentioned_message_ids(
        norm_reader, since=since, limit=limit,
    )
    mentioned_ids = {r["message_id"] for r in mentioned_rows}
    if not mentioned_ids:
        return []

    all_rows = await archive.query(event_type=EVENT_A2A, limit=100_000)
    msg_thread: dict[int, str] = {}
    children: dict[int, list[dict]] = {}
    for row in all_rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        thread = data.get("thread") or row.get("app_id") or "general"
        msg_thread[row["id"]] = thread
        reply_to = data.get("reply_to")
        if reply_to is not None:
            try:
                parent_id = int(reply_to)
                children.setdefault(parent_id, []).append(row)
            except (TypeError, ValueError):
                continue

    reply_chain_ids = set(mentioned_ids)
    queue = list(mentioned_ids)
    while queue:
        parent_id = queue.pop()
        for child_row in children.get(parent_id, []):
            if child_row["id"] in reply_chain_ids:
                continue
            child_thread = msg_thread.get(child_row["id"])
            parent_thread = msg_thread.get(parent_id)
            if child_thread and parent_thread and child_thread == parent_thread:
                reply_chain_ids.add(child_row["id"])
                queue.append(child_row["id"])

    thread_roots: dict[int, int] = {}
    for mid in reply_chain_ids:
        root = await _find_thread_root(mid, archive)
        if root is not None:
            thread_roots[mid] = root

    result = []
    for row in all_rows:
        if row["id"] not in reply_chain_ids:
            continue
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        if data.get("admin_action"):
            continue
        msg = {
            "id": row["id"],
            "ts": row["timestamp"],
            "from": data.get("from"),
            "body": data.get("body"),
            "thread": data.get("thread") or row.get("app_id") or "general",
            "reply_to": data.get("reply_to"),
            "thread_root": thread_roots.get(row["id"]),
            "kind": data.get("kind") or "chat",
        }
        result.append(msg)

    result.sort(key=lambda m: m["ts"])
    return result[:limit]


async def _find_thread_root(message_id: int, archive) -> int | None:
    """Walk reply_to links upward to find the root message ID."""
    visited: set[int] = set()
    current_id = message_id
    while current_id:
        if current_id in visited:
            break
        visited.add(current_id)
        row = await archive.get_event(current_id)
        if not row:
            break
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            break
        reply_to = data.get("reply_to")
        if reply_to is None:
            return current_id
        try:
            current_id = int(reply_to)
        except (TypeError, ValueError):
            break
    return None


async def can_read(reader: str, msg: dict, data_dir=None) -> bool:
    """Thread-scoped read guard (#211 anti-bypass).

    ``canRead(reader, msg) = channelACL(reader, msg.thread) OR
    mentionGrant(reader, threadRoot(msg))``

    A mention grants visibility of the mentioned message and its full
    reply_to chain, but never widens channel access. Channel ACL
    enforcement (tsk-dp6fyv) plugs into the ``channelACL`` slot; until
    then it is effectively always-true for compatibility.
    """
    return True


async def a2a_inbox(
    consumer: str,
    *,
    limit: int = 50,
    include_kinds: list | None = None,
    data_dir=None,
) -> list[dict]:
    """Return messages past ``consumer``'s cursor that are addressed to it.

    A message is addressed when at least one of:
    - the consumer's handle is mentioned in the body
    - the message is in a thread owned by the consumer (thread name == consumer)
    - the message has a direct ``recipient`` matching the consumer

    The consumer's own posts are always excluded.  By default kinds
    ``alarm``, ``ack``, ``receipt``, and ``digest`` are excluded; pass
    ``include_kinds`` to widen the set.  Results are oldest-first.  Reading
    does NOT advance the cursor.
    """
    if not isinstance(consumer, str) or not consumer:
        raise ValueError("consumer must be a non-empty string")
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    cursor = await archive.get_a2a_inbox_cursor(consumer)

    rows = await archive.query(event_type=EVENT_A2A, limit=100_000)

    excluded_kinds = {"alarm", "ack", "receipt", "digest"}
    if include_kinds is not None:
        excluded_kinds -= set(include_kinds)
    allowed_kinds = _A2A_KINDS - excluded_kinds

    norm_consumer = _normalise_handle(consumer)
    mention_re = re.compile(r'(?<![\w/])@([a-zA-Z0-9_-]+)')

    result = []
    for row in rows:
        if row["id"] <= cursor:
            continue
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        sender = data.get("from") or ""
        if sender == consumer:
            continue
        kind = data.get("kind") or "chat"
        if kind not in allowed_kinds:
            continue
        body = data.get("body") or ""
        thread = data.get("thread") or row.get("app_id") or "general"
        recipient = data.get("recipient")
        addressed = False
        if recipient == consumer:
            addressed = True
        elif thread == consumer:
            addressed = True
        else:
            for m in mention_re.finditer(body):
                if _normalise_handle(m.group(1)) == norm_consumer:
                    addressed = True
                    break
        if not addressed:
            continue
        msg = {
            "id": row["id"],
            "ts": row["timestamp"],
            "from": sender,
            "body": body,
            "thread": thread,
            "reply_to": data.get("reply_to"),
            "kind": kind,
        }
        if "refs" in data:
            msg["refs"] = data["refs"]
        if "blocks" in data:
            msg["blocks"] = data["blocks"]
        result.append(msg)

    result.sort(key=lambda m: (m["ts"], m["id"]))
    limit_i = max(1, min(limit, 1000))
    return result[:limit_i]


async def a2a_inbox_advance(
    consumer: str,
    to_id: int,
    *,
    data_dir=None,
) -> dict:
    """Advance ``consumer``'s inbox cursor to ``to_id``.

    The cursor is persisted in the archive store so it survives restarts
    and is visible to every process sharing the same data dir.
    """
    if not isinstance(consumer, str) or not consumer:
        raise ValueError("consumer must be a non-empty string")
    if not isinstance(to_id, int) or to_id < 0:
        raise ValueError("to_id must be a non-negative integer")
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    await archive.set_a2a_inbox_cursor(consumer, to_id)
    return {"ok": True}


async def a2a_record_delivered(
    message_id: int, agent_id: str, *, ts: float | None = None, data_dir=None
) -> dict:
    """Record that a message was delivered to an agent.

    Thin wrapper over :func:`taosmd.receipts.ReceiptStore.record_delivered`.
    ``ts`` defaults to ``time.time()`` when not supplied.
    Returns ``{"ok": True}``.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    if ts is None:
        ts = time.time()
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_record_delivered(message_id, agent_id, ts=ts)
    stores = await _api._ensure_stores(data_dir)
    receipt_store = stores["receipts"]
    await receipt_store.record_delivered(message_id, agent_id, ts)
    return {"ok": True}


async def a2a_record_seen(
    message_id: int, agent_id: str, *, ts: float | None = None, data_dir=None
) -> dict:
    """Record that an agent has seen a message.

    Thin wrapper over :func:`taosmd.receipts.ReceiptStore.record_seen`.
    ``ts`` defaults to ``time.time()`` when not supplied.
    Returns ``{"ok": True}``.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    if ts is None:
        ts = time.time()
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_record_seen(message_id, agent_id, ts=ts)
    stores = await _api._ensure_stores(data_dir)
    receipt_store = stores["receipts"]
    await receipt_store.record_seen(message_id, agent_id, ts)
    return {"ok": True}


async def a2a_get_receipts(message_id: int, *, data_dir=None) -> dict:
    """Return delivery and read receipts for a message.

    Thin wrapper over :func:`taosmd.receipts.ReceiptStore.get_receipts_for_message`.
    Returns ``{"delivered": [...], "read": [...]}``.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_get_receipts(message_id)
    stores = await _api._ensure_stores(data_dir)
    receipt_store = stores["receipts"]
    return await receipt_store.get_receipts_for_message(message_id)


async def a2a_get_receipt(
    message_id: int, agent_id: str, *, data_dir=None
) -> dict | None:
    """Return a single receipt or ``None`` when not found.

    Thin wrapper over :func:`taosmd.receipts.ReceiptStore.get_receipt`.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_get_receipt(message_id, agent_id)
    stores = await _api._ensure_stores(data_dir)
    receipt_store = stores["receipts"]
    return await receipt_store.get_receipt(message_id, agent_id)


async def a2a_prune_receipts(
    older_than_ts: float, *, data_dir=None
) -> dict:
    """Prune receipts older than ``older_than_ts`` (by delivered_at).

    Thin wrapper over :func:`taosmd.receipts.ReceiptStore.prune`.
    Returns ``{"pruned": int}``.

    When a remote server URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_prune_receipts(older_than_ts)
    stores = await _api._ensure_stores(data_dir)
    receipt_store = stores["receipts"]
    n = await receipt_store.prune(older_than_ts)
    return {"pruned": n}


async def a2a_ack(message_id: int, by: str, *, data_dir=None) -> dict:
    """Record that a principal has acknowledged a message, as server state.

    Per the A2A delivery contract v2 (section 3), an acknowledgement is
    never a new bus message: it mutates the message envelope in place by
    appending ``by`` to an ``acked_by`` list on the archived event, via
    :meth:`taosmd.archive.ArchiveStore.update_event_data_json`. The JSONL
    source files are never touched; only the derived index is updated.

    Idempotent: acking the same message twice by the same principal leaves
    exactly one entry in ``acked_by``. The ``acked_by`` list is surfaced
    on the envelope by the existing read paths (``a2a_feed`` and
    ``a2a_thread_messages``).

    The composed "unhandled for X" query (mentions past X's cursor minus
    acks) is deferred to slice 2c, which needs 2a's server-side cursor.

    Returns ``{"id", "acked_by", "ok"}``.
    """
    if not isinstance(by, str) or not by:
        raise ValueError("by must be a non-empty string")
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    stored = await archive.get_event(message_id)
    if stored is None:
        raise ValueError(f"message {message_id} not found")
    data = dict(stored.get("data") or {})
    acked_by = data.get("acked_by")
    if not isinstance(acked_by, list):
        acked_by = []
    if by not in acked_by:
        acked_by = [*acked_by, by]
    data["acked_by"] = acked_by
    await archive.update_event_data_json(message_id, data)
    return {"id": message_id, "acked_by": acked_by, "ok": True}


async def task_create(
    title: str,
    *,
    body: str | None = None,
    project: str | None = None,
    assignee: str | None = None,
    priority: int = 0,
    depends_on: list[str] | None = None,
    created_by: str,
    data_dir=None,
) -> dict:
    """Create a task and return the task object.

    Thin wrapper over :func:`taosmd.tasks.create_task`. When a remote server
    URL is configured the call is forwarded to
    :class:`~taosmd.remote.RemoteClient` transparently.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.task_create(
            title, body=body, project=project, assignee=assignee,
            priority=priority, depends_on=depends_on, created_by=created_by,
        )
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.create_task(
        title, body=body, project=project, assignee=assignee,
        priority=priority, depends_on=depends_on, created_by=created_by,
        data_dir=data_dir,
    )


async def task_list(
    *,
    status: str | None = None,
    project: str | None = None,
    assignee: str | None = None,
    limit: int = 50,
    data_dir=None,
) -> list[dict]:
    """Return tasks matching the given filters.

    Thin wrapper over :func:`taosmd.tasks.list_tasks`.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.task_list(
            status=status, project=project, assignee=assignee, limit=limit
        )
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.list_tasks(
        status=status, project=project, assignee=assignee,
        limit=limit, data_dir=data_dir,
    )


async def task_ready(
    *,
    project: str | None = None,
    assignee: str | None = None,
    limit: int = 20,
    data_dir=None,
) -> list[dict]:
    """Return the ready-queue ordered list.

    Thin wrapper over :func:`taosmd.tasks.ready_tasks`.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.task_ready(
            project=project, assignee=assignee, limit=limit
        )
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.ready_tasks(
        project=project, assignee=assignee, limit=limit, data_dir=data_dir,
    )


async def task_prime(
    *,
    project: str | None = None,
    assignee: str | None = None,
    data_dir=None,
) -> dict:
    """Return the prime session-bootstrap briefing.

    Thin wrapper over :func:`taosmd.tasks.prime`.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.task_prime(project=project, assignee=assignee)
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.prime(project=project, assignee=assignee, data_dir=data_dir)


async def task_list_edges(
    *,
    from_id: str | None = None,
    to_id: str | None = None,
    type: str | None = None,
    project: str | None = None,
    limit: int = 50,
    data_dir=None,
) -> list[dict]:
    """Return task edges matching the given filters.

    Thin wrapper over :func:`taosmd.tasks.list_edges`.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.task_list_edges(
            from_id=from_id, to_id=to_id, type=type,
            project=project, limit=limit,
        )
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.list_edges(
        from_id=from_id, to_id=to_id, type=type,
        project=project, limit=limit, data_dir=data_dir,
    )


async def task_update(
    task_id: str,
    *,
    status: str | None = None,
    assignee: str | None = None,
    priority: int | None = None,
    body: str | None = None,
    data_dir=None,
) -> dict:
    """Update a task and return the updated object.

    Thin wrapper over :func:`taosmd.tasks.update_task`.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.task_update(
            task_id, status=status, assignee=assignee,
            priority=priority, body=body,
        )
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.update_task(
        task_id, status=status, assignee=assignee,
        priority=priority, body=body, data_dir=data_dir,
    )


async def task_add_edge(
    from_id: str,
    to_id: str,
    edge_type: str,
    created_by: str,
    *,
    data_dir=None,
) -> dict:
    """Add an edge between two tasks.

    Thin wrapper over :func:`taosmd.tasks.add_edge`.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.task_add_edge(from_id, to_id, edge_type, created_by)
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.add_edge(
        from_id, to_id, edge_type, created_by, data_dir=data_dir
    )


async def task_projects(task_ids: list[str], *, data_dir=None) -> dict:
    """Return ``{task_id: project}`` for the ids that exist locally.

    Auth-layer helper for the HTTP server's edge-endpoint project scoping.
    This intentionally does NOT forward to a remote server: the server that
    enforces token binding is the owner of the task store, so the lookup
    always reads the local projection (fails closed when the tasks are not
    present locally).
    """
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.get_task_projects(task_ids, data_dir=data_dir)


async def task_remove_edge(
    from_id: str,
    to_id: str,
    edge_type: str,
    *,
    data_dir=None,
) -> dict:
    """Soft-remove an edge (sets removed_ts, never deletes).

    Thin wrapper over :func:`taosmd.tasks.remove_edge`.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.task_remove_edge(from_id, to_id, edge_type)
    from . import tasks as _tasks  # noqa: PLC0415
    return await _tasks.remove_edge(
        from_id, to_id, edge_type, data_dir=data_dir
    )


# ---------------------------------------------------------------------------
# Admin surface service wrappers
# ---------------------------------------------------------------------------

async def admin_shelf_create(
    shelf_id: str,
    *,
    project_id: str | None = None,
    display_name: str | None = None,
    data_dir=None,
) -> dict:
    """Create or return an existing shelf. Returns ``{"shelf": {...}, "created": bool}``."""
    if data_dir is None:
        stores = await _api._ensure_stores(data_dir)
        data_dir = stores["data_dir"]
    from .admin import shelf_create  # noqa: PLC0415
    return await shelf_create(
        shelf_id, project_id=project_id, display_name=display_name, data_dir=data_dir,
    )


async def admin_shelf_archive(
    shelf_id: str,
    *,
    expect_empty: bool = False,
    data_dir=None,
) -> dict:
    """Archive a shelf, soft-hiding its vector rows."""
    stores = await _api._ensure_stores(data_dir)
    if data_dir is None:
        data_dir = stores["data_dir"]
    from .admin import shelf_archive  # noqa: PLC0415
    return await shelf_archive(
        shelf_id, expect_empty=expect_empty, data_dir=data_dir, stores=stores,
    )


async def admin_shelf_unarchive(
    shelf_id: str,
    *,
    data_dir=None,
) -> dict:
    """Unarchive a shelf, restoring only shelf-archive-hidden rows."""
    stores = await _api._ensure_stores(data_dir)
    if data_dir is None:
        data_dir = stores["data_dir"]
    from .admin import shelf_unarchive  # noqa: PLC0415
    return await shelf_unarchive(shelf_id, data_dir=data_dir, stores=stores)


async def admin_a2a_delete_channel(channel: str, *, data_dir=None) -> dict:
    """Soft-delete an A2A channel."""
    stores = await _api._ensure_stores(data_dir)
    if data_dir is None:
        data_dir = stores["data_dir"]
    from .admin import a2a_admin_delete_channel  # noqa: PLC0415
    return await a2a_admin_delete_channel(channel, data_dir=data_dir, stores=stores)


async def admin_a2a_rename_channel(
    from_channel: str,
    to_channel: str,
    *,
    data_dir=None,
) -> dict:
    """Rename an A2A channel via alias."""
    stores = await _api._ensure_stores(data_dir)
    if data_dir is None:
        data_dir = stores["data_dir"]
    from .admin import a2a_admin_rename_channel  # noqa: PLC0415
    return await a2a_admin_rename_channel(
        from_channel, to_channel, data_dir=data_dir, stores=stores
    )


async def admin_a2a_supersede_message(msg_id: int, *, data_dir=None) -> dict:
    """Supersede (hide) a single A2A message from feeds."""
    stores = await _api._ensure_stores(data_dir)
    if data_dir is None:
        data_dir = stores["data_dir"]
    from .admin import a2a_admin_supersede_message  # noqa: PLC0415
    return await a2a_admin_supersede_message(msg_id, data_dir=data_dir, stores=stores)


# ---------------------------------------------------------------------------
# Collections service wrappers
# ---------------------------------------------------------------------------
#
# Like the shelf admin wrappers these are local-only (no remote forwarding):
# the server that owns the filesystem being indexed is the server that runs
# the collection ops. All wrappers open the store against the resolved data
# dir and close it, so no sqlite connection outlives a call.

def _collection_store(data_dir):
    from .collections import CollectionStore  # noqa: PLC0415
    return CollectionStore(_api._resolve_data_dir(data_dir))


async def collections_create(
    *,
    name: str,
    kind: str,
    source_path: str,
    embedder: str | None = None,
    data_dir=None,
) -> dict:
    """Create a collection row (admin operation). Returns the collection."""
    store = _collection_store(data_dir)
    try:
        return store.create(
            name=name, kind=kind, source_path=source_path, embedder=embedder,
        )
    finally:
        store.close()


async def collections_list(*, project: str | None = None, data_dir=None) -> list[dict]:
    """List collections, optionally filtered to one project's links."""
    store = _collection_store(data_dir)
    try:
        return store.list(project=project)
    finally:
        store.close()


async def collections_get(collection_id: str, *, data_dir=None) -> dict:
    """Return one collection with full stats, links, and grants."""
    store = _collection_store(data_dir)
    try:
        return store.get(collection_id)
    finally:
        store.close()


async def collections_index_start(collection_id: str, *, data_dir=None) -> dict:
    """Validate and mark a collection ``indexing``; the walk runs separately.

    Raises ``CollectionNotFoundError`` (404) for an unknown id,
    ``CollectionBusyError`` (409) when an index is already running, and
    ``ValueError`` (400) for an archived collection or a source path that no
    longer resolves inside an allowed root, so callers get a synchronous
    error before the background job is spawned.
    """
    from .collections import CollectionBusyError  # noqa: PLC0415
    store = _collection_store(data_dir)
    try:
        col = store.get(collection_id)
        if col["status"] == "archived":
            raise ValueError(
                f"collection {collection_id!r} is archived; unarchive before indexing"
            )
        if col["status"] == "indexing":
            raise CollectionBusyError(
                f"collection {collection_id!r} is already indexing; "
                f"poll GET /collections/{collection_id} until it settles"
            )
        store.resolve_source_path(col["source_path"])
        store.set_status(collection_id, "indexing")
    finally:
        store.close()
    return {"status": "indexing", "job": collection_id}


async def collections_index_run(collection_id: str, *, data_dir=None) -> dict:
    """Run the folder walk + ingest for one collection (blocking variant)."""
    from .collections import ingest_folder  # noqa: PLC0415
    return await ingest_folder(collection_id, data_dir=data_dir)


async def collections_index_background(collection_id: str, *, data_dir=None) -> None:
    """Background wrapper for the HTTP 202 path: never raises, only logs.

    ``ingest_folder`` records failures on the collection row
    (status='error' + message) so pollers see them; this wrapper keeps the
    fire-and-forget future from warning about an unobserved exception.
    """
    try:
        await collections_index_run(collection_id, data_dir=data_dir)
    except Exception:  # noqa: BLE001 - surfaced via the collection row
        logger.exception("collections: background index failed for %s", collection_id)


async def collections_link(
    collection_id: str, link_type: str, ext_id: str, *, data_dir=None
) -> dict:
    """Attach a typed project link ({taos|git, id}). Metadata only."""
    store = _collection_store(data_dir)
    try:
        return store.link(collection_id, link_type, ext_id)
    finally:
        store.close()


async def collections_unlink(
    collection_id: str, link_type: str, ext_id: str, *, data_dir=None
) -> dict:
    """Remove a typed project link. Metadata only, content untouched."""
    store = _collection_store(data_dir)
    try:
        return store.unlink(collection_id, link_type, ext_id)
    finally:
        store.close()


async def collections_grant(collection_id: str, agent: str, *, data_dir=None) -> dict:
    """Grant ``agent`` query access to the collection."""
    store = _collection_store(data_dir)
    try:
        return store.grant(collection_id, agent)
    finally:
        store.close()


async def collections_revoke(collection_id: str, agent: str, *, data_dir=None) -> dict:
    """Revoke ``agent``'s query access."""
    store = _collection_store(data_dir)
    try:
        return store.revoke(collection_id, agent)
    finally:
        store.close()


async def collections_archive(collection_id: str, *, data_dir=None) -> dict:
    """Archive a collection (the DELETE alias). Reversible; nothing destroyed."""
    store = _collection_store(data_dir)
    try:
        return store.archive(collection_id)
    finally:
        store.close()


__all__ = ["ingest", "search", "pending_list", "pending_resolve", "reconcile", "stats",
           "supersede", "fetch_by_ref", "a2a_send", "a2a_feed", "a2a_channels", "a2a_sender_census",
           "a2a_members", "a2a_threads", "a2a_thread_messages",
           "a2a_mentions_feed", "a2a_migrate_kinds", "a2a_alarms_clear", "can_read",
           "a2a_inbox", "a2a_inbox_advance",
           "task_create", "task_list", "task_ready", "task_prime",
           "task_update", "task_add_edge", "task_remove_edge", "task_projects",
           "admin_shelf_create", "admin_shelf_archive", "admin_shelf_unarchive",
           "admin_a2a_delete_channel", "admin_a2a_rename_channel",
           "admin_a2a_supersede_message",
           "a2a_record_delivered", "a2a_record_seen", "a2a_get_receipts",
           "a2a_get_receipt", "a2a_prune_receipts", "a2a_ack",
           "collections_create", "collections_list", "collections_get",
           "collections_index_start", "collections_index_run",
           "collections_index_background", "collections_link",
           "collections_unlink", "collections_grant", "collections_revoke",
           "collections_archive"]
