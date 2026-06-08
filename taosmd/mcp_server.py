"""MCP server for taOSmd memory and A2A bus over stdio (#84).

An activation surface so any MCP-capable agent (Claude, Cursor, Codex,
OpenWebUI, …) can read and write taOSmd memory and the agent-to-agent bus
directly, without a custom integration. It is a thin shell over
:mod:`taosmd.service` — the same shared core the local HTTP/REST server (#85)
sits on — so behaviour matches the Python API and CLI exactly.

Design choices (matching the project's local-first, offline, additive vision):

* **optional dependency** — the ``mcp`` Python SDK is imported lazily/guarded
  here, so ``import taosmd`` (and standalone use) never fails when ``mcp`` is
  not installed. Install with ``pip install taosmd[mcp]``.
* **local-first / offline** — the server speaks the stdio transport (the
  standard for desktop MCP clients); there is no network listener and no
  cloud dependency.
* **additive + opt-in** — the server only runs when you start it via
  ``taosmd mcp`` (or :func:`serve_stdio`); the Python API, CLI, and HTTP
  server are untouched.
* **per-agent scoping** — every tool takes an ``agent`` argument and forwards
  it to the service layer, honouring the same isolation as the Python API.

Concurrency model
-----------------
FastMCP runs each tool call on its own asyncio event loop, but the underlying
stores hold thread-affine SQLite connections (created and cached on first
use). So rather than running service coroutines directly on FastMCP's loop —
which would bind those connections to whichever loop happens to be live — every
async service call is dispatched onto a single, long-lived background
event-loop thread (the same :class:`taosmd.http_server._ServiceLoop` the HTTP
server uses). All DB work therefore happens in one context, sequentially,
exactly like the single-threaded Python API. The tools ``await`` the
cross-thread result so FastMCP's loop is never blocked.
"""

from __future__ import annotations

from . import service
from .http_server import _ServiceLoop


class MissingMCPDependencyError(RuntimeError):
    """Raised when the optional ``mcp`` SDK is not installed.

    The MCP server is an optional surface; the core package installs without
    it. Install the extra with ``pip install taosmd[mcp]``.
    """

    def __init__(self) -> None:
        super().__init__(
            "the MCP server requires the optional 'mcp' SDK. "
            "Install it with: pip install taosmd[mcp]"
        )


def _require_fastmcp():
    """Import :class:`FastMCP` lazily, with a clear error if mcp is absent."""
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised when mcp absent
        raise MissingMCPDependencyError() from exc
    return FastMCP


def build_server(data_dir=None, *, runner: _ServiceLoop | None = None):
    """Build (but do not run) a FastMCP server bound to ``data_dir``.

    Registers the memory tools and A2A bus tools, each routing to
    :mod:`taosmd.service` on the shared service-loop thread. Returns the
    configured ``FastMCP`` instance
    (its ``_taosmd_service_loop`` attribute holds the :class:`_ServiceLoop` so
    callers/tests can close it on teardown). ``runner`` lets a caller supply an
    existing loop (e.g. tests); otherwise one is created here.

    Raises :class:`MissingMCPDependencyError` if the ``mcp`` SDK is not
    installed.
    """
    FastMCP = _require_fastmcp()

    loop = runner if runner is not None else _ServiceLoop()
    mcp = FastMCP("taosmd")
    mcp._taosmd_service_loop = loop  # for teardown / introspection

    async def _dispatch(coro):
        """Run a service coroutine on the shared loop without blocking FastMCP."""
        import asyncio  # noqa: PLC0415

        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, loop._loop)
        )

    @mcp.tool()
    async def memory_ingest(text: str, agent: str, project: str | None = None) -> dict:
        """Store a transcript or note in an agent's long-term memory.

        Shelves ``text`` and embeds it so it can be retrieved later with
        ``memory_search``. ``agent`` scopes the write to one agent's memory.
        Optional ``project`` tags the memory with a project id so agents
        working on the same project can share it (see ``memory_search``).
        Returns ``{"archived", "agent", "project", "data_dir"}``.
        """
        opts = {"project": project} if project else {}
        return await _dispatch(service.ingest(text, agent=agent, data_dir=data_dir, **opts))

    @mcp.tool()
    async def memory_search(
        query: str,
        agent: str,
        limit: int = 5,
        project: str | None = None,
        also_include: list[str] | None = None,
    ) -> list[dict]:
        """Search an agent's memory for passages relevant to ``query``.

        Returns up to ``limit`` ranked hits, each with
        ``text``/``source``/``timestamp``/``confidence``/``metadata``. Scoped
        to ``agent``. Optional ``project`` scopes the search to one project;
        ``also_include`` (a list of agent names, only honoured with ``project``)
        adds those agents' memories within the project (cross-agent reads).
        """
        opts: dict = {}
        if project:
            opts["project"] = project
        if also_include:
            opts["also_include"] = also_include
        return await _dispatch(
            service.search(query, agent=agent, data_dir=data_dir, limit=limit, **opts)
        )

    @mcp.tool()
    async def memory_list_projects() -> list[dict]:
        """List projects that have stored memories.

        Returns ``[{"project_id", "agents", "last_ingest"}]``. Useful for
        discovering which project an agent shares with others before using
        ``project`` / ``also_include`` on ``memory_search``.
        """
        return await _dispatch(service.list_projects(data_dir=data_dir))

    @mcp.tool()
    async def memory_list_shelves(project: str) -> list[dict]:
        """List the agent shelves that have memories within ``project``.

        Returns ``[{"agent", "facts", "last_ingest"}]`` for the given project.
        """
        return await _dispatch(service.list_shelves(project=project, data_dir=data_dir))

    @mcp.tool()
    async def memory_pending_list(agent: str) -> list[dict]:
        """List unresolved knowledge-graph update decisions awaiting review.

        These are KG updates the librarian deferred for a human (or agent) to
        accept/reject. The queue is per install; ``agent`` is accepted for
        symmetry with the other tools.
        """
        return await _dispatch(service.pending_list(agent=agent, data_dir=data_dir))

    @mcp.tool()
    async def memory_pending_resolve(
        decision_id: str, decision: str, note: str = ""
    ) -> dict:
        """Resolve a pending decision with an explicit choice.

        ``decision`` is one of ``accept`` / ``reject`` / ``modify``. ``note``
        is an optional free-text justification. Returns
        ``{"ok", "applied_kg", "resolution"}``.
        """
        return await _dispatch(
            service.pending_resolve(
                decision_id, decision, note=note, data_dir=data_dir
            )
        )

    @mcp.tool()
    async def memory_stats(agent: str) -> dict:
        """Report lightweight stats for an agent's memory.

        Returns ``{"agent", "data_dir", "registered", "created_at",
        "last_ingest_at", "total_chunks"}``. Unknown agents report
        ``registered=False`` with zeroed counters rather than erroring.
        """
        return await _dispatch(service.stats(agent=agent, data_dir=data_dir))

    # ---- A2A bus tools -------------------------------------------------------

    @mcp.tool()
    async def a2a_channels() -> list[dict]:
        """List all channels (named threads) on the A2A bus."""
        return await _dispatch(service.a2a_channels(data_dir=data_dir))

    @mcp.tool()
    async def a2a_members(channel: str) -> list[str]:
        """List distinct sender names observed on ``channel``."""
        return await _dispatch(service.a2a_members(channel=channel, data_dir=data_dir))

    @mcp.tool()
    async def a2a_send(
        channel: str, sender: str, body: str, reply_to: str | None = None
    ) -> dict:
        """Post a message to ``channel`` from ``sender``."""
        return await _dispatch(
            service.a2a_send(sender, body, thread=channel, reply_to=reply_to, data_dir=data_dir)
        )

    @mcp.tool()
    async def a2a_read(
        channel: str, since: float | None = None, limit: int = 50
    ) -> list[dict]:
        """Read messages from ``channel``, oldest-first, optionally filtered by ``since``."""
        return await _dispatch(
            service.a2a_feed(thread=channel, since=since, limit=limit, data_dir=data_dir)
        )

    @mcp.tool()
    async def a2a_join(channel: str, agent: str) -> dict:
        """Announce ``agent``'s presence on ``channel`` with a join marker."""
        return await _dispatch(
            service.a2a_send(
                agent,
                f"[JOIN] {agent} joined the channel",
                thread=channel,
                data_dir=data_dir,
            )
        )

    return mcp


def serve_stdio(data_dir=None) -> int:
    """Run the taOSmd MCP server over the stdio transport until the client exits.

    This is the entry point for desktop MCP clients (Claude, Cursor, Codex):
    they spawn ``taosmd mcp`` and speak MCP over stdin/stdout. Returns 0 on a
    clean shutdown.

    Raises :class:`MissingMCPDependencyError` if the ``mcp`` SDK is not
    installed (the CLI turns this into a friendly message + non-zero exit).
    """
    mcp = build_server(data_dir)
    try:
        mcp.run(transport="stdio")
    finally:
        mcp._taosmd_service_loop.close()
    return 0


__all__ = ["build_server", "serve_stdio", "MissingMCPDependencyError"]
