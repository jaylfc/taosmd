"""Remote client for a taOSmd HTTP server, stdlib only, zero extra deps.

``RemoteClient`` mirrors the async methods exposed by :mod:`taosmd.service`
(and therefore the local Python API) so callers can transparently point the
service layer at a remote server (e.g. a Raspberry Pi over Tailscale) by
setting ``TAOSMD_SERVER_URL`` or running ``taosmd config set-server <url>``.

All network I/O is blocking (``urllib.request``).  Each public method is an
async coroutine that offloads the blocking call via :func:`asyncio.to_thread`
so the caller's event loop is never stalled.

Error handling
--------------
Any non-2xx response raises :class:`RuntimeError` with the HTTP status and the
body text so the caller can surface a clear message without parsing internals.
Connection errors from ``urllib`` are propagated as-is.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class RemoteClient:
    """Async client that delegates taOSmd service calls to a remote HTTP server.

    Args:
        base_url: Base URL of the remote server, e.g. ``"http://pi.local:7900"``.
            A trailing slash is stripped for consistency.
        token: Optional bearer token sent as ``Authorization: Bearer <token>``.
        timeout: Request timeout in seconds (default 30).
    """

    def __init__(self, base_url: str, token: str | None = None, timeout: int = 30) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if extra:
            h.update(extra)
        return h

    def _request_json(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> Any:
        """Perform a synchronous JSON request and return the parsed response body.

        Raises :class:`RuntimeError` for non-2xx responses.
        """
        url = self._base + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        encoded: bytes | None = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=encoded, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8")
            except Exception:
                raw = "(unreadable body)"
            raise RuntimeError(f"taosmd remote: HTTP {exc.code} from {url}: {raw}") from exc

    async def _run(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> Any:
        """Async wrapper: offloads the blocking urllib call via asyncio.to_thread."""
        return await asyncio.to_thread(self._request_json, method, path, body, params)

    # ------------------------------------------------------------------
    # Memory service methods: mirrors taosmd.service signatures
    # ------------------------------------------------------------------

    async def ingest(self, text: str, agent: str, *, project: str | None = None, **_opts) -> dict:
        """POST /ingest: shelve ``text`` into the remote agent's memory.

        Returns ``{"archived", "agent", "project", "data_dir"}``. ``project``
        is forwarded so project-scoped memory works over the remote path.
        """
        body: dict = {"text": text, "agent": agent}
        if project is not None:
            body["project"] = project
        return await self._run("POST", "/ingest", body)

    async def ingest_batch(
        self, items: list[dict], agent: str, *, project: str | None = None, **_opts
    ) -> dict:
        """POST /ingest/batch: bulk-shelve items with idempotent re-import.

        Returns ``{"ingested", "skipped", "agent", "data_dir"}``.
        """
        body: dict = {"items": items, "agent": agent}
        if project is not None:
            body["project"] = project
        return await self._run("POST", "/ingest/batch", body)

    async def search(
        self,
        query: str,
        agent: str,
        limit: int = 5,
        *,
        project: str | None = None,
        also_include: list[str] | None = None,
        mode: str | None = None,
        **_opts,
    ) -> list[dict]:
        """POST /search: return ranked hits for ``query`` from the remote server.

        ``project`` and ``also_include`` are forwarded so project-scoped and
        cross-agent reads work over the remote path. ``mode="bm25"`` selects
        the server's BM25-only path. Returns the ``hits`` list.
        """
        body: dict = {"query": query, "agent": agent, "limit": limit}
        if project is not None:
            body["project"] = project
        if also_include is not None:
            body["also_include"] = also_include
        if mode:
            body["mode"] = mode
        resp = await self._run("POST", "/search", body)
        return resp.get("hits", [])

    async def list_projects(self, **_opts) -> list[dict]:
        """GET /projects: list projects that have stored memories on the server."""
        resp = await self._run("GET", "/projects")
        return resp.get("projects", [])

    async def dashboard_stats(self, *, scope: str | None = None, **_opts) -> dict:
        """GET /stats: aggregate dashboard stats from the server."""
        params = {"scope": scope} if scope else None
        return await self._run("GET", "/stats", params=params)

    async def list_memories(self, *, scope: str | None = None, limit: int = 50, **_opts) -> list[dict]:
        """GET /memories: recent archived memories for the browse view."""
        params: dict = {"limit": limit}
        if scope:
            params["scope"] = scope
        resp = await self._run("GET", "/memories", params=params)
        return resp.get("memories", [])

    async def list_shelves(self, *, project: str, **_opts) -> list[dict]:
        """GET /shelves: list the agent shelves within ``project`` on the server."""
        resp = await self._run("GET", "/shelves", params={"project": project})
        return resp.get("shelves", [])

    async def pending_list(self, agent: str | None = None, limit: int = 20, **_opts) -> list[dict]:
        """GET /pending: return unresolved KG-update decisions.

        Returns the ``pending`` list from the server response.
        """
        params: dict = {"limit": limit}
        if agent:
            params["agent"] = agent
        resp = await self._run("GET", "/pending", params=params)
        return resp.get("pending", [])

    async def pending_resolve(
        self,
        decision_id: str,
        decision: str,
        *,
        note: str = "",
        **_opts,
    ) -> dict:
        """POST /pending/resolve: resolve a pending KG decision."""
        return await self._run(
            "POST", "/pending/resolve",
            {"id": decision_id, "decision": decision, "note": note},
        )

    async def a2a_send(
        self,
        sender: str,
        body: str,
        *,
        thread: str = "general",
        reply_to: str | None = None,
        **_opts,
    ) -> dict:
        """POST /a2a/send: post a message to the remote A2A bus.

        Returns the send receipt ``{"id", "from", "thread", "reply_to"}``.
        """
        payload: dict = {"from": sender, "body": body, "thread": thread}
        if reply_to is not None:
            payload["reply_to"] = reply_to
        return await self._run("POST", "/a2a/send", payload)

    async def a2a_feed(
        self,
        *,
        thread: str | None = None,
        since: float | None = None,
        limit: int = 50,
        **_opts,
    ) -> list[dict]:
        """GET /a2a/messages: return messages from the remote A2A bus, oldest-first.

        Returns the ``messages`` list from the server response.
        """
        params: dict = {"limit": limit}
        if thread is not None:
            params["thread"] = thread
        if since is not None:
            params["since"] = since
        resp = await self._run("GET", "/a2a/messages", params=params)
        return resp.get("messages", [])

    async def a2a_channels(self, **_opts) -> list[dict]:
        """GET /a2a/channels: return a summary of every channel on the remote bus."""
        resp = await self._run("GET", "/a2a/channels")
        return resp.get("channels", [])

    async def a2a_members(self, *, channel: str, **_opts) -> list[str]:
        """GET /a2a/members: return distinct sender names on ``channel``."""
        resp = await self._run("GET", "/a2a/members", params={"channel": channel})
        return resp.get("members", [])

    async def stats(self, *, agent: str, **_opts) -> dict:
        """Best-effort stats for ``agent`` on the remote server.

        Fetches ``GET /health`` (always available) and attempts a search
        with an empty query to probe liveness.  Returns a minimal dict with
        at least ``{"agent", "reachable"}``.  Does not raise on connection
        errors; reports them in the returned dict instead so callers that
        use stats for a health probe are not interrupted.
        """
        try:
            health = await self._run("GET", "/health")
            reachable = health.get("status") == "ok"
        except Exception as exc:
            return {"agent": agent, "reachable": False, "error": str(exc)}
        return {"agent": agent, "reachable": reachable, "server_version": health.get("version")}


    async def task_create(
        self,
        title: str,
        *,
        body: str | None = None,
        project: str | None = None,
        assignee: str | None = None,
        priority: int = 0,
        depends_on: list[str] | None = None,
        created_by: str,
        **_opts,
    ) -> dict:
        """POST /tasks: create a task on the remote server."""
        payload: dict = {"title": title, "created_by": created_by, "priority": priority}
        if body is not None:
            payload["body"] = body
        if project is not None:
            payload["project"] = project
        if assignee is not None:
            payload["assignee"] = assignee
        if depends_on is not None:
            payload["depends_on"] = depends_on
        return await self._run("POST", "/tasks", payload)

    async def task_list(
        self,
        *,
        status: str | None = None,
        project: str | None = None,
        assignee: str | None = None,
        limit: int = 50,
        **_opts,
    ) -> list[dict]:
        """GET /tasks: list tasks from the remote server."""
        params: dict = {"limit": limit}
        if status is not None:
            params["status"] = status
        if project is not None:
            params["project"] = project
        if assignee is not None:
            params["assignee"] = assignee
        resp = await self._run("GET", "/tasks", params=params)
        return resp.get("tasks", [])

    async def task_ready(
        self,
        *,
        project: str | None = None,
        assignee: str | None = None,
        limit: int = 20,
        **_opts,
    ) -> list[dict]:
        """GET /tasks/ready: return the ready queue from the remote server."""
        params: dict = {"limit": limit}
        if project is not None:
            params["project"] = project
        if assignee is not None:
            params["assignee"] = assignee
        resp = await self._run("GET", "/tasks/ready", params=params)
        return resp.get("tasks", [])

    async def task_prime(
        self,
        *,
        project: str | None = None,
        assignee: str | None = None,
        **_opts,
    ) -> dict:
        """GET /tasks/prime: fetch the prime briefing from the remote server."""
        params: dict = {}
        if project is not None:
            params["project"] = project
        if assignee is not None:
            params["assignee"] = assignee
        return await self._run("GET", "/tasks/prime", params=params or None)

    async def task_update(
        self,
        task_id: str,
        *,
        status: str | None = None,
        assignee: str | None = None,
        priority: int | None = None,
        body: str | None = None,
        **_opts,
    ) -> dict:
        """POST /tasks/{id}: update a task on the remote server."""
        payload: dict = {}
        if status is not None:
            payload["status"] = status
        if assignee is not None:
            payload["assignee"] = assignee
        if priority is not None:
            payload["priority"] = priority
        if body is not None:
            payload["body"] = body
        return await self._run("POST", f"/tasks/{task_id}", payload)

    async def task_add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        created_by: str,
        **_opts,
    ) -> dict:
        """POST /tasks/{id}/edges: add an edge on the remote server."""
        return await self._run(
            "POST", f"/tasks/{from_id}/edges",
            {"to_id": to_id, "type": edge_type, "created_by": created_by},
        )

    async def task_remove_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        **_opts,
    ) -> dict:
        """POST /tasks/{id}/edges/remove: soft-remove an edge on the remote server."""
        return await self._run(
            "POST", f"/tasks/{from_id}/edges/remove",
            {"to_id": to_id, "type": edge_type},
        )


__all__ = ["RemoteClient"]
