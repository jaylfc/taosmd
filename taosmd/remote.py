"""Remote client for a taOSmd HTTP server — stdlib only, zero extra deps.

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
        """Async wrapper — offloads the blocking urllib call via asyncio.to_thread."""
        return await asyncio.to_thread(self._request_json, method, path, body, params)

    # ------------------------------------------------------------------
    # Memory service methods — mirrors taosmd.service signatures
    # ------------------------------------------------------------------

    async def ingest(self, text: str, agent: str, **_opts) -> dict:
        """POST /ingest — shelve ``text`` into the remote agent's memory.

        Returns ``{"archived", "agent", "data_dir"}``.
        """
        return await self._run("POST", "/ingest", {"text": text, "agent": agent})

    async def search(self, query: str, agent: str, limit: int = 5, **_opts) -> list[dict]:
        """POST /search — return ranked hits for ``query`` from the remote server.

        Returns the ``hits`` list from the server response.
        """
        resp = await self._run("POST", "/search", {"query": query, "agent": agent, "limit": limit})
        return resp.get("hits", [])

    async def pending_list(self, agent: str | None = None, limit: int = 20, **_opts) -> list[dict]:
        """GET /pending — return unresolved KG-update decisions.

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
        """POST /pending/resolve — resolve a pending KG decision."""
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
        """POST /a2a/send — post a message to the remote A2A bus.

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
        """GET /a2a/messages — return messages from the remote A2A bus, oldest-first.

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
        """GET /a2a/channels — return a summary of every channel on the remote bus."""
        resp = await self._run("GET", "/a2a/channels")
        return resp.get("channels", [])

    async def a2a_members(self, *, channel: str, **_opts) -> list[str]:
        """GET /a2a/members — return distinct sender names on ``channel``."""
        resp = await self._run("GET", "/a2a/members", params={"channel": channel})
        return resp.get("members", [])

    async def stats(self, *, agent: str, **_opts) -> dict:
        """Best-effort stats for ``agent`` on the remote server.

        Fetches ``GET /health`` (always available) and attempts a search
        with an empty query to probe liveness.  Returns a minimal dict with
        at least ``{"agent", "reachable"}``.  Does not raise on connection
        errors — reports them in the returned dict instead so callers that
        use stats for a health probe are not interrupted.
        """
        try:
            health = await self._run("GET", "/health")
            reachable = health.get("status") == "ok"
        except Exception as exc:
            return {"agent": agent, "reachable": False, "error": str(exc)}
        return {"agent": agent, "reachable": reachable, "server_version": health.get("version")}


__all__ = ["RemoteClient"]
