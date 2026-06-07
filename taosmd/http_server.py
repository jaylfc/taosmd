"""Local HTTP/REST API for taOSmd memory (stdlib only, zero deps).

An activation surface so non-Python apps and remote-on-LAN agents can read
and write taOSmd memory without embedding the Python package — the spirit of
``qmd serve``. It is a thin JSON shell over :mod:`taosmd.service`, the same
shared core the upcoming MCP server (#84) sits on, so behaviour matches the
Python API and CLI exactly.

Design choices (matching the project's local-first, offline, additive vision):

* **stdlib only** — :class:`http.server.ThreadingHTTPServer` +
  :class:`~http.server.BaseHTTPRequestHandler`, :mod:`json`, no new deps.
* **local-first** — binds ``127.0.0.1`` by default. Pass a different host
  (e.g. ``0.0.0.0``) only to expose it on the LAN. There is **no auth**: on
  localhost that is fine (any local process already has the Python API); if
  you bind to a routable address, put it behind your own network controls.
* **additive + opt-in** — the server only runs when you start it; the
  Python API, CLI, and standalone use are untouched.
* **per-agent scoping** — every endpoint takes an ``agent`` and forwards it
  to the service layer, honouring the same isolation as the Python API.

Concurrency model
-----------------
:class:`ThreadingHTTPServer` hands each request to its own thread, but the
underlying stores hold thread-affine SQLite connections (created and cached
on first use). So rather than ``asyncio.run`` per request — which would bind
those connections to a request thread that later disappears — every async
service call is dispatched onto a single, long-lived background event-loop
thread (see :class:`_ServiceLoop`). All DB work therefore happens in one
context, sequentially, exactly like the single-threaded Python API. The
result is marshalled back to the calling request thread.

Endpoints
---------
``GET  /health``           -> ``{"status": "ok", "version": <str>}``
``POST /ingest``           ``{"text", "agent"}``           -> ingest result
``POST /search``           ``{"query", "agent", "limit"?}`` -> ``{"hits": [...]}``
``GET  /search?q=&agent=&limit=``                          -> ``{"hits": [...]}``
``GET  /pending?agent=``                                   -> ``{"pending": [...]}``
``POST /pending/resolve``  ``{"id", "decision", "note"?}`` -> resolve result
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import __version__, service

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7833


class _BadRequest(Exception):
    """Raised for malformed input -> 400."""


class _ServiceLoop:
    """A single background thread running one asyncio event loop.

    All store/DB work runs here so the thread-affine SQLite connections are
    created and used in exactly one thread, no matter which request thread
    issued the call. ``run(coro)`` blocks the caller until the coroutine
    completes and returns its result (or re-raises its exception).
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="taosmd-service-loop", daemon=True,
        )
        self._thread.start()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


def _make_handler(data_dir, runner: _ServiceLoop):
    """Build a handler class bound to a fixed ``data_dir``.

    ThreadingHTTPServer instantiates the handler per request, so the data dir
    is closed over here rather than threaded through every call site.
    """

    class TaosmdHandler(BaseHTTPRequestHandler):
        server_version = f"taosmd/{__version__}"

        # ----- plumbing ----------------------------------------------------
        def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise _BadRequest(f"invalid JSON body: {exc}") from exc
            if not isinstance(parsed, dict):
                raise _BadRequest("JSON body must be an object")
            return parsed

        # ----- routing -----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - stdlib signature
            self._dispatch("GET")

        def do_HEAD(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def _dispatch(self, method: str) -> None:
            parts = urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            query = parse_qs(parts.query)
            try:
                if method == "GET" and path == "/health":
                    self._send_json(200, {"status": "ok", "version": __version__})
                elif method == "GET" and path == "/search":
                    self._handle_search_get(query)
                elif method == "POST" and path == "/search":
                    self._handle_search_post()
                elif method == "POST" and path == "/ingest":
                    self._handle_ingest()
                elif method == "GET" and path == "/pending":
                    self._handle_pending(query)
                elif method == "POST" and path == "/pending/resolve":
                    self._handle_pending_resolve()
                else:
                    self._send_json(404, {"error": f"unknown route: {method} {path}"})
            except _BadRequest as exc:
                self._send_json(400, {"error": str(exc)})
            except ValueError as exc:
                # Service-layer validation (e.g. missing agent, bad action).
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - surface as 500 JSON
                logger.exception("taosmd http: unhandled error for %s %s", method, path)
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        # ----- handlers ----------------------------------------------------
        def _handle_ingest(self) -> None:
            body = self._read_json_body()
            text = body.get("text")
            agent = body.get("agent")
            if not isinstance(text, str) or not text:
                raise _BadRequest("'text' (non-empty string) is required")
            if not isinstance(agent, str) or not agent:
                raise _BadRequest("'agent' (non-empty string) is required")
            result = runner.run(service.ingest(text, agent=agent, data_dir=data_dir))
            self._send_json(200, result)

        def _handle_search_post(self) -> None:
            body = self._read_json_body()
            query = body.get("query")
            agent = body.get("agent")
            limit = body.get("limit", 5)
            self._do_search(query, agent, limit)

        def _handle_search_get(self, qs: dict) -> None:
            query = (qs.get("q") or qs.get("query") or [None])[0]
            agent = (qs.get("agent") or [None])[0]
            limit = (qs.get("limit") or [5])[0]
            self._do_search(query, agent, limit)

        def _do_search(self, query, agent, limit) -> None:
            if not isinstance(query, str) or not query:
                raise _BadRequest("'query' (non-empty string) is required")
            if not isinstance(agent, str) or not agent:
                raise _BadRequest("'agent' (non-empty string) is required")
            try:
                limit_i = int(limit)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            hits = runner.run(
                service.search(query, agent=agent, data_dir=data_dir, limit=limit_i)
            )
            self._send_json(200, {"hits": hits})

        def _handle_pending(self, qs: dict) -> None:
            agent = (qs.get("agent") or [None])[0]
            limit = (qs.get("limit") or [20])[0]
            try:
                limit_i = int(limit)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            pending = runner.run(
                service.pending_list(agent=agent, data_dir=data_dir, limit=limit_i)
            )
            self._send_json(200, {"pending": pending})

        def _handle_pending_resolve(self) -> None:
            body = self._read_json_body()
            decision_id = body.get("id")
            decision = body.get("decision")
            note = body.get("note", "")
            if not isinstance(decision_id, str) or not decision_id:
                raise _BadRequest("'id' (non-empty string) is required")
            if decision not in {"accept", "reject", "modify"}:
                raise _BadRequest("'decision' must be one of accept|reject|modify")
            result = runner.run(
                service.pending_resolve(
                    decision_id, decision, note=note or "", data_dir=data_dir,
                )
            )
            self._send_json(200, result)

    return TaosmdHandler


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, data_dir=None):
    """Create (but do not start) a :class:`ThreadingHTTPServer`.

    Useful for tests that need to bind an ephemeral port (``port=0``) and read
    back the assigned address before serving. Call ``server.serve_forever()``
    to run it, ``server.shutdown()`` + ``server.server_close()`` to stop.

    A :class:`_ServiceLoop` is started and attached as ``server.service_loop``;
    closing it is handled by :func:`serve`. Tests that drive ``make_server``
    directly should call ``server.service_loop.close()`` during teardown.
    """
    runner = _ServiceLoop()
    httpd = ThreadingHTTPServer((host, port), _make_handler(data_dir, runner))
    httpd.service_loop = runner
    return httpd


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, data_dir=None) -> int:
    """Run the local HTTP memory server until interrupted.

    Binds ``host:port`` (default ``127.0.0.1:7833``) and blocks serving
    requests. Returns 0 on a clean Ctrl-C shutdown.
    """
    httpd = make_server(host, port, data_dir)
    bound_host, bound_port = httpd.server_address[:2]
    where = "localhost only" if bound_host in {"127.0.0.1", "::1"} else "LAN-reachable (no auth)"
    print(f"taosmd HTTP API listening on http://{bound_host}:{bound_port} ({where})")
    print("Endpoints: GET /health, POST /ingest, GET|POST /search, "
          "GET /pending, POST /pending/resolve")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        httpd.shutdown()
        httpd.server_close()
        httpd.service_loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
