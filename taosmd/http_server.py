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
``GET  /``                 -> the read-only inspection UI (``text/html``)
``GET  /ui``               -> alias of ``GET /``
``GET  /health``           -> ``{"status": "ok", "version": <str>}``
``POST /ingest``           ``{"text", "agent"}``           -> ingest result
``POST /search``           ``{"query", "agent", "limit"?}`` -> ``{"hits": [...]}``
``GET  /search?q=&agent=&limit=``                          -> ``{"hits": [...]}``
``GET  /pending?agent=``                                   -> ``{"pending": [...]}``
``POST /pending/resolve``  ``{"id", "decision", "note"?}`` -> resolve result

Inspection UI
-------------
``GET /`` (and ``GET /ui``) serves a single self-contained HTML page — one
inline ``<style>`` and one inline vanilla ``<script>``, no external requests
or CDNs — so it works fully offline. It is a **read-only** inspector: it lets
you search memory and view the pending-review queue for an agent and read the
server health/version, all by consuming the JSON endpoints above via ``fetch``.
It exposes no destructive actions (no ingest, no resolve). The JSON endpoints
remain the integration surface; the page is just a thin local viewer.
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


# A single self-contained page: one inline <style> and one inline vanilla
# <script>, no external requests, so it works fully offline. Read-only — it
# only calls the GET /health, POST /search and GET /pending endpoints and
# renders the results. No ingest/resolve/destructive actions are exposed.
_INSPECTION_UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>taOSmd inspector</title>
<style>
  :root {
    --bg: #0f1115; --panel: #181b22; --border: #2a2f3a; --text: #e6e8ec;
    --muted: #9aa3b2; --accent: #6ea8fe; --chip: #232834;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  header {
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  header .tag { color: var(--muted); font-size: 13px; }
  #health { margin-left: auto; color: var(--muted); font-size: 13px; }
  main { max-width: 860px; margin: 0 auto; padding: 20px; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; margin-bottom: 20px;
  }
  .card h2 { font-size: 15px; margin: 0 0 12px; font-weight: 600; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; }
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  input {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px; font: inherit; width: 100%;
  }
  input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .field { flex: 1 1 160px; }
  .field.grow { flex: 3 1 320px; }
  button {
    background: var(--accent); color: #0b1020; border: 0; border-radius: 6px;
    padding: 8px 16px; font: inherit; font-weight: 600; cursor: pointer;
    align-self: flex-end;
  }
  button:hover { filter: brightness(1.08); }
  button:disabled { opacity: .5; cursor: default; }
  .results { margin-top: 14px; }
  .hit {
    border: 1px solid var(--border); border-radius: 8px; padding: 12px;
    margin-bottom: 10px; background: var(--bg);
  }
  .hit .text { white-space: pre-wrap; word-break: break-word; }
  .meta { margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    background: var(--chip); color: var(--muted); font-size: 12px;
    padding: 2px 8px; border-radius: 999px;
  }
  .empty, .err { color: var(--muted); font-size: 14px; padding: 6px 0; }
  .err { color: #ff8c8c; }
  .note { color: var(--muted); font-size: 12px; margin-top: 6px; }
</style>
</head>
<body>
<header>
  <h1>taOSmd inspector</h1>
  <span class="tag">read-only local memory viewer</span>
  <span id="health">checking…</span>
</header>
<main>
  <section class="card" aria-labelledby="search-h">
    <h2 id="search-h">Search memory</h2>
    <div class="row">
      <div class="field">
        <label for="agent">Agent</label>
        <input id="agent" placeholder="agent name" autocomplete="off" value="default">
      </div>
      <div class="field grow">
        <label for="query">Query</label>
        <input id="query" placeholder="what do you want to recall?" autocomplete="off">
      </div>
      <button id="searchBtn" type="button">Search</button>
    </div>
    <div class="results" id="searchResults"></div>
  </section>

  <section class="card" aria-labelledby="pending-h">
    <h2 id="pending-h">Pending review queue</h2>
    <div class="row">
      <button id="pendingBtn" type="button">Load pending decisions</button>
    </div>
    <p class="note">Read-only. Resolve decisions from the CLI (<code>taosmd review</code>).</p>
    <div class="results" id="pendingResults"></div>
  </section>
</main>
<script>
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function chip(label, value) {
    if (value === undefined || value === null || value === "") return "";
    return '<span class="chip">' + esc(label) + ": " + esc(value) + "</span>";
  }

  function loadHealth() {
    fetch("/health").then(function (r) { return r.json(); }).then(function (d) {
      $("health").textContent = "ok · v" + (d.version || "?");
    }).catch(function () {
      $("health").textContent = "unreachable";
    });
  }

  function search() {
    var agent = $("agent").value.trim();
    var query = $("query").value.trim();
    var out = $("searchResults");
    if (!agent || !query) {
      out.innerHTML = '<div class="err">Enter both an agent and a query.</div>';
      return;
    }
    out.innerHTML = '<div class="empty">Searching…</div>';
    fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent: agent, query: query, limit: 10 })
    }).then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, body: d }; });
    }).then(function (res) {
      if (!res.ok) {
        out.innerHTML = '<div class="err">' + esc(res.body.error || "search failed") + "</div>";
        return;
      }
      var hits = res.body.hits || [];
      if (!hits.length) { out.innerHTML = '<div class="empty">No matches.</div>'; return; }
      out.innerHTML = hits.map(function (h) {
        return '<div class="hit"><div class="text">' + esc(h.text) + "</div>" +
          '<div class="meta">' +
          chip("source", h.source) +
          chip("confidence", typeof h.confidence === "number" ? h.confidence.toFixed(3) : h.confidence) +
          chip("when", h.timestamp) +
          "</div></div>";
      }).join("");
    }).catch(function (e) {
      out.innerHTML = '<div class="err">' + esc(e.message || e) + "</div>";
    });
  }

  function loadPending() {
    var agent = $("agent").value.trim();
    var out = $("pendingResults");
    out.innerHTML = '<div class="empty">Loading…</div>';
    var url = "/pending" + (agent ? "?agent=" + encodeURIComponent(agent) : "");
    fetch(url).then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, body: d }; });
    }).then(function (res) {
      if (!res.ok) {
        out.innerHTML = '<div class="err">' + esc(res.body.error || "request failed") + "</div>";
        return;
      }
      var rows = res.body.pending || [];
      if (!rows.length) { out.innerHTML = '<div class="empty">Nothing pending.</div>'; return; }
      out.innerHTML = rows.map(function (p) {
        var triple = [p.subject, p.predicate, p.object].filter(Boolean).join(" → ");
        return '<div class="hit"><div class="text">' + esc(triple || p.id) + "</div>" +
          '<div class="meta">' +
          chip("kind", p.kind) +
          chip("id", p.id) +
          "</div></div>";
      }).join("");
    }).catch(function (e) {
      out.innerHTML = '<div class="err">' + esc(e.message || e) + "</div>";
    });
  }

  $("searchBtn").addEventListener("click", search);
  $("query").addEventListener("keydown", function (e) { if (e.key === "Enter") search(); });
  $("pendingBtn").addEventListener("click", loadPending);
  loadHealth();
})();
</script>
</body>
</html>
"""


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

        def _send_html(self, status: int, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
                if method == "GET" and path in ("/", "/ui"):
                    self._send_html(200, _INSPECTION_UI_HTML)
                elif method == "GET" and path == "/health":
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
    print(f"Inspection UI (read-only): http://{bound_host}:{bound_port}/")
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
