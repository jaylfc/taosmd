"""Local HTTP/REST API for taOSmd memory (stdlib only, zero deps).

An activation surface so non-Python apps and remote-on-LAN agents can read
and write taOSmd memory without embedding the Python package, in the spirit of
``qmd serve``. It is a thin JSON shell over :mod:`taosmd.service`, the same
shared core the upcoming MCP server (#84) sits on, so behaviour matches the
Python API and CLI exactly.

Design choices (matching the project's local-first, offline, additive vision):

* **stdlib only**: :class:`http.server.ThreadingHTTPServer` +
  :class:`~http.server.BaseHTTPRequestHandler`, :mod:`json`, no new deps.
* **local-first**: binds ``127.0.0.1`` by default. Pass a different host
  (e.g. ``0.0.0.0``) only to expose it on the LAN. There is **no auth**: on
  localhost that is fine (any local process already has the Python API); if
  you bind to a routable address, put it behind your own network controls.
* **additive + opt-in**: the server only runs when you start it; the
  Python API, CLI, and standalone use are untouched.
* **per-agent scoping**: every endpoint takes an ``agent`` and forwards it
  to the service layer, honouring the same isolation as the Python API.

Concurrency model
-----------------
:class:`ThreadingHTTPServer` hands each request to its own thread, but the
underlying stores hold thread-affine SQLite connections (created and cached
on first use). So rather than ``asyncio.run`` per request, which would bind
those connections to a request thread that later disappears, every async
service call is dispatched onto a single, long-lived background event-loop
thread (see :class:`_ServiceLoop`). All DB work therefore happens in one
context, sequentially, exactly like the single-threaded Python API. The
result is marshalled back to the calling request thread.

Endpoints
---------
``GET  /``                 -> the read-only inspection UI (``text/html``)
``GET  /ui``               -> alias of ``GET /``
``GET  /health``           -> ``{"status": "ok", "version": <str>}``
``POST /ingest``           ``{"text", "agent", "project"?}`` -> ingest result
``POST /search``           ``{"query", "agent", "limit"?, "project"?, "also_include"?}`` -> ``{"hits": [...]}``
``GET  /search?q=&agent=&limit=&project=&also_include=a,b``  -> ``{"hits": [...]}``
``GET  /projects``                                         -> ``{"projects": [...]}``
``GET  /shelves?project=``                                 -> ``{"shelves": [...]}``
``GET  /pending?agent=``                                   -> ``{"pending": [...]}``
``POST /pending/resolve``  ``{"id", "decision", "note"?}`` -> resolve result
``POST /a2a/send``         ``{"from", "body", "thread"?, "reply_to"?}`` -> send receipt
``GET  /a2a/messages``     ``?thread=&since=&limit=``      -> ``{"messages": [...]}``
``GET  /a2a/stream``       ``?thread=&since=``             -> SSE stream (text/event-stream)
``GET  /a2a/channels``                                     -> ``{"channels": [...]}``
``GET  /a2a/members``      ``?channel=<name>``             -> ``{"members": [...]}``

Inspection UI
-------------
``GET /`` (and ``GET /ui``) serves a single self-contained HTML page: one
inline ``<style>`` and one inline vanilla ``<script>``, no external requests
or CDNs, so it works fully offline. It is a **read-only** inspector: it lets
you search memory and view the pending-review queue for an agent and read the
server health/version, all by consuming the JSON endpoints above via ``fetch``.
It exposes no destructive actions (no ingest, no resolve). The JSON endpoints
remain the integration surface; the page is just a thin local viewer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files as _pkg_files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__, config as _config, service

# ---------------------------------------------------------------------------
# Static webui helpers
# ---------------------------------------------------------------------------

def _webui_dir() -> Path | None:
    """Return the path to the built webui, or None if absent.

    Tries importlib.resources first (works in wheel installs); falls back to
    a path relative to this file (works in editable / source installs).
    """
    try:
        ref = _pkg_files("taosmd").joinpath("webui")
        # In Python 3.9+ files() returns a Traversable; we need a real Path.
        import importlib.resources as _ir  # noqa: PLC0415
        # For wheels, traverse to a concrete path via as_file context is
        # awkward to keep open; instead resolve via __file__ which always works.
        _ = ref  # suppress unused-variable on the import above
    except Exception:
        pass
    # Use __file__-relative path, reliable in both source and wheel.
    candidate = Path(__file__).parent / "webui"
    if (candidate / "index.html").exists():
        return candidate
    return None


_WEBUI_DIR: Path | None = _webui_dir()

# Extend mimetypes with types the stdlib may be missing.
_EXTRA_MIME = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".json": "application/json",
    ".map": "application/json",
}

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7900


# A single self-contained page: one inline <style> and one inline vanilla
# <script>, no external requests, so it works fully offline. Read-only: it
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
  /* A2A bus */
  .a2a-list {
    margin-top: 14px; max-height: 340px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 6px;
  }
  .a2a-msg {
    border-radius: 8px; padding: 8px 12px; max-width: 85%;
    background: var(--chip); word-break: break-word;
  }
  .a2a-msg.left  { align-self: flex-start; background: #1b2330; }
  .a2a-msg.right { align-self: flex-end;   background: #251c33; }
  .a2a-msg .sender { font-size: 12px; color: var(--accent); font-weight: 600; margin-bottom: 2px; }
  .a2a-msg.right .sender { color: #a78bfa; }
  .a2a-msg .body { white-space: pre-wrap; }
  .a2a-msg .ts { font-size: 11px; color: var(--muted); margin-top: 4px; text-align: right; }
  .a2a-status { color: var(--muted); font-size: 13px; margin-top: 8px; }
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

  <section class="card" aria-labelledby="a2a-h">
    <h2 id="a2a-h">A2A bus</h2>
    <div class="row">
      <div class="field">
        <label for="a2aThread">Thread</label>
        <input id="a2aThread" value="general" autocomplete="off">
      </div>
      <button id="a2aConnectBtn" type="button">Connect</button>
    </div>
    <p class="note">Read-only live view. Agents write via <code>POST /a2a/send</code>.</p>
    <div class="a2a-list" id="a2aList" aria-live="polite" aria-label="A2A messages"></div>
    <div class="a2a-status" id="a2aStatus"></div>
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

  // ----- A2A bus -----------------------------------------------------------
  var _a2aEs = null;

  function _a2aSenders() {
    // Collect unique senders from existing messages to decide left/right alignment.
    var items = $("a2aList").querySelectorAll(".a2a-msg");
    var seen = [];
    items.forEach(function (el) { var s = el.dataset.sender; if (s && seen.indexOf(s) === -1) seen.push(s); });
    return seen;
  }

  function _renderMsg(msg, senderIndex) {
    var side = (senderIndex % 2 === 0) ? "left" : "right";
    var ts = msg.ts ? new Date(msg.ts * 1000).toLocaleTimeString() : "";
    return '<div class="a2a-msg ' + side + '" data-sender="' + esc(msg.from || "") + '">' +
      '<div class="sender">' + esc(msg.from || "?") + '</div>' +
      '<div class="body">' + esc(msg.body || "") + '</div>' +
      '<div class="ts">' + esc(ts) + (msg.reply_to ? " · ↩ " + esc(String(msg.reply_to)) : "") + '</div>' +
      '</div>';
  }

  function _appendMsg(msg) {
    var list = $("a2aList");
    var senders = _a2aSenders();
    var idx = senders.indexOf(msg.from || "");
    if (idx === -1) { idx = senders.length; }
    list.insertAdjacentHTML("beforeend", _renderMsg(msg, idx));
    list.scrollTop = list.scrollHeight;
  }

  function a2aConnect() {
    var thread = $("a2aThread").value.trim() || "general";
    var list = $("a2aList");
    var status = $("a2aStatus");

    if (_a2aEs) { _a2aEs.close(); _a2aEs = null; }
    list.innerHTML = "";
    status.textContent = "Loading history…";

    fetch("/a2a/messages?thread=" + encodeURIComponent(thread) + "&limit=200")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var msgs = d.messages || [];
        var senderMap = {};
        var senderCount = 0;
        msgs.forEach(function (msg) {
          var s = msg.from || "";
          if (!(s in senderMap)) { senderMap[s] = senderCount++; }
          list.insertAdjacentHTML("beforeend", _renderMsg(msg, senderMap[s]));
        });
        list.scrollTop = list.scrollHeight;
        status.textContent = "Connected · thread: " + thread;
        var es = new EventSource("/a2a/stream?thread=" + encodeURIComponent(thread));
        _a2aEs = es;
        es.onmessage = function (e) {
          try {
            var msg = JSON.parse(e.data);
            _appendMsg(msg);
          } catch (_) {}
        };
        es.onerror = function () {
          status.textContent = "Disconnected from stream. Reload to reconnect.";
        };
      })
      .catch(function (e) {
        status.textContent = "Failed to load history: " + esc(e.message || e);
      });
  }

  $("a2aConnectBtn").addEventListener("click", a2aConnect);
  $("a2aThread").addEventListener("keydown", function (e) { if (e.key === "Enter") a2aConnect(); });

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

    If the server has ``server_token`` set in its own config (or the
    ``TAOSMD_TOKEN`` env var), every data/A2A JSON endpoint requires a
    matching ``Authorization: Bearer <token>`` header and returns ``401``
    otherwise. ``GET /health``, ``GET /``, ``GET /ui``, and static assets
    are always open so monitoring probes and the inspection UI keep working.
    """
    # Read the server-side expected token once at handler-class creation time.
    # This is the token the *server* checks (not the client's outbound token).
    _server_token: str | None = _config.get_server_token(data_dir)

    # Paths that are always public regardless of the token setting.
    _PUBLIC_PATHS = frozenset({"/", "/ui", "/health"})

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

        def _send_file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _serve_spa(self) -> None:
            """Serve the built React SPA index.html, or fall back to the inline UI."""
            if _WEBUI_DIR is not None:
                self._send_file(_WEBUI_DIR / "index.html", "text/html; charset=utf-8")
            else:
                self._send_html(200, _INSPECTION_UI_HTML)

        def _try_serve_static(self, path: str) -> bool:
            """Try to serve a static asset from webui/.

            Returns True if the asset was served, False if not found.
            """
            if _WEBUI_DIR is None:
                return False
            # Strip leading slash and prevent path traversal
            rel = path.lstrip("/")
            target = (_WEBUI_DIR / rel).resolve()
            try:
                target.relative_to(_WEBUI_DIR.resolve())
            except ValueError:
                return False
            if not target.is_file():
                return False
            suffix = target.suffix.lower()
            ctype = _EXTRA_MIME.get(suffix) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self._send_file(target, ctype)
            return True

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

        # ----- auth helper -------------------------------------------------
        def _check_token(self, path: str) -> bool:
            """Return True when the request is authorised to proceed.

            If ``_server_token`` is not set, every request is authorised.
            Public paths (health, UI) are always authorised.
            Otherwise the ``Authorization: Bearer <token>`` header must match.
            """
            if not _server_token:
                return True
            if path.rstrip("/") in _PUBLIC_PATHS or not path.rstrip("/"):
                return True
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                return auth[len("Bearer "):].strip() == _server_token
            return False

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
            # Token gate: check before routing so even unknown paths are
            # protected (prevents enumeration without a token).
            if not self._check_token(path):
                self._send_json(401, {"error": "Unauthorized"})
                return
            try:
                if method == "GET" and path in ("/", "/ui"):
                    self._serve_spa()
                elif method == "GET" and path == "/health":
                    self._send_json(200, {"status": "ok", "version": __version__})
                elif method == "GET" and path == "/search":
                    self._handle_search_get(query)
                elif method == "POST" and path == "/search":
                    self._handle_search_post()
                elif method == "POST" and path == "/ingest":
                    self._handle_ingest()
                elif method == "GET" and path == "/projects":
                    self._handle_list_projects()
                elif method == "GET" and path == "/shelves":
                    self._handle_list_shelves(query)
                elif method == "GET" and path == "/pending":
                    self._handle_pending(query)
                elif method == "POST" and path == "/pending/resolve":
                    self._handle_pending_resolve()
                elif method == "POST" and path == "/a2a/send":
                    self._handle_a2a_send()
                elif method == "GET" and path == "/a2a/channels":
                    self._handle_a2a_channels()
                elif method == "GET" and path == "/a2a/members":
                    self._handle_a2a_members(query)
                elif method == "GET" and path == "/a2a/messages":
                    self._handle_a2a_messages(query)
                elif method == "GET" and path == "/a2a/stream":
                    self._handle_a2a_stream(query)
                    return  # SSE response already sent; skip _send_json error path
                elif method == "GET" and self._try_serve_static(parts.path):
                    return  # static asset served
                elif method == "GET":
                    # SPA routing: unknown non-API paths return index.html
                    self._serve_spa()
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
            project = body.get("project")
            if not isinstance(text, str) or not text:
                raise _BadRequest("'text' (non-empty string) is required")
            if not isinstance(agent, str) or not agent:
                raise _BadRequest("'agent' (non-empty string) is required")
            if project is not None and not isinstance(project, str):
                raise _BadRequest("'project' must be a string when provided")
            opts = {"project": project} if project else {}
            result = runner.run(service.ingest(text, agent=agent, data_dir=data_dir, **opts))
            self._send_json(200, result)

        def _handle_search_post(self) -> None:
            body = self._read_json_body()
            query = body.get("query")
            agent = body.get("agent")
            limit = body.get("limit", 5)
            project = body.get("project")
            also_include = body.get("also_include")
            self._do_search(query, agent, limit, project, also_include)

        def _handle_search_get(self, qs: dict) -> None:
            query = (qs.get("q") or qs.get("query") or [None])[0]
            agent = (qs.get("agent") or [None])[0]
            limit = (qs.get("limit") or [5])[0]
            project = (qs.get("project") or [None])[0]
            # Comma-separated list in the query string, e.g. also_include=a,b
            ai_raw = (qs.get("also_include") or [None])[0]
            also_include = [s for s in ai_raw.split(",") if s] if ai_raw else None
            self._do_search(query, agent, limit, project, also_include)

        def _do_search(self, query, agent, limit, project=None, also_include=None) -> None:
            if not isinstance(query, str) or not query:
                raise _BadRequest("'query' (non-empty string) is required")
            if not isinstance(agent, str) or not agent:
                raise _BadRequest("'agent' (non-empty string) is required")
            if project is not None and not isinstance(project, str):
                raise _BadRequest("'project' must be a string when provided")
            if also_include is not None and not (
                isinstance(also_include, list) and all(isinstance(s, str) for s in also_include)
            ):
                raise _BadRequest("'also_include' must be a list of strings when provided")
            try:
                limit_i = int(limit)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            opts: dict = {}
            if project:
                opts["project"] = project
            if also_include:
                opts["also_include"] = also_include
            hits = runner.run(
                service.search(query, agent=agent, data_dir=data_dir, limit=limit_i, **opts)
            )
            self._send_json(200, {"hits": hits})

        def _handle_list_projects(self) -> None:
            projects = runner.run(service.list_projects(data_dir=data_dir))
            self._send_json(200, {"projects": projects})

        def _handle_list_shelves(self, qs: dict) -> None:
            project = (qs.get("project") or [None])[0]
            if not isinstance(project, str) or not project:
                raise _BadRequest("'project' (non-empty string) is required")
            shelves = runner.run(service.list_shelves(project=project, data_dir=data_dir))
            self._send_json(200, {"shelves": shelves})

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

        def _handle_a2a_channels(self) -> None:
            channels = runner.run(service.a2a_channels(data_dir=data_dir))
            self._send_json(200, {"channels": channels})

        def _handle_a2a_members(self, qs: dict) -> None:
            channel = (qs.get("channel") or [None])[0]
            if not channel:
                raise _BadRequest("'channel' query parameter is required")
            members = runner.run(service.a2a_members(channel=channel, data_dir=data_dir))
            self._send_json(200, {"members": members})

        def _handle_a2a_send(self) -> None:
            body = self._read_json_body()
            from_ = body.get("from")
            body_text = body.get("body")
            thread = body.get("thread", "general") or "general"
            reply_to = body.get("reply_to")
            if not isinstance(from_, str) or not from_:
                raise _BadRequest("'from' (non-empty string) is required")
            if not isinstance(body_text, str) or not body_text:
                raise _BadRequest("'body' (non-empty string) is required")
            result = runner.run(
                service.a2a_send(
                    sender=from_, body=body_text,
                    thread=thread, reply_to=reply_to,
                    data_dir=data_dir,
                )
            )
            self._send_json(200, result)

        def _handle_a2a_messages(self, qs: dict) -> None:
            thread = (qs.get("thread") or [None])[0]
            since_raw = (qs.get("since") or [None])[0]
            limit_raw = (qs.get("limit") or [50])[0]
            try:
                since = float(since_raw) if since_raw is not None else None
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'since' must be a float timestamp") from exc
            try:
                limit_i = int(limit_raw)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            messages = runner.run(
                service.a2a_feed(thread=thread, since=since, limit=limit_i, data_dir=data_dir)
            )
            self._send_json(200, {"messages": messages})

        def _handle_a2a_stream(self, qs: dict) -> None:
            """Server-Sent Events stream for the A2A bus.

            Runs in its own request thread (ThreadingHTTPServer); polls the
            archive once per second via the service loop and pushes new
            messages as ``data:`` SSE frames. Each ``runner.run(...)`` call
            is a quick, bounded archive query; the sleep happens here in
            the request thread, never inside the service loop. Disconnects
            are detected via write errors and exit the loop cleanly.
            """
            thread = (qs.get("thread") or [None])[0]
            since_raw = (qs.get("since") or [None])[0]
            try:
                last_ts = float(since_raw) if since_raw is not None else time.time()
            except (TypeError, ValueError):
                last_ts = time.time()

            # Send SSE response headers before entering the poll loop.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                while True:
                    msgs = runner.run(
                        service.a2a_feed(
                            thread=thread, since=last_ts, limit=200, data_dir=data_dir,
                        )
                    )
                    # Keep only messages strictly newer than last_ts to avoid
                    # re-sending the boundary row on subsequent polls.
                    new_msgs = [m for m in msgs if m["ts"] > last_ts]
                    if new_msgs:
                        for msg in new_msgs:
                            frame = f"data: {json.dumps(msg)}\n\n"
                            self.wfile.write(frame.encode("utf-8"))
                            last_ts = msg["ts"]
                        self.wfile.flush()
                    else:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Client disconnected; exit the thread cleanly.
                return

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

    Binds ``host:port`` (default ``127.0.0.1:7900``) and blocks serving
    requests. Returns 0 on a clean Ctrl-C shutdown.
    """
    httpd = make_server(host, port, data_dir)
    bound_host, bound_port = httpd.server_address[:2]
    where = "localhost only" if bound_host in {"127.0.0.1", "::1"} else "LAN-reachable (no auth)"
    print(f"taosmd HTTP API listening on http://{bound_host}:{bound_port} ({where})")
    print(f"Inspection UI (read-only): http://{bound_host}:{bound_port}/")
    print("Endpoints: GET /health, POST /ingest, GET|POST /search, "
          "GET /projects, GET /shelves, "
          "GET /pending, POST /pending/resolve, "
          "POST /a2a/send, GET /a2a/messages, GET /a2a/stream, "
          "GET /a2a/channels, GET /a2a/members")
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
