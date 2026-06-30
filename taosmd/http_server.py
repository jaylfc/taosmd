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
  (e.g. ``0.0.0.0``) only to expose it on the LAN. There is **no auth by
  default**: on localhost that is fine (any local process already has the
  Python API). When binding to a routable address, set a bearer token
  (``TAOSMD_TOKEN`` or ``taosmd config set-token``) so every data and A2A
  endpoint requires ``Authorization: Bearer``, and put the port behind your
  own network controls as defense in depth.

Security note
-------------
When a registry verifier is configured, ``POST /a2a/send`` runs in one of
two modes controlled by the ``a2a_auth_enforce`` config key:

* **verify-and-warn** (default, ``a2a_auth_enforce=false``): the registry
  EdDSA-JWT and grant check are performed; on failure a ``WARNING`` is logged
  (including the sender handle and reason) but the message is still accepted.
  This lets a deployment observe auth violations before enabling hard enforcement.
* **enforce** (``a2a_auth_enforce=true``): failure returns ``401`` (missing
  token) or ``403`` (bad token / no grant) and the message is dropped.

In both modes the token's ``sub`` claim is matched against the ``from`` field
to prevent impersonation, and an active grant in the grants feed is required.

Data endpoints (ingest, search, tasks) additionally support *optional* token
binding: when a Bearer token is present and the registry verifier is
configured, the token is verified and any ``project_id`` claim in it overrides
the request-supplied ``project`` value so callers cannot spoof their project
scope.  When no token is presented those endpoints behave exactly as today,
preserving the no-lockout, token-optional design.
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
                           Metadata note: callers may include ``forget_after`` (unix float) and
                           ``forget_reason`` (str) in the user metadata dict.  Rows whose
                           ``forget_after`` has passed are hidden from retrieval (zero-loss:
                           the raw row is never deleted).
``POST /ingest/batch``     ``{"items": [{"text", "id"?, "metadata"?: {"forget_after"?: float, "forget_reason"?: str, ...}}], "agent", "project"?}`` -> ``{"ingested", "skipped", ...}``
                           Per-item ``metadata`` may include ``forget_after`` (unix float) and
                           ``forget_reason`` (str) with the same TTL semantics as single ingest.
``POST /search``           ``{"query", "agent", "limit"?, "project"?, "also_include"?, "mode"?}`` -> ``{"hits": [...]}``
``GET  /search?q=&agent=&limit=&project=&also_include=a,b&mode=bm25``  -> ``{"hits": [...]}``
``GET  /projects``                                         -> ``{"projects": [...]}``
``GET  /shelves?project=``                                 -> ``{"shelves": [...]}``
``GET  /pending?agent=``                                   -> ``{"pending": [...]}``
``POST /pending/resolve``  ``{"id", "decision", "note"?}`` -> resolve result
``POST /a2a/send``         ``{"from", "body", "thread"?, "reply_to"?}`` -> send receipt
``GET  /a2a/messages``     ``?thread=&since=&limit=&fields=&format=``  -> ``{"messages": [...]}`` (``fields=id,sender,body`` projects keys; ``format=ndjson`` emits one message per line)
``GET  /a2a/stream``       ``?thread=&since=``             -> SSE stream (text/event-stream)
``GET  /a2a/channels``                                     -> ``{"channels": [...]}``
``GET  /a2a/members``      ``?channel=<name>``             -> ``{"members": [...]}``
``POST /tasks``            ``{"title", "body"?, "project"?, "assignee"?, "priority"?, "depends_on"?: [...], "created_by"}`` -> task object
``GET  /tasks``            ``?status=&project=&assignee=&limit=``  -> ``{"tasks": [...]}``
``GET  /tasks/ready``      ``?project=&assignee=&limit=``  -> ``{"tasks": [...]}``
``GET  /tasks/prime``      ``?project=&assignee=``         -> ``{"text": ..., "tasks": [...]}``
``POST /tasks/{id}``       ``{"status"?, "assignee"?, "priority"?, "body"?}`` -> updated task object
``POST /tasks/{id}/edges`` ``{"to_id", "type", "created_by"}``     -> edge record
``POST /tasks/{id}/edges/remove`` ``{"to_id", "type"}``   -> edge record with removed_ts

Admin endpoints (all require a configured server token; 403 if none is set)
``POST /shelves``                              ``{"shelf_id", "project_id"?, "display_name"?}`` -> ``{"shelf": {...}, "created": bool}``
``POST /shelves/{id}/archive``                 ``?expect_empty=true``  -> ``{"archived": true, "rows_hidden": int}``
``POST /shelves/{id}/unarchive``               -> ``{"archived": false, "rows_restored": int}``
``POST /a2a/admin/delete-channel``             ``{"channel": str}`` -> ``{"deleted": true, "channel": str}``
``POST /a2a/admin/rename-channel``             ``{"from": str, "to": str}`` -> ``{"renamed": true, "from": str, "to": str}``
``POST /a2a/admin/supersede-message``          ``{"id": int}`` -> ``{"superseded": true, "id": int}``

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
    // Escapes for BOTH element and attribute contexts: quotes must be
    // encoded because values like the A2A sender name are interpolated
    // into attributes (data-sender="..."), and the bus is free-handle.
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
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
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # Loop is stopped/closing (e.g. server teardown mid-poll). Close the
            # coroutine so it is not left un-awaited, then surface the failure.
            coro.close()
            raise
        return future.result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


def _make_handler(data_dir, runner: _ServiceLoop, verifier=None,
                  grants_verifier=None):
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

    # Whether to serve the web dashboard (/ /ui static assets SPA fallback).
    # Defaults True for standalone installs; False when managed_by=taos (taOS
    # apps render everything). Overridable via config/env.
    _serve_dashboard: bool = _config.get_serve_dashboard(data_dir)

    # Opt-in A2A registry verifier. Injected for tests; otherwise built from the
    # configured registry URL. When None, the bus trusts the handle (standalone).
    _registry_verifier = verifier
    _grants_verifier = grants_verifier
    if _registry_verifier is None:
        _registry_url = _config.get_registry_url(data_dir)
        if _registry_url:
            from . import registry_auth  # noqa: PLC0415 - optional path
            # The revoked and grants feeds are admin-gated (#710/#719): send the
            # configured taOS local token on them; pin the issuer.
            _admin_token = _config.get_registry_token(data_dir)
            _registry_verifier = registry_auth.verifier_from_url(
                _registry_url,
                revoked_token=_admin_token,
                expected_iss=registry_auth.REGISTRY_ISS,
            )
            _grants_verifier = registry_auth.grants_verifier_from_url(
                _registry_url,
                grants_token=_admin_token,
            )

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

        # ----- auth helpers ------------------------------------------------
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

        def _check_admin_token(self) -> bool:
            """Return True when the request carries the correct admin token.

            Admin endpoints FAIL CLOSED: if no server token is configured, all
            admin requests return 403. This is the inverse of ``_check_token``
            which passes everything through when no token is configured.
            If a token is configured the Bearer must match it exactly.
            Returns False and writes the error response when auth fails; the
            caller must return immediately in that case.
            """
            if not _server_token:
                self._send_json(
                    403,
                    {"error": "admin surface requires a configured server token"},
                )
                return False
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer ") and auth[len("Bearer "):].strip() == _server_token:
                return True
            self._send_json(401, {"error": "Unauthorized"})
            return False

        def _apply_token_binding(self, identity: str | None, project: str | None) -> tuple[str | None, bool]:
            """Apply optional registry-token project binding for data endpoints.

            When a registry verifier is configured AND the request carries an
            Authorization Bearer token, the token is verified (returning 403
            and ``False`` on failure; callers must check the bool before
            continuing).  On success the ``project_id`` claim from the
            verified token, if present, OVERRIDES the request-supplied
            ``project`` value (anti-spoof: request-supplied values are ignored
            when a token is present).  A grants verifier, when present,
            additionally requires an active grant for ``(sub, project_id)``
            when a project binding is found; a missing grant yields 403.

            When NO token is presented the call is a no-op: the original
            ``project`` is returned unchanged and ``True`` is returned.

            ``identity`` (the request's ``agent``/``created_by`` field) is
            accepted for symmetry but NOT compared against ``sub``: for data
            endpoints the agent field names a target shelf, not the caller
            (the taOS proxy writes the ``user-memory`` shelf under its own
            controller token). The token's own ``sub`` is used for the
            authorize check, which still enforces signature, issuer, and
            revocation; whether ``sub`` may act on a given shelf is the
            deferred identity-keying work, not this layer.

            Returns ``(resolved_project, ok)`` where ``ok=False`` means the
            response has already been written and the caller must return.
            """
            if _registry_verifier is None:
                return project, True
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                # No token presented: leave behaviour exactly as before.
                return project, True
            token = auth[len("Bearer "):].strip()
            if not token:
                return project, True
            from . import registry_auth as _ra  # noqa: PLC0415
            # Peek at sub without verifying signature (only to satisfy the
            # authorize(token, claimed_from) API when the request has no
            # identity field).  The real signature check happens inside
            # authorize(); a bad token will still fail there.
            raw_sub: str = ""
            try:
                import jwt as _jwt  # noqa: PLC0415
                unverified = _jwt.decode(token, options={"verify_signature": False})
                raw_sub = unverified.get("sub", "") or ""
            except Exception:  # noqa: BLE001
                pass
            claimed_identity = raw_sub
            try:
                claims = _registry_verifier.authorize(token, claimed_identity)
            except _ra.AuthError as exc:
                self._send_json(403, {"error": f"registry auth: {exc}"})
                return None, False
            verified_project = claims.get("project_id")
            if verified_project is not None:
                project = verified_project
            # Token proves identity; a grant proves permission. Any verified
            # token, project-bound or global, must hold an active grant when
            # the grants verifier is configured. Tokenless requests stay
            # unchanged (no-lockout). Caught by the #744 e2e: the previous
            # gate only fired when the token carried a project_id claim, so
            # a grant-less GLOBAL token could still write.
            if _grants_verifier is not None:
                sub = claims.get("sub", "")
                try:
                    if not _grants_verifier.has_grant(sub, project_id=verified_project):
                        scope_desc = (
                            f"({sub!r}, project {verified_project!r})"
                            if verified_project is not None else repr(sub)
                        )
                        self._send_json(403, {"error": f"registry auth: no active grant for {scope_desc}"})
                        return None, False
                except _ra.AuthError as exc:
                    self._send_json(403, {"error": f"registry auth: {exc}"})
                    return None, False
            return project, True

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
                    if _serve_dashboard:
                        self._serve_spa()
                    else:
                        self._send_json(404, {"error": "dashboard disabled (managed_by=taos)"})
                elif method == "GET" and path == "/health":
                    self._send_json(200, {"status": "ok", "version": __version__})
                elif method == "GET" and path == "/controls":
                    self._handle_controls_get()
                elif method == "POST" and path == "/controls":
                    self._handle_controls_post()
                elif method == "GET" and path == "/generator-profile":
                    self._handle_generator_profile_get(query)
                elif method == "POST" and path == "/generator-profile":
                    self._handle_generator_profile_post()
                elif method == "GET" and path == "/stats":
                    self._handle_stats(query)
                elif method == "GET" and path == "/memories":
                    self._handle_memories(query)
                elif method == "GET" and path == "/graph":
                    self._handle_graph(query)
                elif method == "GET" and path == "/graph/activations":
                    self._handle_graph_activations(query)
                elif method == "GET" and path == "/search":
                    self._handle_search_get(query)
                elif method == "POST" and path == "/search":
                    self._handle_search_post()
                elif method == "POST" and path == "/ingest":
                    self._handle_ingest()
                elif method == "POST" and path == "/ingest/batch":
                    self._handle_ingest_batch()
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
                # Task graph endpoints — prefix matching for /tasks/{id} paths
                elif method == "POST" and path == "/tasks":
                    self._handle_task_create()
                elif method == "GET" and path == "/tasks":
                    self._handle_task_list(query)
                elif method == "GET" and path == "/tasks/ready":
                    self._handle_task_ready(query)
                elif method == "GET" and path == "/tasks/prime":
                    self._handle_task_prime(query)
                elif method == "POST" and path.startswith("/tasks/"):
                    # /tasks/{id}/edges/remove  or  /tasks/{id}/edges  or  /tasks/{id}
                    rest = path[len("/tasks/"):]
                    if rest.endswith("/edges/remove"):
                        task_id = rest[: -len("/edges/remove")]
                        if not task_id:
                            self._send_json(404, {"error": "task id required"})
                        else:
                            self._handle_task_remove_edge(task_id)
                    elif rest.endswith("/edges"):
                        task_id = rest[: -len("/edges")]
                        if not task_id:
                            self._send_json(404, {"error": "task id required"})
                        else:
                            self._handle_task_add_edge(task_id)
                    else:
                        task_id = rest
                        if not task_id:
                            self._send_json(404, {"error": "task id required"})
                        else:
                            self._handle_task_update(task_id)
                # ----- admin surface: shelf lifecycle ---------------------
                elif method == "POST" and path == "/shelves":
                    self._handle_admin_shelf_create()
                elif method == "POST" and path.startswith("/shelves/"):
                    rest = path[len("/shelves/"):]
                    if rest.endswith("/archive"):
                        shelf_id = rest[: -len("/archive")]
                        if not shelf_id:
                            self._send_json(404, {"error": "shelf id required"})
                        else:
                            self._handle_admin_shelf_archive(shelf_id, query)
                    elif rest.endswith("/unarchive"):
                        shelf_id = rest[: -len("/unarchive")]
                        if not shelf_id:
                            self._send_json(404, {"error": "shelf id required"})
                        else:
                            self._handle_admin_shelf_unarchive(shelf_id)
                    else:
                        self._send_json(404, {"error": f"unknown shelf action: {rest}"})
                # ----- admin surface: A2A channel admin -------------------
                elif method == "POST" and path == "/a2a/admin/delete-channel":
                    self._handle_admin_a2a_delete_channel()
                elif method == "POST" and path == "/a2a/admin/rename-channel":
                    self._handle_admin_a2a_rename_channel()
                elif method == "POST" and path == "/a2a/admin/supersede-message":
                    self._handle_admin_a2a_supersede_message()
                elif method == "GET" and _serve_dashboard and self._try_serve_static(parts.path):
                    return  # static asset served
                elif method == "GET" and _serve_dashboard:
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
            project, ok = self._apply_token_binding(agent, project)
            if not ok:
                return
            opts = {"project": project} if project else {}
            result = runner.run(service.ingest(text, agent=agent, data_dir=data_dir, **opts))
            self._send_json(200, result)

        def _handle_ingest_batch(self) -> None:
            body = self._read_json_body()
            items = body.get("items")
            agent = body.get("agent")
            project = body.get("project")
            if not isinstance(items, list):
                raise _BadRequest("'items' (list) is required")
            if not isinstance(agent, str) or not agent:
                raise _BadRequest("'agent' (non-empty string) is required")
            if project is not None and not isinstance(project, str):
                raise _BadRequest("'project' must be a string when provided")
            project, ok = self._apply_token_binding(agent, project)
            if not ok:
                return
            opts = {"project": project} if project else {}
            # Per-item shape validation lives in api.ingest_batch and runs
            # before any write; its ValueError surfaces as a 400 here.
            result = runner.run(
                service.ingest_batch(items, agent=agent, data_dir=data_dir, **opts)
            )
            self._send_json(200, result)

        def _handle_search_post(self) -> None:
            body = self._read_json_body()
            query = body.get("query")
            agent = body.get("agent")
            limit = body.get("limit", 5)
            project = body.get("project")
            also_include = body.get("also_include")
            mode = body.get("mode")
            project, ok = self._apply_token_binding(agent, project)
            if not ok:
                return
            self._do_search(query, agent, limit, project, also_include, mode)

        def _handle_search_get(self, qs: dict) -> None:
            query = (qs.get("q") or qs.get("query") or [None])[0]
            agent = (qs.get("agent") or [None])[0]
            limit = (qs.get("limit") or [5])[0]
            project = (qs.get("project") or [None])[0]
            # Comma-separated list in the query string, e.g. also_include=a,b
            ai_raw = (qs.get("also_include") or [None])[0]
            also_include = [s for s in ai_raw.split(",") if s] if ai_raw else None
            mode = (qs.get("mode") or [None])[0]
            project, ok = self._apply_token_binding(agent, project)
            if not ok:
                return
            self._do_search(query, agent, limit, project, also_include, mode)

        def _do_search(self, query, agent, limit, project=None, also_include=None, mode=None) -> None:
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
            if mode is not None and not isinstance(mode, str):
                raise _BadRequest("'mode' must be a string when provided")
            opts: dict = {}
            if project:
                opts["project"] = project
            if also_include:
                opts["also_include"] = also_include
            if mode:
                opts["mode"] = mode
            hits = runner.run(
                service.search(query, agent=agent, data_dir=data_dir, limit=limit_i, **opts)
            )
            self._send_json(200, {"hits": hits})

        def _handle_list_projects(self) -> None:
            projects = runner.run(service.list_projects(data_dir=data_dir))
            self._send_json(200, {"projects": projects})

        def _handle_controls_get(self) -> None:
            from taosmd import config as _config, controls as _controls  # noqa: PLC0415
            self._send_json(200, {
                "settings": _config.get_controls(data_dir=data_dir),
                "schema": _controls.controls_schema(),
            })

        def _handle_controls_post(self) -> None:
            from taosmd import config as _config, controls as _controls  # noqa: PLC0415
            body = self._read_json_body()
            preset = body.get("preset")
            if preset:
                p = _controls.PRESETS.get(preset)
                if not p:
                    self._send_json(400, {"error": f"unknown preset: {preset!r}"})
                    return
                updates = dict(p["values"])
            else:
                vals = body.get("values")
                updates = vals if isinstance(vals, dict) else body
            if not isinstance(updates, dict) or not updates:
                self._send_json(400, {
                    "error": "body must be {control_id: value}, {values: {...}}, or {preset: id}"})
                return
            errors = {}
            for cid, val in updates.items():
                try:
                    _config.set_control(cid, val, data_dir=data_dir)
                except ValueError as exc:
                    errors[cid] = str(exc)
            self._send_json(400 if errors else 200, {
                "settings": _config.get_controls(data_dir=data_dir),
                "errors": errors,
            })

        def _handle_generator_profile_get(self, query) -> None:
            from taosmd import generator_profiles as _gp, config as _cfg, agents as _agents  # noqa: PLC0415
            raw = query.get("agent")
            agent = raw[0] if isinstance(raw, list) else raw
            profiles = [
                {"id": p.id, "label": p.label, "workload": p.workload, "models": p.models}
                for p in _gp.list_profiles()
            ]
            active, scope = None, "global"
            if agent:
                try:
                    pid = _agents.get_agent_generator_profile(agent, data_dir=data_dir)
                except Exception:
                    pid = None
                if pid:
                    active, scope = pid, agent
            if active is None:
                active = _cfg.get_generator_profile(data_dir=data_dir) or _gp.default_profile_id()
            self._send_json(200, {"profiles": profiles, "active": active, "scope": scope})

        def _handle_generator_profile_post(self) -> None:
            from taosmd import generator_profiles as _gp, config as _cfg, agents as _agents  # noqa: PLC0415
            body = self._read_json_body()
            pid = body.get("profile_id")
            agent = body.get("agent")
            if not pid or _gp.get_profile(pid) is None:
                self._send_json(400, {"error": f"unknown profile: {pid!r}"})
                return
            try:
                if agent:
                    _agents.set_agent_generator_profile(agent, pid, data_dir=data_dir)
                else:
                    _cfg.set_generator_profile(pid, data_dir=data_dir)
            except (_agents.AgentNotFoundError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            profiles = [
                {"id": p.id, "label": p.label, "workload": p.workload, "models": p.models}
                for p in _gp.list_profiles()
            ]
            if agent:
                active, scope = pid, agent
            else:
                active, scope = pid, "global"
            self._send_json(200, {"profiles": profiles, "active": active, "scope": scope})

        def _handle_stats(self, qs: dict) -> None:
            scope = (qs.get("scope") or [None])[0]
            result = runner.run(service.dashboard_stats(scope=scope, data_dir=data_dir))
            self._send_json(200, result)

        def _handle_memories(self, qs: dict) -> None:
            scope = (qs.get("scope") or [None])[0]
            limit = (qs.get("limit") or [50])[0]
            try:
                limit_i = int(limit)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            memories = runner.run(
                service.list_memories(scope=scope, limit=limit_i, data_dir=data_dir)
            )
            self._send_json(200, {"memories": memories})

        def _handle_graph(self, qs: dict) -> None:
            limit = (qs.get("limit") or [300])[0]
            try:
                limit_i = int(limit)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            result = runner.run(service.graph(limit=limit_i, data_dir=data_dir))
            self._send_json(200, result)

        def _handle_graph_activations(self, qs: dict) -> None:
            since = (qs.get("since") or [None])[0]
            window = (qs.get("window") or [60])[0]
            limit = (qs.get("limit") or [100])[0]
            try:
                since_f = float(since) if since is not None else None
                window_f = float(window)
                limit_i = int(limit)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("since/window must be numbers and limit an integer") from exc
            result = runner.run(
                service.graph_activations(since=since_f, window=window_f, limit=limit_i, data_dir=data_dir)
            )
            self._send_json(200, result)

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
            # Registry auth (opt-in): when a verifier is configured, run the
            # identity + grant checks and collect any failure reason.
            # In enforce mode (a2a_auth_enforce=true) failures are rejected with
            # 401/403. In verify-and-warn mode (default) failures are logged as
            # a WARNING but the message is accepted, allowing operators to observe
            # violations before enabling hard enforcement.
            if _registry_verifier is not None:
                from . import registry_auth  # noqa: PLC0415 - optional path
                auth = self.headers.get("Authorization", "")
                token = auth[len("Bearer "):].strip() if auth.startswith("Bearer ") else ""

                # Compute warn_reason (None = auth passed) and the status/message
                # to use in enforce mode. We collect these without returning early
                # so the enforce vs. warn decision is made in one place below.
                warn_reason: str | None = None
                _reject_status: int = 403
                _reject_msg: str = ""

                if not token:
                    warn_reason = "missing Bearer token"
                    _reject_status = 401
                    _reject_msg = "registry auth: Bearer token required"
                else:
                    try:
                        _registry_verifier.authorize(token, from_)
                    except registry_auth.AuthError as exc:
                        warn_reason = str(exc)
                        _reject_status = 403
                        _reject_msg = f"registry auth: {exc}"

                # Grant check: token proves identity; grant proves permission.
                if warn_reason is None and _grants_verifier is not None:
                    try:
                        if not _grants_verifier.has_grant(from_):
                            warn_reason = "no a2a_send grant"
                            _reject_status = 403
                            _reject_msg = f"registry auth: no active grant for {from_!r}"
                    except registry_auth.AuthError as exc:
                        warn_reason = str(exc)
                        _reject_status = 403
                        _reject_msg = f"registry auth: {exc}"

                if warn_reason is not None:
                    enforce = _config.get_a2a_auth_enforce(data_dir)
                    if enforce:
                        self._send_json(_reject_status, {"error": _reject_msg})
                        return
                    logger.warning(
                        "a2a verify-and-warn: accepting unverified post from %r: %s",
                        from_, warn_reason,
                    )
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
            fields_raw = (qs.get("fields") or [None])[0]
            fmt = (qs.get("format") or ["json"])[0]
            try:
                since = float(since_raw) if since_raw is not None else None
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'since' must be a float timestamp") from exc
            try:
                limit_i = int(limit_raw)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            if fmt not in ("json", "ndjson"):
                raise _BadRequest("'format' must be 'json' or 'ndjson'")
            messages = runner.run(
                service.a2a_feed(thread=thread, since=since, limit=limit_i, data_dir=data_dir)
            )
            # Compact mode: ?fields=id,sender,body projects each message down
            # to the named keys so token-frugal consumers (LLM agents) skip
            # framing they never read. Unknown names are ignored, never a 400,
            # so consumers stay forward-compatible across shape changes.
            if fields_raw:
                wanted = [f.strip() for f in fields_raw.split(",") if f.strip()]
                if wanted:
                    messages = [
                        {k: m[k] for k in wanted if k in m} for m in messages
                    ]
            if fmt == "ndjson":
                body = "".join(
                    json.dumps(m, ensure_ascii=False) + "\n" for m in messages
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
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

        # ----- task graph handlers ----------------------------------------

        def _handle_task_create(self) -> None:
            body = self._read_json_body()
            title = body.get("title")
            created_by = body.get("created_by")
            if not isinstance(title, str) or not title:
                raise _BadRequest("'title' (non-empty string) is required")
            if not isinstance(created_by, str) or not created_by:
                raise _BadRequest("'created_by' (non-empty string) is required")
            task_body = body.get("body")
            project = body.get("project")
            assignee = body.get("assignee")
            priority = body.get("priority", 0)
            depends_on = body.get("depends_on")
            project, ok = self._apply_token_binding(created_by, project)
            if not ok:
                return
            try:
                priority = int(priority)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'priority' must be an integer") from exc
            if depends_on is not None and not isinstance(depends_on, list):
                raise _BadRequest("'depends_on' must be a list of task IDs when provided")
            result = runner.run(
                service.task_create(
                    title,
                    body=task_body,
                    project=project,
                    assignee=assignee,
                    priority=priority,
                    depends_on=depends_on,
                    created_by=created_by,
                    data_dir=data_dir,
                )
            )
            self._send_json(200, result)

        def _handle_task_list(self, qs: dict) -> None:
            status = (qs.get("status") or [None])[0]
            project = (qs.get("project") or [None])[0]
            assignee = (qs.get("assignee") or [None])[0]
            limit_raw = (qs.get("limit") or [50])[0]
            try:
                limit_i = int(limit_raw)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            project, ok = self._apply_token_binding(assignee, project)
            if not ok:
                return
            tasks = runner.run(
                service.task_list(
                    status=status,
                    project=project,
                    assignee=assignee,
                    limit=limit_i,
                    data_dir=data_dir,
                )
            )
            self._send_json(200, {"tasks": tasks})

        def _handle_task_ready(self, qs: dict) -> None:
            project = (qs.get("project") or [None])[0]
            assignee = (qs.get("assignee") or [None])[0]
            limit_raw = (qs.get("limit") or [20])[0]
            try:
                limit_i = int(limit_raw)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'limit' must be an integer") from exc
            project, ok = self._apply_token_binding(assignee, project)
            if not ok:
                return
            tasks = runner.run(
                service.task_ready(
                    project=project,
                    assignee=assignee,
                    limit=limit_i,
                    data_dir=data_dir,
                )
            )
            self._send_json(200, {"tasks": tasks})

        def _handle_task_prime(self, qs: dict) -> None:
            project = (qs.get("project") or [None])[0]
            assignee = (qs.get("assignee") or [None])[0]
            project, ok = self._apply_token_binding(assignee, project)
            if not ok:
                return
            result = runner.run(
                service.task_prime(
                    project=project,
                    assignee=assignee,
                    data_dir=data_dir,
                )
            )
            self._send_json(200, result)

        def _handle_task_update(self, task_id: str) -> None:
            body = self._read_json_body()
            status = body.get("status")
            assignee = body.get("assignee")
            priority = body.get("priority")
            task_body = body.get("body")
            _, ok = self._apply_token_binding(assignee, None)
            if not ok:
                return
            if priority is not None:
                try:
                    priority = int(priority)
                except (TypeError, ValueError) as exc:
                    raise _BadRequest("'priority' must be an integer") from exc
            try:
                result = runner.run(
                    service.task_update(
                        task_id,
                        status=status,
                        assignee=assignee,
                        priority=priority,
                        body=task_body,
                        data_dir=data_dir,
                    )
                )
            except ValueError as exc:
                raise _BadRequest(str(exc)) from exc
            self._send_json(200, result)

        def _handle_task_add_edge(self, from_id: str) -> None:
            body = self._read_json_body()
            to_id = body.get("to_id")
            edge_type = body.get("type")
            created_by = body.get("created_by", "unknown")
            if not isinstance(to_id, str) or not to_id:
                raise _BadRequest("'to_id' (non-empty string) is required")
            if not isinstance(edge_type, str) or not edge_type:
                raise _BadRequest("'type' (non-empty string) is required")
            try:
                result = runner.run(
                    service.task_add_edge(
                        from_id, to_id, edge_type, created_by,
                        data_dir=data_dir,
                    )
                )
            except ValueError as exc:
                raise _BadRequest(str(exc)) from exc
            self._send_json(200, result)

        def _handle_task_remove_edge(self, from_id: str) -> None:
            body = self._read_json_body()
            to_id = body.get("to_id")
            edge_type = body.get("type")
            if not isinstance(to_id, str) or not to_id:
                raise _BadRequest("'to_id' (non-empty string) is required")
            if not isinstance(edge_type, str) or not edge_type:
                raise _BadRequest("'type' (non-empty string) is required")
            try:
                result = runner.run(
                    service.task_remove_edge(
                        from_id, to_id, edge_type,
                        data_dir=data_dir,
                    )
                )
            except ValueError as exc:
                raise _BadRequest(str(exc)) from exc
            self._send_json(200, result)

        # ----- admin: shelf lifecycle ------------------------------------

        def _handle_admin_shelf_create(self) -> None:
            if not self._check_admin_token():
                return
            body = self._read_json_body()
            shelf_id = body.get("shelf_id")
            project_id = body.get("project_id")
            display_name = body.get("display_name")
            if not isinstance(shelf_id, str) or not shelf_id:
                raise _BadRequest("'shelf_id' (non-empty string) is required")
            if project_id is not None and not isinstance(project_id, str):
                raise _BadRequest("'project_id' must be a string when provided")
            if display_name is not None and not isinstance(display_name, str):
                raise _BadRequest("'display_name' must be a string when provided")
            from .admin import InvalidAgentNameError  # noqa: PLC0415
            try:
                result = runner.run(
                    service.admin_shelf_create(
                        shelf_id,
                        project_id=project_id,
                        display_name=display_name,
                        data_dir=data_dir,
                    )
                )
            except InvalidAgentNameError as exc:
                raise _BadRequest(str(exc)) from exc
            status = 200
            self._send_json(status, result)

        def _handle_admin_shelf_archive(self, shelf_id: str, query: dict) -> None:
            if not self._check_admin_token():
                return
            expect_empty_raw = (query.get("expect_empty") or [None])[0]
            expect_empty = (
                expect_empty_raw is not None and expect_empty_raw.lower() == "true"
            )
            from .admin import ShelfNotFoundError, ShelfNotEmptyError  # noqa: PLC0415
            try:
                result = runner.run(
                    service.admin_shelf_archive(
                        shelf_id,
                        expect_empty=expect_empty,
                        data_dir=data_dir,
                    )
                )
            except ShelfNotFoundError as exc:
                self._send_json(404, {"error": str(exc)})
                return
            except ShelfNotEmptyError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            self._send_json(200, result)

        def _handle_admin_shelf_unarchive(self, shelf_id: str) -> None:
            if not self._check_admin_token():
                return
            from .admin import ShelfNotFoundError  # noqa: PLC0415
            try:
                result = runner.run(
                    service.admin_shelf_unarchive(shelf_id, data_dir=data_dir)
                )
            except ShelfNotFoundError as exc:
                self._send_json(404, {"error": str(exc)})
                return
            self._send_json(200, result)

        # ----- admin: A2A channel admin ----------------------------------

        def _handle_admin_a2a_delete_channel(self) -> None:
            if not self._check_admin_token():
                return
            body = self._read_json_body()
            channel = body.get("channel")
            if not isinstance(channel, str) or not channel:
                raise _BadRequest("'channel' (non-empty string) is required")
            result = runner.run(
                service.admin_a2a_delete_channel(channel, data_dir=data_dir)
            )
            self._send_json(200, result)

        def _handle_admin_a2a_rename_channel(self) -> None:
            if not self._check_admin_token():
                return
            body = self._read_json_body()
            from_channel = body.get("from")
            to_channel = body.get("to")
            if not isinstance(from_channel, str) or not from_channel:
                raise _BadRequest("'from' (non-empty string) is required")
            if not isinstance(to_channel, str) or not to_channel:
                raise _BadRequest("'to' (non-empty string) is required")
            try:
                result = runner.run(
                    service.admin_a2a_rename_channel(
                        from_channel, to_channel, data_dir=data_dir
                    )
                )
            except ValueError as exc:
                raise _BadRequest(str(exc)) from exc
            self._send_json(200, result)

        def _handle_admin_a2a_supersede_message(self) -> None:
            if not self._check_admin_token():
                return
            body = self._read_json_body()
            msg_id = body.get("id")
            if msg_id is None:
                raise _BadRequest("'id' (integer) is required")
            try:
                msg_id = int(msg_id)
            except (TypeError, ValueError) as exc:
                raise _BadRequest("'id' must be an integer") from exc
            result = runner.run(
                service.admin_a2a_supersede_message(msg_id, data_dir=data_dir)
            )
            self._send_json(200, result)

    return TaosmdHandler


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, data_dir=None,
                verifier=None, grants_verifier=None):
    """Create (but do not start) a :class:`ThreadingHTTPServer`.

    Useful for tests that need to bind an ephemeral port (``port=0``) and read
    back the assigned address before serving. Call ``server.serve_forever()``
    to run it, ``server.shutdown()`` + ``server.server_close()`` to stop.

    A :class:`_ServiceLoop` is started and attached as ``server.service_loop``;
    closing it is handled by :func:`serve`. Tests that drive ``make_server``
    directly should call ``server.service_loop.close()`` during teardown.
    """
    if verifier is not None and grants_verifier is None:
        # Identity without permission is half the locked auth contract
        # (token proves who you are; a grant proves you may post). Warn
        # loudly so a deployment cannot end up with partial auth silently.
        logger.warning(
            "registry auth: a registry verifier is configured but no grants "
            "verifier — A2A sends will be identity-checked but NOT grant-checked"
        )
    runner = _ServiceLoop()
    httpd = ThreadingHTTPServer(
        (host, port),
        _make_handler(data_dir, runner, verifier, grants_verifier),
    )
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
    print("Endpoints: GET /health, POST /ingest, POST /ingest/batch, GET|POST /search, "
          "GET /projects, GET /shelves, "
          "GET /pending, POST /pending/resolve, "
          "POST /a2a/send, GET /a2a/messages, GET /a2a/stream, "
          "GET /a2a/channels, GET /a2a/members, "
          "POST /tasks, GET /tasks, GET /tasks/ready, GET /tasks/prime, "
          "POST /tasks/{id}, POST /tasks/{id}/edges, POST /tasks/{id}/edges/remove")
    print("Admin (token required): POST /shelves, POST /shelves/{id}/archive, "
          "POST /shelves/{id}/unarchive, "
          "POST /a2a/admin/delete-channel, POST /a2a/admin/rename-channel, "
          "POST /a2a/admin/supersede-message")
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
