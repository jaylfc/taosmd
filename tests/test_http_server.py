"""Tests for taosmd.http_server — the local HTTP/REST activation surface.

Offline + fast: the server runs in a background thread on an ephemeral port
with an isolated tmp data dir, and the vector embedder is patched (same
deterministic hash vector as tests/test_api.py) so no ONNX/QMD model is
needed. Requests go over the loopback via urllib.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server


def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder so search finds matching text."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """Start the HTTP server on an ephemeral port against an isolated data dir.

    Yields the base URL (e.g. ``http://127.0.0.1:54321``). Tears the server
    and the cached SQLite stores down cleanly afterwards.
    """
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))

    # Init the stores *on the server's service loop thread* (so the thread-
    # affine SQLite connections live where the handlers will use them), then
    # patch the embedder so search doesn't need a real ONNX/QMD model.
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        httpd.service_loop.close()


def _post(url: str, payload) -> tuple[int, dict]:
    data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    return _send(req)


def _get(url: str) -> tuple[int, dict]:
    return _send(urllib.request.Request(url, method="GET"))


def _send(req) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _get_raw(url: str) -> tuple[int, str, str]:
    """GET returning (status, content_type, body_text) — for the HTML UI."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode()


def test_health(live_server):
    status, body = _get(f"{live_server}/health")
    assert status == 200
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]


def test_ingest_then_search_roundtrip(live_server):
    status, body = _post(
        f"{live_server}/ingest",
        {"text": "The HTTP API ships on the feat/http-api branch.", "agent": "http-test"},
    )
    assert status == 200, body
    assert body["archived"] == 1
    assert body["agent"] == "http-test"

    status, body = _post(
        f"{live_server}/search",
        {"query": "The HTTP API ships on the feat/http-api branch.", "agent": "http-test"},
    )
    assert status == 200, body
    assert body["hits"], "expected the ingested text to be retrievable"
    assert "HTTP API" in body["hits"][0]["text"]


def test_search_get_query_param(live_server):
    _post(
        f"{live_server}/ingest",
        {"text": "GET-style search works over query params.", "agent": "http-test"},
    )
    status, body = _get(
        f"{live_server}/search?q=GET-style%20search%20works%20over%20query%20params.&agent=http-test&limit=3"
    )
    assert status == 200, body
    assert body["hits"]
    assert "GET-style" in body["hits"][0]["text"]


def test_bad_json_body_returns_400(live_server):
    status, body = _post(f"{live_server}/ingest", b"{not valid json")
    assert status == 400
    assert "error" in body


def test_missing_field_returns_400(live_server):
    status, body = _post(f"{live_server}/ingest", {"text": "no agent here"})
    assert status == 400
    assert "agent" in body["error"]


def test_unknown_route_serves_spa_or_404(live_server):
    """Unknown GET paths serve the SPA index (for client-side routing) or a fallback.

    With the built webui present (the default in a source checkout), unknown
    paths return 200 HTML so the SPA can handle client-side routing.
    With no webui, the inline UI is served (also 200 HTML).  POST to unknown
    paths still returns 404 JSON.
    """
    # GET unknown path -> HTML (SPA routing or inline fallback)
    status, ctype, html = _get_raw(f"{live_server}/does-not-exist")
    assert status == 200
    assert "text/html" in ctype
    assert "<html" in html.lower()

    # POST unknown path -> 404 JSON
    status, body = _post(f"{live_server}/no-such-endpoint", {})
    assert status == 404
    assert "error" in body


def test_pending_empty(live_server):
    status, body = _get(f"{live_server}/pending?agent=http-test")
    assert status == 200
    assert body["pending"] == []


def test_pending_resolve_bad_decision_returns_400(live_server):
    status, body = _post(
        f"{live_server}/pending/resolve",
        {"id": "abc", "decision": "frobnicate"},
    )
    assert status == 400
    assert "decision" in body["error"]


# ----- UI serving (built SPA + fallback) ----------------------------------


def _make_server_with_webui(tmp_path, monkeypatch):
    """Start the HTTP server with a *fake* built webui injected via monkeypatch."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    # Create a minimal fake webui that the server can serve.
    fake_webui = tmp_path / "fake_webui"
    fake_webui.mkdir()
    (fake_webui / "index.html").write_text(
        "<!doctype html><html><head><title>taOSmd-test</title></head><body>test-spa</body></html>",
        encoding="utf-8",
    )
    assets = fake_webui / "assets"
    assets.mkdir()
    (assets / "app.js").write_bytes(b"console.log('hello');")
    (assets / "style.css").write_bytes(b"body { color: red; }")

    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    import taosmd.http_server as _hs  # noqa: PLC0415
    monkeypatch.setattr(_hs, "_WEBUI_DIR", fake_webui)

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return httpd


def test_root_serves_built_spa_when_webui_present(tmp_path, monkeypatch):
    """GET / returns the built index.html when taosmd/webui/ exists."""
    httpd = _make_server_with_webui(tmp_path, monkeypatch)
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        status, ctype, html = _get_raw(f"http://{host}:{port}/")
        assert status == 200
        assert "text/html" in ctype
        assert "taOSmd-test" in html
        assert "test-spa" in html
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


def test_root_falls_back_to_inline_ui_when_webui_absent(tmp_path, monkeypatch):
    """GET / returns the inline fallback HTML when taosmd/webui/ is absent."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    import taosmd.http_server as _hs  # noqa: PLC0415
    monkeypatch.setattr(_hs, "_WEBUI_DIR", None)  # force fallback

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        status, ctype, html = _get_raw(f"http://{host}:{port}/")
        assert status == 200
        assert "text/html" in ctype
        # Must be the inline fallback, not an SPA
        assert "taOSmd inspector" in html
        assert 'id="query"' in html
        assert 'id="searchResults"' in html
        assert 'id="pendingResults"' in html
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


def test_assets_served_with_correct_js_content_type(tmp_path, monkeypatch):
    """GET /assets/app.js returns Content-Type: text/javascript."""
    httpd = _make_server_with_webui(tmp_path, monkeypatch)
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        status, ctype, body = _get_raw(f"http://{host}:{port}/assets/app.js")
        assert status == 200, f"expected 200, got {status}"
        assert "javascript" in ctype
        assert "hello" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


def test_assets_served_with_correct_css_content_type(tmp_path, monkeypatch):
    """GET /assets/style.css returns Content-Type: text/css."""
    httpd = _make_server_with_webui(tmp_path, monkeypatch)
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        status, ctype, body = _get_raw(f"http://{host}:{port}/assets/style.css")
        assert status == 200, f"expected 200, got {status}"
        assert "css" in ctype
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


def test_unknown_path_serves_spa_index(tmp_path, monkeypatch):
    """Unknown GET paths serve index.html for SPA routing when webui is present."""
    httpd = _make_server_with_webui(tmp_path, monkeypatch)
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        status, ctype, html = _get_raw(f"http://{host}:{port}/some/deep/spa/path")
        assert status == 200
        assert "text/html" in ctype
        assert "test-spa" in html
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


def test_ui_alias_serves_html(live_server):
    """/ui always returns 200 HTML (SPA or fallback)."""
    status, ctype, html = _get_raw(f"{live_server}/ui")
    assert status == 200
    assert "text/html" in ctype
    assert "<html" in html.lower()


def test_ui_backing_endpoints_still_work(live_server):
    """The JSON endpoints the UI consumes (search + pending) still respond."""
    status, body = _post(
        f"{live_server}/ingest",
        {"text": "The inspection UI consumes the search endpoint.", "agent": "ui-test"},
    )
    assert status == 200, body

    status, body = _post(
        f"{live_server}/search",
        {"query": "The inspection UI consumes the search endpoint.", "agent": "ui-test"},
    )
    assert status == 200, body
    assert body["hits"]
    assert "inspection UI" in body["hits"][0]["text"]

    status, body = _get(f"{live_server}/pending?agent=ui-test")
    assert status == 200
    assert body["pending"] == []


# ----- A2A channels / members ---------------------------------------------


def test_a2a_channels_empty(live_server):
    status, body = _get(f"{live_server}/a2a/channels")
    assert status == 200
    assert "channels" in body
    assert isinstance(body["channels"], list)


def test_a2a_members_missing_channel_returns_400(live_server):
    status, body = _get(f"{live_server}/a2a/members")
    assert status == 400
    assert "channel" in body["error"]


def test_a2a_channels_populated(live_server):
    """After posting messages, channels lists the threads."""
    _post(f"{live_server}/a2a/send", {"from": "alice", "body": "hello", "thread": "general"})
    _post(f"{live_server}/a2a/send", {"from": "bob", "body": "world", "thread": "general"})
    _post(f"{live_server}/a2a/send", {"from": "alice", "body": "hi", "thread": "ops"})
    status, body = _get(f"{live_server}/a2a/channels")
    assert status == 200
    channels = {c["channel"] for c in body["channels"]}
    assert "general" in channels
    assert "ops" in channels
    general = next(c for c in body["channels"] if c["channel"] == "general")
    assert general["message_count"] == 2
    assert set(general["members"]) == {"alice", "bob"}


def test_a2a_members_populated(live_server):
    _post(f"{live_server}/a2a/send", {"from": "agent-x", "body": "ping", "thread": "chan1"})
    _post(f"{live_server}/a2a/send", {"from": "agent-y", "body": "pong", "thread": "chan1"})
    status, body = _get(f"{live_server}/a2a/members?channel=chan1")
    assert status == 200
    assert set(body["members"]) == {"agent-x", "agent-y"}


# ---------------------------------------------------------------------------
# POST /ingest/batch + ?mode=bm25 (#25 user-memory contract)
# ---------------------------------------------------------------------------

def test_ingest_batch_roundtrip_and_idempotent_reimport(live_server):
    items = [
        {"text": "Reverse a list in Python with slicing.",
         "id": "abc123", "metadata": {"collection": "snippets", "title": "Reverse"}},
        {"text": "The planning meeting moved to Thursday.",
         "id": "def456", "metadata": {"collection": "notes", "title": "Meeting"}},
    ]
    status, body = _post(
        f"{live_server}/ingest/batch", {"agent": "user-memory", "items": items},
    )
    assert status == 200
    assert body["ingested"] == 2
    assert body["skipped"] == 0

    # Re-POST of the same batch must skip everything on id (migration re-run).
    status, body = _post(
        f"{live_server}/ingest/batch", {"agent": "user-memory", "items": items},
    )
    assert status == 200
    assert body["ingested"] == 0
    assert body["skipped"] == 2

    # BM25-only search over the migrated chunks: GET with mode=bm25.
    status, body = _get(
        f"{live_server}/search?q=planning+meeting+Thursday&agent=user-memory&limit=10&mode=bm25"
    )
    assert status == 200
    assert body["hits"], "expected a bm25 hit for overlapping keywords"
    hit = body["hits"][0]
    assert "meeting" in hit["text"]
    assert set(hit.keys()) >= {"text", "source", "timestamp", "confidence", "metadata"}
    assert hit["metadata"].get("collection") == "notes"
    assert hit["metadata"].get("source_id") == "def456"

    # POST body form of the same search.
    status, body = _post(
        f"{live_server}/search",
        {"query": "reverse python list", "agent": "user-memory", "mode": "bm25"},
    )
    assert status == 200
    assert body["hits"]
    assert "Reverse" in body["hits"][0]["text"]


def test_ingest_batch_validation_errors(live_server):
    status, body = _post(f"{live_server}/ingest/batch", {"agent": "a"})
    assert status == 400
    assert "items" in body["error"]

    status, body = _post(f"{live_server}/ingest/batch", {"items": []})
    assert status == 400
    assert "agent" in body["error"]

    status, body = _post(
        f"{live_server}/ingest/batch",
        {"agent": "a", "items": [{"id": "no-text"}]},
    )
    assert status == 400
    assert "text" in body["error"]


def test_search_unknown_mode_is_400(live_server):
    status, body = _get(f"{live_server}/search?q=x&agent=a&mode=cosine9000")
    assert status == 400
    assert "unsupported search mode" in body["error"]


# ---------------------------------------------------------------------------
# Task graph HTTP tests
# ---------------------------------------------------------------------------


def test_task_create(live_server):
    """POST /tasks creates a task and returns the task object."""
    status, body = _post(
        f"{live_server}/tasks",
        {"title": "Implement feature X", "created_by": "agent-test"},
    )
    assert status == 200, body
    assert body["id"].startswith("t-")
    assert body["title"] == "Implement feature X"
    assert body["status"] == "open"
    assert body["created_by"] == "agent-test"


def test_task_create_missing_title(live_server):
    status, body = _post(
        f"{live_server}/tasks",
        {"created_by": "agent-test"},
    )
    assert status == 400
    assert "title" in body["error"]


def test_task_create_missing_created_by(live_server):
    status, body = _post(
        f"{live_server}/tasks",
        {"title": "Some task"},
    )
    assert status == 400
    assert "created_by" in body["error"]


def test_task_list(live_server):
    """GET /tasks returns a list of tasks."""
    _post(f"{live_server}/tasks", {"title": "T1", "created_by": "a"})
    _post(f"{live_server}/tasks", {"title": "T2", "created_by": "a"})
    status, body = _get(f"{live_server}/tasks")
    assert status == 200, body
    assert "tasks" in body
    assert len(body["tasks"]) >= 2


def test_task_list_status_filter(live_server):
    """GET /tasks?status= filters correctly."""
    r, created = _post(f"{live_server}/tasks", {"title": "T-filter", "created_by": "a"})
    assert r == 200
    _post(f"{live_server}/tasks/{created['id']}", {"status": "closed"})

    status, body = _get(f"{live_server}/tasks?status=closed")
    assert status == 200
    assert any(t["id"] == created["id"] for t in body["tasks"])


def test_task_ready(live_server):
    """GET /tasks/ready returns only open tasks with no active blockers."""
    _, t1 = _post(f"{live_server}/tasks", {"title": "Blocker", "created_by": "a"})
    _, t2 = _post(f"{live_server}/tasks", {"title": "Blocked", "created_by": "a"})
    # Add blocks edge
    _post(f"{live_server}/tasks/{t1['id']}/edges",
          {"to_id": t2["id"], "type": "blocks", "created_by": "a"})

    status, body = _get(f"{live_server}/tasks/ready")
    assert status == 200
    ready_ids = {t["id"] for t in body["tasks"]}
    assert t1["id"] in ready_ids
    assert t2["id"] not in ready_ids


def test_task_prime(live_server):
    """GET /tasks/prime returns a briefing with text and tasks keys."""
    _post(f"{live_server}/tasks", {"title": "Prime target", "created_by": "a"})
    status, body = _get(f"{live_server}/tasks/prime")
    assert status == 200
    assert "text" in body
    assert "tasks" in body
    assert isinstance(body["text"], str)
    assert isinstance(body["tasks"], list)


def test_task_update(live_server):
    """POST /tasks/{id} updates task fields."""
    _, task = _post(f"{live_server}/tasks", {"title": "To update", "created_by": "a"})
    status, body = _post(
        f"{live_server}/tasks/{task['id']}",
        {"status": "in_progress"},
    )
    assert status == 200
    assert body["status"] == "in_progress"


def test_task_update_bad_status(live_server):
    _, task = _post(f"{live_server}/tasks", {"title": "T", "created_by": "a"})
    status, body = _post(
        f"{live_server}/tasks/{task['id']}",
        {"status": "not_a_status"},
    )
    assert status == 400


def test_task_update_not_found(live_server):
    status, body = _post(
        f"{live_server}/tasks/t-000000000000",
        {"status": "closed"},
    )
    assert status == 400
    assert "not found" in body["error"]


def test_task_add_edge(live_server):
    """POST /tasks/{id}/edges adds a dependency edge."""
    _, t1 = _post(f"{live_server}/tasks", {"title": "Blocker", "created_by": "a"})
    _, t2 = _post(f"{live_server}/tasks", {"title": "Blocked", "created_by": "a"})
    status, body = _post(
        f"{live_server}/tasks/{t1['id']}/edges",
        {"to_id": t2["id"], "type": "blocks", "created_by": "a"},
    )
    assert status == 200
    assert body["from_id"] == t1["id"]
    assert body["to_id"] == t2["id"]
    assert body["type"] == "blocks"
    assert body["removed_ts"] is None


def test_task_add_edge_bad_type(live_server):
    _, t1 = _post(f"{live_server}/tasks", {"title": "T1", "created_by": "a"})
    _, t2 = _post(f"{live_server}/tasks", {"title": "T2", "created_by": "a"})
    status, body = _post(
        f"{live_server}/tasks/{t1['id']}/edges",
        {"to_id": t2["id"], "type": "badtype", "created_by": "a"},
    )
    assert status == 400


def test_task_remove_edge(live_server):
    """POST /tasks/{id}/edges/remove soft-removes a blocking edge."""
    _, t1 = _post(f"{live_server}/tasks", {"title": "Blocker", "created_by": "a"})
    _, t2 = _post(f"{live_server}/tasks", {"title": "Blocked", "created_by": "a"})
    _post(f"{live_server}/tasks/{t1['id']}/edges",
          {"to_id": t2["id"], "type": "blocks", "created_by": "a"})

    # t2 should not be in ready queue
    _, ready_body = _get(f"{live_server}/tasks/ready")
    assert t2["id"] not in {t["id"] for t in ready_body["tasks"]}

    # Remove the edge
    status, body = _post(
        f"{live_server}/tasks/{t1['id']}/edges/remove",
        {"to_id": t2["id"], "type": "blocks"},
    )
    assert status == 200
    assert body["removed_ts"] is not None

    # t2 should now be in ready queue
    _, ready_body = _get(f"{live_server}/tasks/ready")
    assert t2["id"] in {t["id"] for t in ready_body["tasks"]}


def test_task_remove_edge_not_found(live_server):
    _, t1 = _post(f"{live_server}/tasks", {"title": "T1", "created_by": "a"})
    _, t2 = _post(f"{live_server}/tasks", {"title": "T2", "created_by": "a"})
    status, body = _post(
        f"{live_server}/tasks/{t1['id']}/edges/remove",
        {"to_id": t2["id"], "type": "blocks"},
    )
    assert status == 400


def test_task_404_unknown_id(live_server):
    """POST /tasks/{unknown-non-t-id} should 404 when id segment is empty."""
    # The dispatch only routes /tasks/{id} where id is non-empty; an unknown
    # ID with valid format returns 400 from the update handler (task not found)
    status, body = _post(f"{live_server}/tasks/t-000000000000", {"status": "closed"})
    assert status == 400
    assert "not found" in body["error"]


# --- /controls: the memory-controls settings surface -------------------------


def test_controls_get_returns_settings_and_schema(live_server):
    status, body = _get(f"{live_server}/controls")
    assert status == 200, body
    assert set(body) >= {"settings", "schema"}
    settings = body["settings"]
    # prefer_verified ships on by default (tri-judge evidenced flip).
    assert settings["prefer_verified"] == "prefer_verified"
    schema = body["schema"]
    assert {c["id"] for c in schema["controls"]} >= {
        "prefer_verified", "reranker", "late_interaction", "fusion",
        "adjacent_turns", "embedder", "binary_quant", "self_verify",
    }
    assert {p["id"] for p in schema["presets"]} == {"minimal", "quality", "integrity"}
    # Every control carries the docs the README and UI render.
    for c in schema["controls"]:
        assert c["pros"] and c["cons"] and c["cost"] and c["description"]


def test_controls_post_single_runtime_control_roundtrips(live_server):
    status, body = _post(f"{live_server}/controls", {"prefer_verified": "off"})
    assert status == 200, body
    assert body["errors"] == {}
    assert body["settings"]["prefer_verified"] == "off"
    # Persisted: a fresh GET reflects it.
    _, after = _get(f"{live_server}/controls")
    assert after["settings"]["prefer_verified"] == "off"


def test_controls_post_preset_applies_bundle(live_server):
    status, body = _post(f"{live_server}/controls", {"preset": "integrity"})
    assert status == 200, body
    assert body["errors"] == {}
    s = body["settings"]
    assert s["prefer_verified"] == "prefer_verified"
    assert s["reranker"] == "bge-v2-m3"
    assert s["adjacent_turns"] == 2


def test_controls_post_unknown_preset_returns_400(live_server):
    status, body = _post(f"{live_server}/controls", {"preset": "nope"})
    assert status == 400
    assert "unknown preset" in body["error"]


def test_controls_post_bad_value_returns_400_with_errors(live_server):
    status, body = _post(f"{live_server}/controls", {"prefer_verified": "banana"})
    assert status == 400
    assert "prefer_verified" in body["errors"]
    # The good default is untouched after a rejected write.
    _, after = _get(f"{live_server}/controls")
    assert after["settings"]["prefer_verified"] == "prefer_verified"


def test_controls_post_store_scope_control_rejected(live_server):
    # embedder is store-scope (a re-index, not a live toggle); set_control refuses it.
    status, body = _post(f"{live_server}/controls", {"embedder": "minilm-onnx"})
    assert status == 400
    assert "embedder" in body["errors"]


def test_controls_post_empty_body_returns_400(live_server):
    status, body = _post(f"{live_server}/controls", {})
    assert status == 400
    assert "error" in body
