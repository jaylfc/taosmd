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
