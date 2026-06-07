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


def test_unknown_route_returns_404(live_server):
    status, body = _get(f"{live_server}/does-not-exist")
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
