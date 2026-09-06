"""Tests for GET /a2a/threads principal= sender-derived filtering.

Offline + fast: the server runs in a background thread on an ephemeral port
with an isolated tmp data dir, and the vector embedder is patched so no ONNX
model is needed.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-a2a-threads"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        httpd.service_loop.close()


def _post(url: str, payload) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def test_principal_filters_to_sender_threads(live_server):
    """principal=alice returns only threads alice has sent to, not bob's.

    Also covers the absent-param regression (no principal returns all) and
    the unknown-principal case (empty list, 200).
    """
    _post(f"{live_server}/a2a/send", {"from": "alice", "body": "hello", "thread": "t1"})
    time.sleep(0.02)
    _post(f"{live_server}/a2a/send", {"from": "bob", "body": "world", "thread": "t2"})

    # Positive: alice sees only t1
    status, body = _get(f"{live_server}/a2a/threads?principal=alice")
    assert status == 200
    thread_names = {t["thread"] for t in body["threads"]}
    assert "t1" in thread_names
    assert "t2" not in thread_names

    # Negative control: absent param returns both threads, most-recently-active first
    status, body = _get(f"{live_server}/a2a/threads")
    assert status == 200
    all_names = [t["thread"] for t in body["threads"]]
    assert "t1" in all_names
    assert "t2" in all_names
    assert all_names == ["t2", "t1"]

    # Unknown principal returns empty list, 200
    status, body = _get(f"{live_server}/a2a/threads?principal=nobody-here")
    assert status == 200
    assert body["threads"] == []


def test_threads_empty_on_fresh_server(live_server):
    status, body = _get(f"{live_server}/a2a/threads")
    assert status == 200
    assert body["threads"] == []


def test_threads_ordered_desc_with_principal(live_server):
    """Filtered threads stay in most-recently-active order."""
    _post(f"{live_server}/a2a/send", {"from": "alice", "body": "old", "thread": "alpha"})
    time.sleep(0.02)
    _post(f"{live_server}/a2a/send", {"from": "alice", "body": "new", "thread": "beta"})

    status, body = _get(f"{live_server}/a2a/threads?principal=alice")
    assert status == 200
    names = [t["thread"] for t in body["threads"]]
    assert names == ["beta", "alpha"]


def test_threads_participants_include_sender(live_server):
    """participants list is built from the 'from' field of archived messages."""
    _post(f"{live_server}/a2a/send", {"from": "alice", "body": "hi", "thread": "general"})
    _post(f"{live_server}/a2a/send", {"from": "bob", "body": "there", "thread": "general"})

    status, body = _get(f"{live_server}/a2a/threads?principal=alice")
    assert status == 200
    assert len(body["threads"]) == 1
    assert set(body["threads"][0]["participants"]) == {"alice", "bob"}
