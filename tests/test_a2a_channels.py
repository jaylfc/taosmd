"""Tests for A2A channel/member discovery, HTTP endpoints, and the setup guide.

Offline + fast: no ONNX/QMD model required; the vector embedder is patched
using the same deterministic hash helper used by the existing A2A tests.
The HTTP server is bound on port 0 (ephemeral) and torn down after each test.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import taosmd
from taosmd import api as taosmd_api
from taosmd import http_server, service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder — no ONNX/QMD model required."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Isolated data dir with a clean stores cache for each test."""
    data_dir = tmp_path / "taosmd-a2a-ch"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    yield data_dir
    for s in list(taosmd_api._stores_cache.values()):
        for store in (s.get("archive"), s.get("vector"), s.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """HTTP server on an ephemeral port against an isolated data dir."""
    data_dir = tmp_path / "taosmd-a2a-ch-http"
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


def _get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# service.a2a_channels
# ---------------------------------------------------------------------------

def test_a2a_channels_groups_by_thread(isolated_data_dir):
    """a2a_channels groups events by thread and returns one dict per channel."""
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("alice", "hi", thread="alpha", data_dir=dd))
    time.sleep(0.01)
    asyncio.run(service.a2a_send("bob", "hey", thread="alpha", data_dir=dd))
    asyncio.run(service.a2a_send("carol", "hello", thread="beta", data_dir=dd))

    channels = asyncio.run(service.a2a_channels(data_dir=dd))
    names = [c["channel"] for c in channels]
    assert "alpha" in names
    assert "beta" in names


def test_a2a_channels_members_sorted(isolated_data_dir):
    """Members list is sorted and unique even when a sender posts multiple times."""
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("zara", "msg1", thread="gamma", data_dir=dd))
    asyncio.run(service.a2a_send("alice", "msg2", thread="gamma", data_dir=dd))
    asyncio.run(service.a2a_send("zara", "msg3", thread="gamma", data_dir=dd))

    channels = asyncio.run(service.a2a_channels(data_dir=dd))
    gamma = next(c for c in channels if c["channel"] == "gamma")
    assert gamma["members"] == sorted(set(gamma["members"]))
    assert gamma["members"] == ["alice", "zara"]


def test_a2a_channels_message_count(isolated_data_dir):
    """message_count matches the number of messages sent to that thread."""
    dd = str(isolated_data_dir)

    for i in range(3):
        asyncio.run(service.a2a_send("agentA", f"msg{i}", thread="count-ch", data_dir=dd))

    channels = asyncio.run(service.a2a_channels(data_dir=dd))
    ch = next(c for c in channels if c["channel"] == "count-ch")
    assert ch["message_count"] == 3


def test_a2a_channels_created_and_last_ts(isolated_data_dir):
    """created_ts <= last_ts and both are floats."""
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "first", thread="ts-ch", data_dir=dd))
    time.sleep(0.02)
    asyncio.run(service.a2a_send("agentB", "second", thread="ts-ch", data_dir=dd))

    channels = asyncio.run(service.a2a_channels(data_dir=dd))
    ch = next(c for c in channels if c["channel"] == "ts-ch")
    assert isinstance(ch["created_ts"], float)
    assert isinstance(ch["last_ts"], float)
    assert ch["created_ts"] <= ch["last_ts"]


def test_a2a_channels_sorted_by_last_ts_desc(isolated_data_dir):
    """Channels are returned most-recently-active first."""
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("a", "old", thread="older-ch", data_dir=dd))
    time.sleep(0.02)
    asyncio.run(service.a2a_send("b", "new", thread="newer-ch", data_dir=dd))

    channels = asyncio.run(service.a2a_channels(data_dir=dd))
    names = [c["channel"] for c in channels]
    assert names.index("newer-ch") < names.index("older-ch")


def test_a2a_channels_empty_store(isolated_data_dir):
    """a2a_channels returns an empty list when no messages have been sent."""
    channels = asyncio.run(service.a2a_channels(data_dir=str(isolated_data_dir)))
    assert channels == []


# ---------------------------------------------------------------------------
# service.a2a_members
# ---------------------------------------------------------------------------

def test_a2a_members_distinct_senders(isolated_data_dir):
    """a2a_members returns sorted distinct senders on that channel."""
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("carol", "hi", thread="mem-ch", data_dir=dd))
    asyncio.run(service.a2a_send("alice", "hey", thread="mem-ch", data_dir=dd))
    asyncio.run(service.a2a_send("carol", "again", thread="mem-ch", data_dir=dd))

    members = asyncio.run(service.a2a_members(channel="mem-ch", data_dir=dd))
    assert members == ["alice", "carol"]


def test_a2a_members_ignores_other_channels(isolated_data_dir):
    """a2a_members only counts senders on the requested channel."""
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("only-here", "msg", thread="chan-a", data_dir=dd))
    asyncio.run(service.a2a_send("other-agent", "msg", thread="chan-b", data_dir=dd))

    members = asyncio.run(service.a2a_members(channel="chan-a", data_dir=dd))
    assert members == ["only-here"]
    assert "other-agent" not in members


def test_a2a_members_unknown_channel_returns_empty(isolated_data_dir):
    """a2a_members returns [] for a channel with no messages."""
    members = asyncio.run(
        service.a2a_members(channel="nonexistent", data_dir=str(isolated_data_dir))
    )
    assert members == []


# ---------------------------------------------------------------------------
# HTTP layer — GET /a2a/channels
# ---------------------------------------------------------------------------

def test_http_a2a_channels_empty(live_server):
    """GET /a2a/channels returns an empty list when no messages exist."""
    status, body = _get(f"{live_server}/a2a/channels")
    assert status == 200, body
    assert body == {"channels": []}


def _http_post(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def test_http_a2a_channels_after_sends(live_server):
    """GET /a2a/channels returns correct channel entries after posting."""
    _http_post(f"{live_server}/a2a/send",
               {"from": "alice", "body": "hello", "thread": "proj-a"})
    _http_post(f"{live_server}/a2a/send",
               {"from": "bob", "body": "hi", "thread": "proj-a"})
    _http_post(f"{live_server}/a2a/send",
               {"from": "carol", "body": "hey", "thread": "proj-b"})

    status, body = _get(f"{live_server}/a2a/channels")
    assert status == 200, body
    channels = body["channels"]
    names = [c["channel"] for c in channels]
    assert "proj-a" in names
    assert "proj-b" in names

    pa = next(c for c in channels if c["channel"] == "proj-a")
    assert pa["message_count"] == 2
    assert sorted(pa["members"]) == ["alice", "bob"]


# ---------------------------------------------------------------------------
# HTTP layer — GET /a2a/members
# ---------------------------------------------------------------------------

def test_http_a2a_members_returns_members(live_server):
    """GET /a2a/members?channel= returns sorted member list."""
    _http_post(f"{live_server}/a2a/send", {"from": "zara", "body": "yo", "thread": "m-ch"})
    _http_post(f"{live_server}/a2a/send", {"from": "alice", "body": "hi", "thread": "m-ch"})

    status, body = _get(f"{live_server}/a2a/members?channel=m-ch")
    assert status == 200, body
    assert body["members"] == ["alice", "zara"]


def test_http_a2a_members_missing_channel_returns_400(live_server):
    """GET /a2a/members without ?channel= returns 400."""
    status, body = _get(f"{live_server}/a2a/members")
    assert status == 400
    assert "channel" in body["error"]


def test_http_a2a_members_unknown_channel_empty(live_server):
    """GET /a2a/members for an unknown channel returns empty list."""
    status, body = _get(f"{live_server}/a2a/members?channel=ghost")
    assert status == 200, body
    assert body["members"] == []


# ---------------------------------------------------------------------------
# a2a_setup_guide()
# ---------------------------------------------------------------------------

def test_a2a_setup_guide_nonempty():
    """a2a_setup_guide() returns non-empty text."""
    text = taosmd.a2a_setup_guide()
    assert isinstance(text, str) and text.strip()


def test_a2a_setup_guide_contains_install_step():
    """Guide mentions the install check."""
    text = taosmd.a2a_setup_guide()
    assert "taosmd --version" in text or "pip install" in text


def test_a2a_setup_guide_contains_running_instance_check():
    """Guide mentions checking for an already-running instance."""
    text = taosmd.a2a_setup_guide()
    assert "already" in text.lower() or "running" in text.lower()


def test_a2a_setup_guide_contains_channel_step():
    """Guide mentions channels."""
    text = taosmd.a2a_setup_guide()
    assert "channel" in text.lower()


def test_a2a_setup_guide_contains_join_step():
    """Guide mentions a JOIN message."""
    text = taosmd.a2a_setup_guide()
    assert "JOIN" in text or "join" in text.lower()
