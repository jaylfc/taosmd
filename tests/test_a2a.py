"""Tests for the A2A (agent-to-agent) message bus.

Covers the service-layer round-trips, thread/since filtering, the HTTP
endpoints, SSE streaming, and secret redaction in A2A message bodies.
All tests are offline — no ONNX/QMD model needed; the vector embedder is
patched wherever stores are initialised.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server, service


# ---------------------------------------------------------------------------
# Helpers shared across tests
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
    data_dir = tmp_path / "taosmd-a2a"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _setup_stores(data_dir):
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """HTTP server on an ephemeral port against an isolated data dir."""
    data_dir = tmp_path / "taosmd-a2a-http"
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
    return _send(req)


def _get(url: str) -> tuple[int, dict]:
    return _send(urllib.request.Request(url, method="GET"))


def _send(req) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# Service-layer tests
# ---------------------------------------------------------------------------

def test_a2a_send_feed_roundtrip(isolated_data_dir):
    """send then feed returns the message oldest-first with correct fields."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    receipt = asyncio.run(service.a2a_send(
        "taOSmd", "hello from the memory layer",
        thread="dev", data_dir=dd,
    ))
    assert isinstance(receipt["id"], int) and receipt["id"] > 0
    assert receipt["from"] == "taOSmd"
    assert receipt["thread"] == "dev"
    assert receipt["reply_to"] is None

    msgs = asyncio.run(service.a2a_feed(thread="dev", data_dir=dd))
    assert len(msgs) == 1
    m = msgs[0]
    assert m["from"] == "taOSmd"
    assert m["body"] == "hello from the memory layer"
    assert m["thread"] == "dev"
    assert m["reply_to"] is None
    assert isinstance(m["ts"], float)


def test_a2a_feed_oldest_first(isolated_data_dir):
    """feed returns messages in chronological (oldest-first) order."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "first message", thread="order", data_dir=dd))
    time.sleep(0.01)
    asyncio.run(service.a2a_send("agentB", "second message", thread="order", data_dir=dd))

    msgs = asyncio.run(service.a2a_feed(thread="order", data_dir=dd))
    assert len(msgs) == 2
    assert msgs[0]["body"] == "first message"
    assert msgs[1]["body"] == "second message"
    assert msgs[0]["ts"] < msgs[1]["ts"]


def test_a2a_thread_filtering(isolated_data_dir):
    """Messages in thread A must not appear when querying thread B."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "alpha message", thread="alpha", data_dir=dd))
    asyncio.run(service.a2a_send("agentB", "beta message", thread="beta", data_dir=dd))

    alpha_msgs = asyncio.run(service.a2a_feed(thread="alpha", data_dir=dd))
    beta_msgs = asyncio.run(service.a2a_feed(thread="beta", data_dir=dd))

    assert len(alpha_msgs) == 1 and alpha_msgs[0]["body"] == "alpha message"
    assert len(beta_msgs) == 1 and beta_msgs[0]["body"] == "beta message"


def test_a2a_since_filtering(isolated_data_dir):
    """since= returns only messages newer than the given timestamp."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "old message", thread="since-test", data_dir=dd))
    time.sleep(0.02)
    pivot = time.time()
    time.sleep(0.02)
    asyncio.run(service.a2a_send("agentA", "new message", thread="since-test", data_dir=dd))

    msgs = asyncio.run(service.a2a_feed(thread="since-test", since=pivot, data_dir=dd))
    assert len(msgs) == 1
    assert msgs[0]["body"] == "new message"


def test_a2a_send_validates_empty_sender(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    with pytest.raises(ValueError, match="sender"):
        asyncio.run(service.a2a_send("", "body text", data_dir=str(isolated_data_dir)))


def test_a2a_send_validates_empty_body(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    with pytest.raises(ValueError, match="body"):
        asyncio.run(service.a2a_send("agentA", "", data_dir=str(isolated_data_dir)))


def test_a2a_reply_to_field_preserved(isolated_data_dir):
    """reply_to is stored and returned correctly."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    r1 = asyncio.run(service.a2a_send("agentA", "original", thread="replies", data_dir=dd))
    asyncio.run(service.a2a_send(
        "agentB", "reply here", thread="replies",
        reply_to=str(r1["id"]), data_dir=dd,
    ))

    msgs = asyncio.run(service.a2a_feed(thread="replies", data_dir=dd))
    assert msgs[1]["reply_to"] == str(r1["id"])


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

# Synthetic GitHub-style token: prefix + 36 alphanum chars assembled at
# runtime so no contiguous real-looking secret sits in source.
_GH_PREFIX = "ghp_"
_GH_BODY = "A" * 36


def test_a2a_body_secret_redacted(isolated_data_dir):
    """A body containing a secret token is redacted before storage."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)
    secret = _GH_PREFIX + _GH_BODY

    asyncio.run(service.a2a_send("agentA", f"token is {secret}", thread="sec", data_dir=dd))
    msgs = asyncio.run(service.a2a_feed(thread="sec", data_dir=dd))
    assert len(msgs) == 1
    assert secret not in msgs[0]["body"]
    assert "[REDACTED" in msgs[0]["body"]


# ---------------------------------------------------------------------------
# HTTP layer tests
# ---------------------------------------------------------------------------

def test_http_a2a_send_then_messages(live_server):
    """POST /a2a/send then GET /a2a/messages returns the message."""
    status, body = _post(
        f"{live_server}/a2a/send",
        {"from": "taOSmd", "body": "http round-trip test", "thread": "http-t"},
    )
    assert status == 200, body
    assert body["from"] == "taOSmd"
    assert body["thread"] == "http-t"
    assert isinstance(body["id"], int)

    status, body = _get(f"{live_server}/a2a/messages?thread=http-t")
    assert status == 200, body
    msgs = body["messages"]
    assert len(msgs) == 1
    assert msgs[0]["body"] == "http round-trip test"
    assert msgs[0]["from"] == "taOSmd"


def test_http_a2a_send_missing_from_returns_400(live_server):
    status, body = _post(
        f"{live_server}/a2a/send",
        {"body": "no sender here"},
    )
    assert status == 400
    assert "from" in body["error"]


def test_http_a2a_send_missing_body_returns_400(live_server):
    status, body = _post(
        f"{live_server}/a2a/send",
        {"from": "agentA"},
    )
    assert status == 400
    assert "body" in body["error"]


def test_http_a2a_messages_thread_filter(live_server):
    """GET /a2a/messages?thread= filters correctly."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "thread-x msg", "thread": "x"})
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "thread-y msg", "thread": "y"})

    status, body = _get(f"{live_server}/a2a/messages?thread=x")
    assert status == 200
    assert len(body["messages"]) == 1
    assert body["messages"][0]["body"] == "thread-x msg"


def test_http_a2a_messages_since_filter(live_server):
    """GET /a2a/messages?since= returns only newer messages."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "before pivot", "thread": "since-http"})
    time.sleep(0.02)
    pivot = time.time()
    time.sleep(0.02)
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "after pivot", "thread": "since-http"})

    status, body = _get(
        f"{live_server}/a2a/messages?thread=since-http&since={pivot}"
    )
    assert status == 200
    msgs = body["messages"]
    assert len(msgs) == 1
    assert msgs[0]["body"] == "after pivot"


# ---------------------------------------------------------------------------
# SSE smoke test
# ---------------------------------------------------------------------------

def _read_sse_frames(host: str, port: int, path: str, timeout: float = 8.0) -> list[str]:
    """Open a raw TCP connection, send a minimal HTTP GET, read lines until
    at least one ``data:`` frame arrives or timeout expires. Returns the list
    of ``data:`` payload strings received."""
    frames: list[str] = []
    deadline = time.monotonic() + timeout
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode()
        )
        buf = b""
        # Discard the HTTP headers section first.
        header_done = False
        while time.monotonic() < deadline and not frames:
            sock.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                chunk = sock.recv(4096)
            except (socket.timeout, TimeoutError):
                break
            if not chunk:
                break
            buf += chunk
            if not header_done:
                sep = buf.find(b"\r\n\r\n")
                if sep != -1:
                    buf = buf[sep + 4:]
                    header_done = True
            if header_done:
                # Parse SSE lines from what we have so far.
                text = buf.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("data: "):
                        frames.append(line[6:])
    return frames


def test_http_a2a_sse_delivers_message(live_server):
    """GET /a2a/stream receives a data: frame after a message is posted."""
    parsed = urllib.parse.urlsplit(live_server)
    host = parsed.hostname
    port = parsed.port

    thread = "sse-smoke"
    frames_received: list[str] = []
    error_holder: list[Exception] = []

    def _stream_reader():
        try:
            result = _read_sse_frames(
                host, port,
                f"/a2a/stream?thread={thread}",
                timeout=8.0,
            )
            frames_received.extend(result)
        except Exception as exc:
            error_holder.append(exc)

    reader = threading.Thread(target=_stream_reader, daemon=True)
    reader.start()

    # Give the SSE connection a moment to establish, then post a message.
    time.sleep(1.5)
    status, body = _post(
        live_server + "/a2a/send",
        {"from": "sse-sender", "body": "sse smoke payload", "thread": thread},
    )
    assert status == 200, f"send failed: {body}"

    reader.join(timeout=10)
    assert not error_holder, f"SSE reader raised: {error_holder[0]}"
    assert frames_received, "expected at least one SSE data: frame"
    payloads = [json.loads(f) for f in frames_received]
    bodies = [p.get("body") for p in payloads]
    assert "sse smoke payload" in bodies


def test_http_a2a_messages_fields_projection(live_server):
    """GET /a2a/messages?fields= projects each message to the named keys."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "compact me", "thread": "fields-t"})

    status, body = _get(
        f"{live_server}/a2a/messages?thread=fields-t&fields=id,from,body"
    )
    assert status == 200
    msgs = body["messages"]
    assert len(msgs) == 1
    assert set(msgs[0].keys()) == {"id", "from", "body"}
    assert msgs[0]["body"] == "compact me"

    # Unknown field names are ignored, never a 400.
    status, body = _get(
        f"{live_server}/a2a/messages?thread=fields-t&fields=body,nonexistent"
    )
    assert status == 200
    assert set(body["messages"][0].keys()) == {"body"}


def test_http_a2a_messages_ndjson_format(live_server):
    """GET /a2a/messages?format=ndjson emits one JSON object per line."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "line one", "thread": "nd-t"})
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "line two", "thread": "nd-t"})

    req = urllib.request.Request(
        f"{live_server}/a2a/messages?thread=nd-t&format=ndjson&fields=body",
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("application/x-ndjson")
        lines = resp.read().decode().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["body"] for p in parsed] == ["line one", "line two"]


def test_http_a2a_messages_bad_format_returns_400(live_server):
    status, body = _get(f"{live_server}/a2a/messages?format=xml")
    assert status == 400
    assert "format" in body["error"]
