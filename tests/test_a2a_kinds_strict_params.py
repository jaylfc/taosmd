"""Tests for A2A v2 slice 1: kind taxonomy + strict params on every /a2a endpoint.

Covers:
- POST /a2a/send accepts kind and returns it in the envelope
- kind round-trips through GET /a2a/messages and SSE /a2a/stream
- Every /a2a GET endpoint rejects unknown query params with 400
- One-shot migration tags historical messages by body-prefix conventions
  and is idempotent
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server, service
from taosmd.archive import EVENT_A2A


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-a2a-v2"
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
    data_dir = tmp_path / "taosmd-a2a-v2-http"
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
# Gate (a): unknown query param on /a2a/messages must 400
# ---------------------------------------------------------------------------

def test_http_a2a_messages_unknown_param_returns_400(live_server):
    """Unknown query params on /a2a/messages must 400, not silently page."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "msg", "thread": "strict-test"})
    status, body = _get(f"{live_server}/a2a/messages?thread=strict-test&bogus_param=1")
    assert status == 400, body
    assert "bogus_param" in body["error"]


# ---------------------------------------------------------------------------
# Gate (b): kind round-trips send -> messages -> stream
# ---------------------------------------------------------------------------

def test_http_a2a_send_kind_default_chat(live_server):
    """Default kind is chat and is returned in the send receipt."""
    status, body = _post(
        f"{live_server}/a2a/send",
        {"from": "agentA", "body": "chat msg", "thread": "kind-test"},
    )
    assert status == 200, body
    assert body.get("kind") == "chat"


def test_http_a2a_send_kind_alarm_roundtrips(live_server):
    """POST with kind=alarm -> GET /a2a/messages returns kind=alarm."""
    status, body = _post(
        f"{live_server}/a2a/send",
        {"from": "agentA", "body": "alarm msg", "thread": "kind-alarm", "kind": "alarm"},
    )
    assert status == 200, body
    assert body["kind"] == "alarm"

    status, body = _get(f"{live_server}/a2a/messages?thread=kind-alarm")
    assert status == 200, body
    msgs = body["messages"]
    assert len(msgs) == 1
    assert msgs[0]["kind"] == "alarm"


def test_http_a2a_send_kind_system_roundtrips(live_server):
    """POST with kind=system -> GET /a2a/messages returns kind=system."""
    status, body = _post(
        f"{live_server}/a2a/send",
        {"from": "sys", "body": "sys msg", "thread": "kind-sys", "kind": "system"},
    )
    assert status == 200, body
    assert body["kind"] == "system"

    status, body = _get(f"{live_server}/a2a/messages?thread=kind-sys")
    assert status == 200, body
    assert body["messages"][0]["kind"] == "system"


def test_http_a2a_send_empty_kind_returns_400(live_server):
    """An explicit empty-string kind is invalid, never silently coerced to chat."""
    status, body = _post(
        f"{live_server}/a2a/send",
        {"from": "agentA", "body": "msg", "thread": "kind-empty", "kind": ""},
    )
    assert status == 400, body
    assert "kind" in body["error"].lower()


def test_http_a2a_send_unknown_kind_returns_400(live_server):
    """POST /a2a/send with an unknown kind must 400 naming the allowed set."""
    status, body = _post(
        f"{live_server}/a2a/send",
        {"from": "agentA", "body": "msg", "thread": "kind-bad", "kind": "teleport"},
    )
    assert status == 400, body
    err = body["error"].lower()
    assert "chat" in err and "alarm" in err


def test_http_a2a_stream_includes_kind(live_server):
    """SSE frames include the kind field."""
    parsed = urllib.parse.urlsplit(live_server)
    host = parsed.hostname
    port = parsed.port
    thread = "sse-kind"
    frames_received = []
    error_holder = []

    def _stream_reader():
        try:
            result = _read_sse_frames(host, port, f"/a2a/stream?thread={thread}", timeout=8.0)
            frames_received.extend(result)
        except Exception as exc:
            error_holder.append(exc)

    reader = threading.Thread(target=_stream_reader, daemon=True)
    reader.start()
    time.sleep(1.5)

    status, body = _post(
        live_server + "/a2a/send",
        {"from": "sse-sender", "body": "sse kind payload", "thread": thread, "kind": "review"},
    )
    assert status == 200, f"send failed: {body}"

    reader.join(timeout=10)
    assert not error_holder, f"SSE reader raised: {error_holder[0]}"
    assert frames_received, "expected at least one SSE data: frame"
    payloads = [json.loads(f) for f in frames_received]
    matching = [p for p in payloads if p.get("body") == "sse kind payload"]
    assert matching, "expected the posted message in SSE frames"
    assert matching[0]["kind"] == "review"


# ---------------------------------------------------------------------------
# Gate (c): one-shot migration tags historical messages by body prefix
# ---------------------------------------------------------------------------

def _seed_historical_message(stores, body, thread="migrate-test"):
    """Write a raw A2A event directly to the archive without kind."""
    archive = stores["archive"]
    data = {"from": "hist", "body": body, "thread": thread}
    row_id = asyncio.run(archive.record(
        event_type=EVENT_A2A,
        data=data,
        agent_name="hist",
        app_id=thread,
        summary=body[:200],
    ))
    return row_id


def test_a2a_migrate_kinds_idempotent_and_correct(isolated_data_dir):
    """Migration tags [AUTOMATED, [AUTO-ACK], [REVIEW] prefixes and is idempotent."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    _seed_historical_message(_setup_stores(isolated_data_dir), "[AUTOMATED check complete]", "mig1")
    _seed_historical_message(_setup_stores(isolated_data_dir), "[AUTO-ACK] received", "mig2")
    _seed_historical_message(_setup_stores(isolated_data_dir), "[REVIEW] please check", "mig3")
    _seed_historical_message(_setup_stores(isolated_data_dir), "plain chat message", "mig4")

    result = asyncio.run(service.a2a_migrate_kinds(data_dir=dd))
    assert result["migrated"] == 4
    assert result["alarm"] == 1
    assert result["ack"] == 1
    assert result["review"] == 1
    assert result["chat"] == 1

    msgs = asyncio.run(service.a2a_feed(thread="mig1", data_dir=dd))
    assert msgs[0]["kind"] == "alarm"
    msgs = asyncio.run(service.a2a_feed(thread="mig2", data_dir=dd))
    assert msgs[0]["kind"] == "ack"
    msgs = asyncio.run(service.a2a_feed(thread="mig3", data_dir=dd))
    assert msgs[0]["kind"] == "review"
    msgs = asyncio.run(service.a2a_feed(thread="mig4", data_dir=dd))
    assert msgs[0]["kind"] == "chat"

    result2 = asyncio.run(service.a2a_migrate_kinds(data_dir=dd))
    assert result2["migrated"] == 0


# ---------------------------------------------------------------------------
# Strict params on other /a2a GET endpoints
# ---------------------------------------------------------------------------

def test_http_a2a_mentions_unknown_param_returns_400(live_server):
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "msg", "thread": "mention-test"})
    status, body = _get(f"{live_server}/a2a/mentions?reader=agentA&bogus=1")
    assert status == 400, body
    assert "bogus" in body["error"]


def test_http_a2a_stream_unknown_param_returns_400(live_server):
    status = _stream_status_code(live_server, "/a2a/stream?thread=any&bogus=1")
    assert status == 400


def test_http_a2a_threads_unknown_param_returns_400(live_server):
    status, body = _get(f"{live_server}/a2a/threads?principal=agentA&bogus=1")
    assert status == 400, body
    assert "bogus" in body["error"]


def test_http_a2a_thread_messages_unknown_param_returns_400(live_server):
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "msg", "thread": "tm-test"})
    status, body = _get(f"{live_server}/a2a/threads/tm-test/messages?bogus=1")
    assert status == 400, body
    assert "bogus" in body["error"]


def test_http_a2a_channels_unknown_param_returns_400(live_server):
    status, body = _get(f"{live_server}/a2a/channels?bogus=1")
    assert status == 400, body
    assert "bogus" in body["error"]


def test_http_a2a_members_unknown_param_returns_400(live_server):
    status, body = _get(f"{live_server}/a2a/members?channel=general&bogus=1")
    assert status == 400, body
    assert "bogus" in body["error"]


# ---------------------------------------------------------------------------
# Cursor-shaped params that were silently dropped before strict validation:
# after= and since_id= are NOT accepted on the feed/stream/messages endpoints
# (they live on /a2a/threads/{thread}/messages, which is a different surface).
# ---------------------------------------------------------------------------

def test_http_a2a_messages_after_rejected_returns_400(live_server):
    """after= is not a recognised param on /a2a/messages; must 400, not page."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "hi", "thread": "after-t"})
    status, body = _get(
        f"{live_server}/a2a/messages?thread=after-t&after=1"
    )
    assert status == 400, body
    assert "after" in body["error"]


def test_http_a2a_messages_since_id_rejected_returns_400(live_server):
    """since_id= is not a recognised param on /a2a/messages; must 400, not page."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "hi", "thread": "since-id-t"})
    status, body = _get(
        f"{live_server}/a2a/messages?thread=since-id-t&since_id=1"
    )
    assert status == 400, body
    assert "since_id" in body["error"]


def test_http_a2a_stream_after_rejected_returns_400(live_server):
    """after= is not accepted on /a2a/stream; must 400."""
    status = _stream_status_code(
        live_server, "/a2a/stream?thread=any&after=1"
    )
    assert status == 400


def test_http_a2a_stream_since_id_rejected_returns_400(live_server):
    """since_id= is not accepted on /a2a/stream; must 400."""
    status = _stream_status_code(
        live_server, "/a2a/stream?thread=any&since_id=1"
    )
    assert status == 400


def test_http_a2a_mentions_after_rejected_returns_400(live_server):
    """after= is not accepted on /a2a/mentions (the feed path); must 400."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "hi", "thread": "mention-after"})
    status, body = _get(
        f"{live_server}/a2a/mentions?reader=agentA&after=1"
    )
    assert status == 400, body
    assert "after" in body["error"]


def test_http_a2a_mentions_since_id_rejected_returns_400(live_server):
    """since_id= is not accepted on /a2a/mentions; must 400."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "hi", "thread": "mention-since-id"})
    status, body = _get(
        f"{live_server}/a2a/mentions?reader=agentA&since_id=1"
    )
    assert status == 400, body
    assert "since_id" in body["error"]


# ---------------------------------------------------------------------------
# Error shape: names both the offending param and the accepted set
# ---------------------------------------------------------------------------

def test_http_a2a_messages_error_names_accepted_set(live_server):
    """The 400 error must name the offending parameter AND the accepted set."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "msg", "thread": "nameset"})
    status, body = _get(
        f"{live_server}/a2a/messages?thread=nameset&after=1"
    )
    assert status == 400, body
    err = body["error"]
    assert "after" in err
    assert "thread" in err and "since" in err and "limit" in err
    assert "fields" in err and "format" in err


def test_http_a2a_mentions_error_names_accepted_set(live_server):
    """The 400 error on /a2a/mentions names both the offending param and accepted set."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "msg", "thread": "nameset-m"})
    status, body = _get(
        f"{live_server}/a2a/mentions?reader=agentA&bogus=1"
    )
    assert status == 400, body
    err = body["error"]
    assert "bogus" in err
    assert "reader" in err and "since" in err and "limit" in err


# ---------------------------------------------------------------------------
# Positive cases: every accepted param still works; no params still works
# ---------------------------------------------------------------------------

def test_http_a2a_messages_no_params_returns_200(live_server):
    """A plain GET /a2a/messages with no query params must still work (200)."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "no-param msg", "thread": "noparams"})
    status, body = _get(f"{live_server}/a2a/messages")
    assert status == 200, body
    assert "messages" in body


def test_http_a2a_messages_all_accepted_params_return_200(live_server):
    """All accepted params together on /a2a/messages must return 200, not 400."""
    _post(f"{live_server}/a2a/send",
          {"from": "agentA", "body": "allparams", "thread": "allparams"})
    ts = time.time() - 1
    status, body = _get(
        f"{live_server}/a2a/messages"
        f"?thread=allparams&since={ts}&limit=10&fields=id,from,body&format=json"
    )
    assert status == 200, body
    msgs = body["messages"]
    assert len(msgs) == 1
    assert msgs[0]["body"] == "allparams"
    assert set(msgs[0].keys()) == {"id", "from", "body"}


def test_http_a2a_threads_no_params_returns_200(live_server):
    """A plain GET /a2a/threads with no query params must still work (200)."""
    status, body = _get(f"{live_server}/a2a/threads")
    assert status == 200, body
    assert "threads" in body


def test_http_a2a_channels_no_params_returns_200(live_server):
    """A plain GET /a2a/channels with no query params must still work (200)."""
    status, body = _get(f"{live_server}/a2a/channels")
    assert status == 200, body
    assert "channels" in body


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _read_sse_frames(host: str, port: int, path: str, timeout: float = 8.0) -> list[str]:
    frames: list[str] = []
    deadline = time.monotonic() + timeout
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode()
        )
        buf = b""
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
                text = buf.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("data: "):
                        frames.append(line[6:])
    return frames


def _stream_status_code(live_server: str, path: str, timeout: float = 3.0) -> int:
    parsed = urllib.parse.urlsplit(live_server)
    host = parsed.hostname
    port = parsed.port
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode()
        )
        sock.settimeout(timeout)
        line = b""
        while b"\r\n" not in line:
            try:
                chunk = sock.recv(128)
            except (socket.timeout, TimeoutError) as exc:
                raise AssertionError(f"timed out reading stream status line: {exc}") from exc
            if not chunk:
                break
            line += chunk
    status_line = line.decode("utf-8", "replace").splitlines()[0]
    return int(status_line.split()[1])
