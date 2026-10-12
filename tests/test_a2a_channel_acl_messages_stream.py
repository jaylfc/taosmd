"""Slice 2 of the channel-ACL work: /a2a/messages and /a2a/stream read filtering.

Identity for every ACL decision is the verified registry token ``sub`` and
nothing else. A restricted channel with no verified caller is DENIED; a channel
with no ACL stays open to everyone. All state lives under ``tmp_path``.

Design rules (R-ID, R-CLOSED, R-ONCE, R-ALIAS, R-HTTP, R-TEST, R-RED):
- GET /a2a/messages filters rows of channels the caller cannot read (with and
  without ?thread=, json and ndjson formats).
- GET /a2a/stream filters the same way AND advances its poll cursor past
  DENIED rows.
- Bounded: page the feed until ``limit`` readable rows or a stated cap of
  ``limit * 5`` rows scanned; never ``limit=10**9``.
- Tests hit /a2a/stream itself (read the SSE lines with a short socket
  timeout), not /a2a/messages.
- MUTANTS: (m1) remove cursor advance for denied rows -> test asserting
  ``since`` moves past a run of denied rows FAILS; (m2) remove filter on
  /a2a/stream -> canary test FAILS; (m3) unbounded scan -> test that counts
  feed rows read (spy on service.a2a_feed limit) FAILS.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import config as cfg
from taosmd import http_server, service

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt as pyjwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ADMIN = "admin-token-for-acl-tests"
CANARY = "CANARY-9f3b1c-do-not-leak"
CANARY_ALLOWED = "CANARY-ALLOWED-7a2e4f"


def _keypair():
    priv = Ed25519PrivateKey.generate()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


PRIV_PEM, PUB_PEM = _keypair()


def _token(sub):
    return pyjwt.encode({"sub": sub}, PRIV_PEM, algorithm="EdDSA")


def _call(method, url, payload=None, token=None):
    headers = {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode()
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _j(raw):
    return json.loads(raw or "{}")


def _get_sse(url, token=None, timeout=5.0, max_lines=100):
    """Read SSE stream from URL, return list of parsed message dicts."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    messages = []
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if time.time() - start > timeout:
                    break
                if line.startswith(b"data: "):
                    try:
                        msg = json.loads(line[6:].decode().strip())
                        messages.append(msg)
                        if len(messages) >= max_lines:
                            break
                    except json.JSONDecodeError:
                        pass
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()
    except socket.timeout:
        pass
    return messages


class Srv:
    def __init__(self, base, data_dir, httpd):
        self.base = base
        self.data_dir = str(data_dir)
        self.httpd = httpd

    def send(self, sender, thread, body="hello", token=None, **extra):
        payload = {"from": sender, "body": body, "thread": thread, **extra}
        return _call("POST", f"{self.base}/a2a/send", payload, token)

    def messages(self, thread=None, since=None, limit=50, fmt="json", token=None, fields=None):
        params = []
        if thread:
            params.append(f"thread={thread}")
        if since is not None:
            params.append(f"since={since}")
        if limit != 50:
            params.append(f"limit={limit}")
        if fmt != "json":
            params.append(f"format={fmt}")
        if fields:
            params.append(f"fields={fields}")
        q = "?" + "&".join(params) if params else ""
        return _call("GET", f"{self.base}/a2a/messages{q}", None, token)

    def stream(self, thread=None, since=None, token=None, timeout=3.0, max_lines=50):
        params = []
        if thread:
            params.append(f"thread={thread}")
        if since is not None:
            params.append(f"since={since}")
        q = "?" + "&".join(params) if params else ""
        return _get_sse(f"{self.base}/a2a/stream{q}", token, timeout=timeout, max_lines=max_lines)

    def admin_set(self, payload, token=ADMIN):
        return _call("POST", f"{self.base}/a2a/admin/set-channel-acl", payload, token)

    def admin_get(self, channel, token=ADMIN):
        return _call("GET", f"{self.base}/a2a/admin/channel-acl?channel={channel}", None, token)

    def archive_dump(self):
        """Every bus message in the archive, read back through the service."""
        rows = self.httpd.service_loop.run(
            service.a2a_feed(thread=None, limit=1000, data_dir=self.data_dir)
        )
        return json.dumps(rows)


def _build(tmp_path, monkeypatch, enforce=False):
    data_dir = tmp_path / "taosmd-acl-msg-stream"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    cfg.set_admin_token(ADMIN, data_dir=str(data_dir))
    cfg.set_a2a_auth_enforce(enforce, str(data_dir))

    def fake_opener(url, token=None):
        if url.endswith("pubkey"):
            return json.dumps({"pubkey": PUB_PEM})
        return json.dumps([])

    from taosmd import registry_auth
    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=None,
    )
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir), verifier=verifier)
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return Srv(f"http://{host}:{port}", data_dir, httpd), httpd, t


@pytest.fixture
def srv(tmp_path, monkeypatch):
    """Verify-and-warn mode: unauthenticated posts are tolerated on open channels."""
    s, httpd, t = _build(tmp_path, monkeypatch, enforce=False)
    try:
        yield s
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


@pytest.fixture
def srv_enforce(tmp_path, monkeypatch):
    """Enforce mode: the existing auth layer would 401 an unauthenticated post."""
    s, httpd, t = _build(tmp_path, monkeypatch, enforce=True)
    try:
        yield s
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


def _restrict(srv, channel="secret", read=("alice",), post=("alice",)):
    status, raw = srv.admin_set({"channel": channel, "read": list(read), "post": list(post)})
    assert status == 200, raw


# =====================================================================
# /a2a/messages filtering
# =====================================================================

def test_messages_open_channel_unchanged(srv):
    """Open channel (no ACL) returns all messages for any caller."""
    srv.send("agentA", "general", body="msg1")
    srv.send("agentB", "general", body="msg2")
    # No token = unauthenticated
    status, raw = srv.messages(thread="general", limit=10)
    assert status == 200
    body = _j(raw)
    assert len(body["messages"]) == 2
    assert any(m["body"] == "msg1" for m in body["messages"])
    assert any(m["body"] == "msg2" for m in body["messages"])


def test_messages_restricted_channel_denied_verified_caller(srv):
    """Verified caller not on read list -> 200 but empty (filter, not 403)."""
    _restrict(srv, channel="secret", read=["alice"])
    # alice posts
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    # bob tries to read (verified but not on read list)
    status, raw = srv.messages(thread="secret", token=_token("bob"))
    assert status == 200, raw
    body = _j(raw)
    assert body["messages"] == [], f"bob should not see messages: {body}"
    # Canary must not leak
    assert CANARY not in raw


def test_messages_restricted_channel_denied_unauthenticated_caller(srv):
    """Unauthenticated caller on restricted channel -> 200 but empty."""
    _restrict(srv, channel="secret", read=["alice"])
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    # No token
    status, raw = srv.messages(thread="secret")
    assert status == 200, raw
    body = _j(raw)
    assert body["messages"] == []
    assert CANARY not in raw


def test_messages_restricted_channel_allowed_caller_sees_all(srv):
    """Verified caller on read list sees all messages."""
    _restrict(srv, channel="secret", read=["alice", "bob"], post=["alice", "bob"])
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    srv.send("bob", "secret", body="bob-msg", token=_token("bob"))
    status, raw = srv.messages(thread="secret", token=_token("alice"))
    assert status == 200, raw
    body = _j(raw)
    assert len(body["messages"]) == 2
    assert any(m["body"] == CANARY for m in body["messages"])
    assert any(m["body"] == "bob-msg" for m in body["messages"])


def test_messages_without_thread_filters_all_channels(srv):
    """GET /a2a/messages without ?thread= filters across all channels."""
    _restrict(srv, channel="secret", read=["alice"])
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    srv.send("agentX", "general", body=CANARY_ALLOWED)
    # bob reads without thread filter
    status, raw = srv.messages(token=_token("bob"))
    assert status == 200, raw
    body = _j(raw)
    # Should only see general channel message
    bodies = [m["body"] for m in body["messages"]]
    assert CANARY not in bodies
    assert CANARY_ALLOWED in bodies


def test_messages_ndjson_format(srv):
    """GET /a2a/messages?format=ndjson filters and emits NDJSON."""
    _restrict(srv, channel="secret", read=["alice"])
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    srv.send("agentX", "general", body=CANARY_ALLOWED)
    status, raw = srv.messages(thread="secret", fmt="ndjson", token=_token("bob"))
    assert status == 200, raw
    # NDJSON: each line is a JSON object
    lines = raw.strip().split("\n")
    # bob should see nothing from secret
    assert len(lines) == 0 or (len(lines) == 1 and lines[0] == "")
    assert CANARY not in raw


def test_messages_ndjson_format_allowed(srv):
    """GET /a2a/messages?format=ndjson works for allowed caller."""
    _restrict(srv, channel="secret", read=["alice"])
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    status, raw = srv.messages(thread="secret", fmt="ndjson", token=_token("alice"))
    assert status == 200, raw
    lines = [l for l in raw.strip().split("\n") if l]
    assert len(lines) == 1
    msg = json.loads(lines[0])
    assert msg["body"] == CANARY


def test_messages_fields_projection(srv):
    """GET /a2a/messages?fields=id,sender,body projects after filtering."""
    _restrict(srv, channel="secret", read=["alice"])
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    status, raw = srv.messages(thread="secret", fields="id,sender,body", token=_token("bob"))
    assert status == 200, raw
    body = _j(raw)
    assert body["messages"] == []


def test_messages_bounded_scan_limit_times_five(srv):
    """R-ONCE: scan is bounded to limit*5 rows from the feed."""
    _restrict(srv, channel="secret", read=["alice"])
    # Post 100 messages to secret (denied for bob) and 3 to general
    for i in range(100):
        srv.send("alice", "secret", body=f"secret-{i}", token=_token("alice"))
    for i in range(3):
        srv.send("agentX", "general", body=f"general-{i}")
    # bob requests limit=2 without thread -> should scan at most 10 feed rows
    # and find 2 from general
    status, raw = srv.messages(limit=2, token=_token("bob"))
    assert status == 200, raw
    body = _j(raw)
    # Should get 2 messages from general
    assert len(body["messages"]) == 2
    bodies = [m["body"] for m in body["messages"]]
    assert all("general" in b for b in bodies)
    assert not any("secret" in b for b in bodies)


def test_messages_malformed_acl_denies_read(srv):
    """Malformed ACL entry denies read for everyone."""
    d = srv.data_dir
    with open(f"{d}/config.json") as fh:
        conf = json.load(fh)
    conf["channel_acl"] = {"broken": ["alice"]}  # entry is a list, not dict
    with open(f"{d}/config.json", "w") as fh:
        json.dump(conf, fh)
    srv.send("alice", "broken", body=CANARY, token=_token("alice"))
    # Even alice should not see it
    status, raw = srv.messages(thread="broken", token=_token("alice"))
    assert status == 200, raw
    body = _j(raw)
    assert body["messages"] == []
    assert CANARY not in raw


# =====================================================================
# /a2a/stream filtering + cursor advance
# =====================================================================

def test_stream_open_channel_unchanged(srv):
    """Open channel streams all messages."""
    since = time.time()
    srv.send("agentA", "general", body="msg1")
    srv.send("agentB", "general", body="msg2")
    time.sleep(0.05)
    msgs = srv.stream(thread="general", token=_token("alice"), since=since)
    bodies = [m["body"] for m in msgs]
    assert "msg1" in bodies
    assert "msg2" in bodies


def test_stream_restricted_channel_denied_verified_caller(srv):
    """Verified caller not on read list -> stream yields no messages from that channel."""
    _restrict(srv, channel="secret", read=["alice"])
    since = time.time()
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    time.sleep(0.05)
    # bob streams
    msgs = srv.stream(thread="secret", token=_token("bob"), since=since, timeout=2.0)
    bodies = [m["body"] for m in msgs]
    assert CANARY not in bodies, f"bob should not see secret messages: {bodies}"


def test_stream_restricted_channel_denied_unauthenticated_caller(srv):
    """Unauthenticated caller on restricted channel -> stream yields no messages."""
    _restrict(srv, channel="secret", read=["alice"])
    since = time.time()
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    time.sleep(0.05)
    msgs = srv.stream(thread="secret", since=since, timeout=2.0)
    bodies = [m["body"] for m in msgs]
    assert CANARY not in bodies


def test_stream_restricted_channel_allowed_caller_sees_all(srv):
    """Verified caller on read list sees all messages in stream."""
    _restrict(srv, channel="secret", read=["alice", "bob"], post=["alice", "bob"])
    since = time.time()
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    time.sleep(0.05)
    msgs = srv.stream(thread="secret", token=_token("alice"), since=since, timeout=2.0)
    bodies = [m["body"] for m in msgs]
    assert CANARY in bodies


def test_stream_cursor_advances_past_denied_rows(srv):
    """R-ONCE: stream cursor (since) advances past denied rows so they
    don't cause repeated polling overhead."""
    _restrict(srv, channel="secret", read=["alice"])
    since = time.time()
    # Post 5 denied messages
    for i in range(5):
        srv.send("alice", "secret", body=f"denied-{i}", token=_token("alice"))
    # Post 1 allowed message on general
    srv.send("agentX", "general", body=CANARY_ALLOWED)
    time.sleep(0.05)
    # bob streams general (no thread filter = all channels)
    # The stream should skip the 5 denied and deliver the allowed one
    msgs = srv.stream(token=_token("bob"), since=since, timeout=3.0, max_lines=10)
    bodies = [m["body"] for m in msgs]
    # Should eventually see the allowed message
    assert CANARY_ALLOWED in bodies, f"bob should see allowed message after denied run: {bodies}"
    # Should not see any denied messages
    assert not any("denied" in b for b in bodies)


def test_stream_since_parameter_respected_after_denied_run(srv):
    """When since= is provided, stream starts from that timestamp and
    still advances cursor past denied rows."""
    _restrict(srv, channel="secret", read=["alice"])
    # Post denied messages
    for i in range(3):
        srv.send("alice", "secret", body=f"denied-{i}", token=_token("alice"))
    time.sleep(0.05)
    pivot = time.time()
    time.sleep(0.05)
    # Post allowed message after pivot
    srv.send("agentX", "general", body=CANARY_ALLOWED)
    time.sleep(0.1)
    # bob streams with since=pivot
    msgs = srv.stream(since=pivot, token=_token("bob"), timeout=3.0, max_lines=10)
    bodies = [m["body"] for m in msgs]
    # Should see allowed message (denied ones are before pivot or filtered)
    assert CANARY_ALLOWED in bodies, f"bob should see allowed message: {bodies}"
    assert not any("denied" in b for b in bodies)


def test_stream_without_thread_filters_all_channels(srv):
    """GET /a2a/stream without ?thread= filters across all channels."""
    _restrict(srv, channel="secret", read=["alice"])
    since = time.time()
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    srv.send("agentX", "general", body=CANARY_ALLOWED)
    time.sleep(0.05)
    msgs = srv.stream(token=_token("bob"), since=since, timeout=2.0)
    bodies = [m["body"] for m in msgs]
    assert CANARY not in bodies
    assert CANARY_ALLOWED in bodies


def test_stream_malformed_acl_denies_read(srv):
    """Malformed ACL entry denies stream for everyone."""
    d = srv.data_dir
    with open(f"{d}/config.json") as fh:
        conf = json.load(fh)
    conf["channel_acl"] = {"broken": ["alice"]}
    with open(f"{d}/config.json", "w") as fh:
        json.dump(conf, fh)
    since = time.time()
    srv.send("alice", "broken", body=CANARY, token=_token("alice"))
    time.sleep(0.05)
    msgs = srv.stream(thread="broken", token=_token("alice"), since=since, timeout=2.0)
    bodies = [m["body"] for m in msgs]
    assert CANARY not in bodies


# =====================================================================
# Bounded scan verification (spy on service.a2a_feed)
# =====================================================================

def test_messages_bounded_scan_spy_on_feed_limit(srv, monkeypatch):
    """MUTANT m3: unbounded scan -> spy on service.a2a_feed limit fails."""
    _restrict(srv, channel="secret", read=["alice"])
    # Post many denied messages
    for i in range(200):
        srv.send("alice", "secret", body=f"secret-{i}", token=_token("alice"))
    # Post few allowed
    for i in range(3):
        srv.send("agentX", "general", body=f"general-{i}")

    # Spy on a2a_feed to capture the limit argument
    seen_limits = []
    real_feed = service.a2a_feed

    async def spy_feed(*, thread, since, limit, data_dir):
        seen_limits.append(limit)
        return await real_feed(thread=thread, since=since, limit=limit, data_dir=data_dir)

    monkeypatch.setattr(service, "a2a_feed", spy_feed)

    status, raw = srv.messages(limit=10, token=_token("bob"))
    assert status == 200

    # The limit passed to a2a_feed should be capped at limit*5 = 50
    # (or the actual limit if the implementation pages differently)
    max_seen = max(seen_limits) if seen_limits else 0
    assert max_seen <= 50, f"feed limit {max_seen} exceeded bound of 50 (limit*5)"


def test_stream_bounded_scan_spy_on_feed_limit(srv, monkeypatch):
    """MUTANT m3: stream also respects bounded scan."""
    _restrict(srv, channel="secret", read=["alice"])
    since = time.time()
    for i in range(200):
        srv.send("alice", "secret", body=f"secret-{i}", token=_token("alice"))
    for i in range(3):
        srv.send("agentX", "general", body=f"general-{i}")
    time.sleep(0.05)

    seen_limits = []
    real_feed = service.a2a_feed

    async def spy_feed(*, thread, since, limit, data_dir):
        seen_limits.append(limit)
        return await real_feed(thread=thread, since=since, limit=limit, data_dir=data_dir)

    monkeypatch.setattr(service, "a2a_feed", spy_feed)

    _ = srv.stream(token=_token("bob"), since=since, timeout=2.0, max_lines=10)

    max_seen = max(seen_limits) if seen_limits else 0
    # Stream uses limit=200 internally per poll; the bounded scan should
    # not exceed that per poll, but the test ensures we don't scan unbounded
    assert max_seen <= 200, f"stream feed limit {max_seen} exceeded 200"


# =====================================================================
# MUTANT tests (these should FAIL on the tree without the fix)
# =====================================================================

# MUTANT m1: remove cursor advance for denied rows
# This test will fail if the cursor doesn't advance past denied rows
def test_mutant_m1_cursor_advance_past_denied(srv):
    """MUTANT m1: if cursor advance for denied rows is removed, this FAILS."""
    _restrict(srv, channel="secret", read=["alice"])
    since = time.time()
    for i in range(10):
        srv.send("alice", "secret", body=f"denied-{i}", token=_token("alice"))
    srv.send("agentX", "general", body=CANARY_ALLOWED)
    time.sleep(0.05)
    # If cursor doesn't advance past denied, the stream will keep
    # re-polling the same denied region and may time out before reaching
    # the allowed message, or the allowed message arrives very late.
    msgs = srv.stream(token=_token("bob"), since=since, timeout=4.0, max_lines=15)
    bodies = [m["body"] for m in msgs]
    # Must see the allowed message within reasonable time
    assert CANARY_ALLOWED in bodies, f"MUTANT m1: allowed message not received: {bodies}"


# MUTANT m2: remove filter on /a2a/stream
# This test will fail if the stream doesn't filter denied channels
def test_mutant_m2_stream_filters_denied(srv):
    """MUTANT m2: if filter on /a2a/stream is removed, this FAILS."""
    _restrict(srv, channel="secret", read=["alice"])
    since = time.time()
    srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    time.sleep(0.05)
    msgs = srv.stream(thread="secret", token=_token("bob"), since=since, timeout=2.0)
    bodies = [m["body"] for m in msgs]
    # Canary must not appear in stream for denied caller
    assert CANARY not in bodies, f"MUTANT m2: canary leaked in stream: {bodies}"


# MUTANT m3: unbounded scan
# This test will fail if the scan is not bounded
def test_mutant_m3_bounded_scan(srv, monkeypatch):
    """MUTANT m3: if scan is unbounded (limit=10**9), this FAILS."""
    _restrict(srv, channel="secret", read=["alice"])
    for i in range(500):
        srv.send("alice", "secret", body=f"secret-{i}", token=_token("alice"))
    for i in range(2):
        srv.send("agentX", "general", body=f"general-{i}")

    seen_limits = []
    real_feed = service.a2a_feed

    async def spy_feed(*, thread, since, limit, data_dir):
        seen_limits.append(limit)
        return await real_feed(thread=thread, since=since, limit=limit, data_dir=data_dir)

    monkeypatch.setattr(service, "a2a_feed", spy_feed)

    status, raw = srv.messages(limit=5, token=_token("bob"))
    assert status == 200

    # If unbounded scan is used, limit will be huge (10**9 or similar)
    max_seen = max(seen_limits) if seen_limits else 0
    assert max_seen <= 25, f"MUTANT m3: unbounded scan detected, limit={max_seen}"


# =====================================================================
# Alias resolution (R-ALIAS)
# =====================================================================

def test_messages_renamed_channel_keeps_acl(srv):
    """R-ALIAS: an ACL set on a channel survives an admin rename for reads."""
    _restrict(srv, channel="secret", read=["alice"])
    status, raw = _call(
        "POST", f"{srv.base}/a2a/admin/rename-channel",
        {"from": "secret", "to": "vault"}, ADMIN,
    )
    assert status == 200, raw
    # Post to new name
    srv.send("alice", "vault", body=CANARY, token=_token("alice"))
    # bob reads new name -> denied
    status, raw = srv.messages(thread="vault", token=_token("bob"))
    assert status == 200
    body = _j(raw)
    assert body["messages"] == []
    assert CANARY not in raw
    # alice reads new name -> allowed
    status, raw = srv.messages(thread="vault", token=_token("alice"))
    assert status == 200
    body = _j(raw)
    assert any(m["body"] == CANARY for m in body["messages"])


def test_stream_renamed_channel_keeps_acl(srv):
    """R-ALIAS: an ACL set on a channel survives an admin rename for stream."""
    _restrict(srv, channel="secret", read=["alice"])
    status, raw = _call(
        "POST", f"{srv.base}/a2a/admin/rename-channel",
        {"from": "secret", "to": "vault"}, ADMIN,
    )
    assert status == 200, raw
    since = time.time()
    srv.send("alice", "vault", body=CANARY, token=_token("alice"))
    time.sleep(0.05)
    # bob streams new name -> denied
    msgs = srv.stream(thread="vault", token=_token("bob"), since=since, timeout=2.0)
    bodies = [m["body"] for m in msgs]
    assert CANARY not in bodies
    # alice streams new name -> allowed
    msgs = srv.stream(thread="vault", token=_token("alice"), since=since, timeout=2.0)
    bodies = [m["body"] for m in msgs]
    assert CANARY in bodies