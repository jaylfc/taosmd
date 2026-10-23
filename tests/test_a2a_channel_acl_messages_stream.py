"""Slice 2 of the channel-ACL work: read filtering on /a2a/messages and /a2a/stream.

Identity for every ACL decision is the verified registry token ``sub`` and
nothing else. A restricted channel with no verified caller is DENIED; a
channel with no ACL stays open to everyone. Every deny test uses a canary in
a message body plus unique channel and sender names, and asserts none of
them appear anywhere in the response body. All state lives under ``tmp_path``.
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
CHAN = "vault-chan-7d2e"
SENDER = "sender-zq81"
ALICE = SENDER
OPEN_BODY = "open-channel-body-ok"


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
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _j(raw):
    return json.loads(raw or "{}")


class Srv:
    def __init__(self, base, data_dir, httpd):
        self.base = base
        self.data_dir = str(data_dir)
        self.httpd = httpd

    def send(self, sender, thread, body="hello", token=None, **extra):
        payload = {"from": sender, "body": body, "thread": thread, **extra}
        return _call("POST", f"{self.base}/a2a/send", payload, token)

    def imp(self, sender, thread, body="hello", token=None):
        payload = {"envelopes": [{"from": sender, "body": body, "thread": thread}]}
        return _call("POST", f"{self.base}/a2a/import", payload, token)

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


def _build(tmp_path, monkeypatch, enforce):
    data_dir = tmp_path / "taosmd-acl"
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


def _restrict(srv, channel=CHAN, read=(ALICE,), post=(ALICE,)):
    status, raw = srv.admin_set({"channel": channel, "read": list(read), "post": list(post)})
    assert status == 200, raw


def _forged_token(sub):
    other_priv, _ = _keypair()
    return pyjwt.encode({"sub": sub}, other_priv, algorithm="EdDSA")


def _messages(srv, thread=None, fmt="json", token=None, limit=None):
    params = []
    if thread:
        params.append(f"thread={thread}")
    if fmt != "json":
        params.append(f"format={fmt}")
    if limit is not None:
        params.append(f"limit={limit}")
    q = "?" + "&".join(params) if params else ""
    return _call("GET", f"{srv.base}/a2a/messages{q}", None, token)


def _bodies(raw, fmt):
    if fmt == "ndjson":
        return [json.loads(ln)["body"] for ln in raw.splitlines() if ln.strip()]
    return [m["body"] for m in json.loads(raw)["messages"]]


def _stream(srv, since, thread=None, token=None, window=2.6):
    """Read /a2a/stream over a raw socket for ``window`` seconds.

    Returns (raw_response_text, [parsed data frames]). Never blocks past the
    window: the socket timeout is re-derived from the deadline on each recv.
    """
    host, port = srv.base.split("//")[1].split(":")
    q = f"?since={since}" + (f"&thread={thread}" if thread else "")
    head = f"GET /a2a/stream{q} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n"
    if token:
        head += f"Authorization: Bearer {token}\r\n"
    deadline = time.monotonic() + window
    buf = b""
    with socket.create_connection((host, int(port)), timeout=window) as sock:
        sock.sendall((head + "\r\n").encode())
        while time.monotonic() < deadline:
            sock.settimeout(max(0.05, deadline - time.monotonic()))
            try:
                chunk = sock.recv(4096)
            except (socket.timeout, TimeoutError):
                break
            if not chunk:
                break
            buf += chunk
    raw = buf.decode("utf-8", errors="replace")
    frames = [json.loads(ln[6:]) for ln in raw.splitlines() if ln.startswith("data: ")]
    return raw, frames


def _seed(srv):
    """A restricted channel (alice only) with a canary, plus an open channel."""
    _restrict(srv)
    assert srv.send(SENDER, CHAN, body=CANARY, token=_token(ALICE))[0] == 200
    assert srv.send("agentA", "general", body=OPEN_BODY)[0] == 200


def _assert_no_leak(raw):
    assert CANARY not in raw
    assert CHAN not in raw
    assert SENDER not in raw


# ---------------------------------------------------------------------------
# /a2a/messages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["json", "ndjson"])
def test_messages_open_channel_works_unauthenticated(srv, fmt):
    srv.send("agentA", "general", body=OPEN_BODY)
    for thread in ("general", None):
        status, raw = _messages(srv, thread=thread, fmt=fmt)
        assert status == 200, raw
        assert OPEN_BODY in _bodies(raw, fmt)


@pytest.mark.parametrize("fmt", ["json", "ndjson"])
@pytest.mark.parametrize("thread", [CHAN, None])
@pytest.mark.parametrize("who", ["bob", "anon", "forged-alice"])
def test_messages_restricted_denied(srv, who, thread, fmt):
    _seed(srv)
    token = {"bob": _token("bob"), "anon": None, "forged-alice": _forged_token(ALICE)}[who]
    status, raw = _messages(srv, thread=thread, fmt=fmt, token=token)
    assert status == 200, raw
    _assert_no_leak(raw)
    if thread is None:
        assert OPEN_BODY in _bodies(raw, fmt)  # open rows still listed
    else:
        assert _bodies(raw, fmt) == []


@pytest.mark.parametrize("fmt", ["json", "ndjson"])
@pytest.mark.parametrize("thread", [CHAN, None])
def test_messages_restricted_allowed_sees_everything(srv, thread, fmt):
    _seed(srv)
    status, raw = _messages(srv, thread=thread, fmt=fmt, token=_token(ALICE))
    assert status == 200, raw
    assert CANARY in _bodies(raw, fmt)
    assert CHAN in raw and SENDER in raw
    if thread is None:
        assert OPEN_BODY in _bodies(raw, fmt)


def test_messages_malformed_acl_entry_denies(srv):
    _seed(srv)
    with open(f"{srv.data_dir}/config.json") as fh:
        conf = json.load(fh)
    conf["channel_acl"][CHAN] = [ALICE]  # entry is a list, not a dict
    with open(f"{srv.data_dir}/config.json", "w") as fh:
        json.dump(conf, fh)
    status, raw = _messages(srv, thread=CHAN, token=_token(ALICE))
    assert status == 200, raw
    _assert_no_leak(raw)


def test_messages_renamed_channel_keeps_acl(srv):
    _seed(srv)
    status, raw = _call(
        "POST", f"{srv.base}/a2a/admin/rename-channel",
        {"from": CHAN, "to": "renamed-chan-55"}, ADMIN,
    )
    assert status == 200, raw
    for token in (_token("bob"), None):
        for thread in ("renamed-chan-55", None):
            status, raw = _messages(srv, thread=thread, token=token)
            assert status == 200, raw
            assert CANARY not in raw and SENDER not in raw
    status, raw = _messages(srv, thread="renamed-chan-55", token=_token(ALICE))
    assert CANARY in raw


class _FeedSpy:
    def __init__(self, monkeypatch):
        self.calls = []
        real = service.a2a_feed

        async def spy(**kwargs):
            self.calls.append(dict(kwargs))
            return await real(**kwargs)

        monkeypatch.setattr(service, "a2a_feed", spy)

    @property
    def limits(self):
        return [c["limit"] for c in self.calls]


def test_messages_pages_past_denied_rows_to_fill_limit(srv, monkeypatch):
    _restrict(srv)
    srv.send("agentA", "general", body="old-1")
    srv.send("agentA", "general", body="old-2")
    for i in range(3):
        assert srv.send(ALICE, CHAN, body=f"{CANARY}-{i}", token=_token(ALICE))[0] == 200
    spy = _FeedSpy(monkeypatch)
    status, raw = _messages(srv, limit=2, token=_token("bob"))
    assert status == 200, raw
    assert _bodies(raw, "json") == ["old-1", "old-2"]
    assert CANARY not in raw
    assert max(spy.limits) <= 10


def test_messages_scan_is_bounded_at_limit_times_five(srv, monkeypatch):
    _restrict(srv)
    srv.send("agentA", "general", body="too-old-to-reach")
    for i in range(14):
        assert srv.send(ALICE, CHAN, body=f"{CANARY}-{i}", token=_token(ALICE))[0] == 200
    spy = _FeedSpy(monkeypatch)
    status, raw = _messages(srv, limit=2, token=_token("bob"))
    assert status == 200, raw
    assert max(spy.limits) <= 2 * 5, spy.limits
    assert max(spy.limits) == 2 * 5, spy.limits  # it did scan the full cap
    assert _bodies(raw, "json") == []
    assert CANARY not in raw


# ---------------------------------------------------------------------------
# /a2a/stream
# ---------------------------------------------------------------------------

def test_stream_open_channel_works_unauthenticated(srv):
    since = time.time() - 60
    srv.send("agentA", "general", body=OPEN_BODY)
    raw, frames = _stream(srv, since, thread="general", window=1.6)
    assert OPEN_BODY in [f["body"] for f in frames]
    raw, frames = _stream(srv, since, window=1.6)
    assert OPEN_BODY in [f["body"] for f in frames]


@pytest.mark.parametrize("thread", [CHAN, None])
@pytest.mark.parametrize("who", ["bob", "anon", "forged-alice"])
def test_stream_restricted_denied(srv, who, thread):
    since = time.time() - 60
    _seed(srv)
    token = {"bob": _token("bob"), "anon": None, "forged-alice": _forged_token(ALICE)}[who]
    raw, frames = _stream(srv, since, thread=thread, token=token)
    assert raw.startswith("HTTP/1.") and " 200 " in raw.split("\r\n")[0]
    _assert_no_leak(raw)
    if thread is None:
        assert OPEN_BODY in [f["body"] for f in frames]


@pytest.mark.parametrize("thread", [CHAN, None])
def test_stream_restricted_allowed_sees_everything(srv, thread):
    since = time.time() - 60
    _seed(srv)
    raw, frames = _stream(srv, since, thread=thread, token=_token(ALICE), window=1.6)
    assert CANARY in [f["body"] for f in frames]
    assert CHAN in raw and SENDER in raw


def test_stream_malformed_acl_entry_denies(srv):
    since = time.time() - 60
    _seed(srv)
    with open(f"{srv.data_dir}/config.json") as fh:
        conf = json.load(fh)
    conf["channel_acl"][CHAN] = ALICE
    with open(f"{srv.data_dir}/config.json", "w") as fh:
        json.dump(conf, fh)
    raw, _ = _stream(srv, since, thread=CHAN, token=_token(ALICE), window=1.6)
    _assert_no_leak(raw)


def test_stream_poll_cursor_moves_past_denied_rows(srv, monkeypatch):
    since = time.time() - 60
    _restrict(srv)
    for i in range(4):
        assert srv.send(ALICE, CHAN, body=f"{CANARY}-{i}", token=_token(ALICE))[0] == 200
    dump = json.loads(srv.archive_dump())
    newest_denied = max(m["ts"] for m in dump if m["thread"] == CHAN)
    spy = _FeedSpy(monkeypatch)
    raw, frames = _stream(srv, since, token=_token("bob"), window=3.4)
    _assert_no_leak(raw)
    sinces = [c["since"] for c in spy.calls]
    assert len(sinces) >= 2, sinces
    assert sinces[0] == pytest.approx(since)
    assert all(s >= newest_denied for s in sinces[1:]), (sinces, newest_denied)


def test_stream_delivers_allowed_message_posted_after_denied_run(srv):
    since = time.time() - 60
    _restrict(srv)
    for i in range(5):
        assert srv.send(ALICE, CHAN, body=f"{CANARY}-{i}", token=_token(ALICE))[0] == 200

    def later():
        time.sleep(1.3)
        srv.send("agentA", "general", body=OPEN_BODY)

    t = threading.Thread(target=later)
    t.start()
    raw, frames = _stream(srv, since, token=_token("bob"), window=3.6)
    t.join()
    assert [f["body"] for f in frames] == [OPEN_BODY]
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# New tests for channel-ACL revision tsk-3pqsbf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_newer", [2, 5, 12])
def test_messages_short_page_pages_past_deleted_channel(srv, n_newer):
    """Rows of a deleted channel are dropped by the feed. A page that is empty
    or short after the drop must keep paging back to the older readable rows."""
    for i in range(5):
        assert srv.send("agentA", "general", body=f"old-{i}")[0] == 200
    for i in range(n_newer):
        assert srv.send("agentA", CHAN, body=f"gone-{i}")[0] == 200
    status, raw = _call("POST", f"{srv.base}/a2a/admin/delete-channel", {"channel": CHAN}, ADMIN)
    assert status == 200, raw
    status, raw = _messages(srv, limit=5, token=_token("bob"))
    assert status == 200, raw
    assert _bodies(raw, "json") == [f"old-{i}" for i in range(5)]
    assert "gone-" not in raw


def test_messages_stops_paging_when_archive_exhausted(srv, monkeypatch):
    for i in range(2):
        assert srv.send("agentA", "general", body=f"only-{i}")[0] == 200
    spy = _FeedSpy(monkeypatch)
    status, raw = _messages(srv, limit=5, token=_token("bob"))
    assert status == 200, raw
    assert _bodies(raw, "json") == ["only-0", "only-1"]
    assert spy.limits == [5], spy.limits


def test_caller_verified_once_per_request(srv, monkeypatch):
    from taosmd import registry_auth
    chan2 = "vault-chan-7d2e-2"
    _restrict(srv)
    _restrict(srv, channel=chan2)
    for i in range(6):
        assert srv.send(ALICE, CHAN, body=f"canary-{i}", token=_token(ALICE))[0] == 200
        assert srv.send(ALICE, chan2, body=f"canary2-{i}", token=_token(ALICE))[0] == 200
    calls = []
    real = registry_auth.RegistryVerifier.authorize

    def counting(self, token, claimed_from):
        calls.append(claimed_from)
        return real(self, token, claimed_from)

    monkeypatch.setattr(registry_auth.RegistryVerifier, "authorize", counting)
    status, raw = _messages(srv, limit=10, token=_token("bob"))
    assert status == 200, raw
    assert _bodies(raw, "json") == []
    assert len(calls) == 1, calls
    calls.clear()
    status, raw = _messages(srv, limit=10)
    assert status == 200, raw
    assert _bodies(raw, "json") == []
    assert calls == []


def test_messages_acl_allowed_sees_all_denied_sees_none(srv):
    """Test C: an allowed caller sees every row, a denied caller sees none.

    Confirms identity semantics: verified token ``sub`` only; no token on a
    restricted channel stays a DENY.
    """
    _restrict(srv)
    # Channel restricted to ALICE only
    srv.admin_set({"channel": CHAN, "read": (ALICE,), "post": (ALICE,)})

    # Send a canary message as ALICE to the restricted channel
    assert srv.send(ALICE, CHAN, body=CANARY, token=_token(ALICE))[0] == 200

    # Send an open-message as agentA to general (no ACL)
    assert srv.send("agentA", "general", body=OPEN_BODY)[0] == 200

    # ALICE (allowed caller) should see everything including the canary
    status, raw = _messages(srv, thread=CHAN, token=_token(ALICE))
    assert status == 200, raw
    assert CANARY in _bodies(raw, "json")

    # BOB (denied caller) should see nothing from the restricted channel
    status, raw = _messages(srv, thread=CHAN, token=_token("bob"))
    assert status == 200, raw
    bodies = _bodies(raw, "json")
    assert len(bodies) == 0, f"Bob should see 0 messages from restricted channel, got {bodies}"
    assert all(CANARY not in b for b in bodies), f"Bob should not see canary: {bodies}"
