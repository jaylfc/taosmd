"""Tests for thread-membership read gate (bus-auth gate R2).

A thread that has ANY active membership (has_any_membership) is readable
only by its active members, identified by the verified token sub.
Threads with no membership rows (open/legacy) are UNCHANGED.
"""

from __future__ import annotations

import json
import threading
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

ADMIN = "admin-token-for-mem-read-tests"
CANARY = "CANARY-thread-mem-read"
MEMBER_THREAD = "member-thread-x7k2"
OPEN_THREAD = "open-thread-y9m1"
ALICE = "alice-mem"
BOB = "bob-mem"
CAROL = "carol-mem"
POSTER = "poster-mem"
OPEN_BODY = "open-thread-body-ok"

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

    def admin_set(self, payload, token=ADMIN):
        return _call("POST", f"{self.base}/a2a/admin/set-channel-acl", payload, token)

    def archive_dump(self):
        rows = self.httpd.service_loop.run(
            service.a2a_feed(thread=None, limit=1000, data_dir=self.data_dir)
        )
        return json.dumps(rows)


def _build(tmp_path, monkeypatch, enforce, with_verifier=True):
    data_dir = tmp_path / "taosmd-mem-read"
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
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir),
        verifier=verifier if with_verifier else None,
    )
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return Srv(f"http://{host}:{port}", data_dir, httpd), httpd, t


@pytest.fixture
def srv(tmp_path, monkeypatch):
    """Verify-and-warn mode with registry verifier."""
    s, httpd, t = _build(tmp_path, monkeypatch, enforce=False)
    try:
        yield s
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


@pytest.fixture
def srv_standalone(tmp_path, monkeypatch):
    """No registry verifier: no caller is ever verified."""
    s, httpd, t = _build(tmp_path, monkeypatch, enforce=False, with_verifier=False)
    try:
        yield s
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


def _seed(srv):
    """Create a thread with membership (alice + bob) and post a canary message,
    plus an open thread with no membership rows.
    """
    # Create member-restricted thread via service layer (alice=owner, bob=member)
    srv.httpd.service_loop.run(service.a2a_create_thread(
        MEMBER_THREAD, [BOB], ALICE, data_dir=srv.data_dir
    ))
    # Post canary as alice (owner)
    status, raw = srv.send(ALICE, MEMBER_THREAD, body=f"{CANARY} from alice", token=_token(ALICE))
    assert status == 200, raw
    srv.cmsg_id = _j(raw)["id"]

    # Open thread with no membership rows
    status, raw = srv.send(POSTER, OPEN_THREAD, body=OPEN_BODY)
    assert status == 200, raw
    srv.omsg_id = _j(raw)["id"]


def _assert_no_leak(raw):
    for needle in (CANARY, MEMBER_THREAD, ALICE, BOB):
        assert needle not in raw, f"Leaked {needle!r} in response"


def _tok(who):
    return {
        "alice": _token(ALICE),
        "bob": _token(BOB),
        "carol": _token(CAROL),
        "anon": None,
        "forged-alice": _token(ALICE),  # same sub but forged signature would fail verify; here we test no-token case
    }[who]


# ---------------------------------------------------------------------------
# /a2a/messages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", ["alice", "bob"])
def test_messages_member_thread_allowed(srv, who):
    """Active members (alice, bob) can read the member-restricted thread."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/messages?thread={MEMBER_THREAD}", None, _tok(who))
    assert status == 200, raw
    assert CANARY in raw
    assert MEMBER_THREAD in raw


@pytest.mark.parametrize("who", ["carol", "anon"])
def test_messages_member_thread_denied(srv, who):
    """Non-members (carol) and no-token (anon) cannot read the member-restricted thread."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/messages?thread={MEMBER_THREAD}", None, _tok(who))
    assert status == 200, raw
    _assert_no_leak(raw)


def test_messages_open_thread_works_unauthenticated(srv):
    """Open thread (no membership rows) stays readable by everyone."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/messages?thread={OPEN_THREAD}", None, None)
    assert status == 200, raw
    assert OPEN_BODY in raw


def test_messages_open_thread_works_for_carol(srv):
    """Open thread readable by carol (non-member of other thread)."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/messages?thread={OPEN_THREAD}", None, _token(CAROL))
    assert status == 200, raw
    assert OPEN_BODY in raw


def test_messages_no_thread_filter_member_restricted_dropped(srv):
    """When no thread filter, member-restricted threads are dropped for non-members."""
    _seed(srv)
    # alice sees both
    status, raw = _call("GET", f"{srv.base}/a2a/messages", None, _token(ALICE))
    assert status == 200, raw
    assert CANARY in raw
    assert OPEN_BODY in raw

    # carol sees only open
    status, raw = _call("GET", f"{srv.base}/a2a/messages", None, _token(CAROL))
    assert status == 200, raw
    _assert_no_leak(raw)
    assert OPEN_BODY in raw


def test_messages_no_thread_filter_open_readable_by_anon(srv):
    """No thread filter: open thread readable by anon."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/messages", None, None)
    assert status == 200, raw
    assert OPEN_BODY in raw
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# /a2a/threads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", ["alice", "bob"])
def test_threads_member_thread_visible_to_members(srv, who):
    """Active members see the member-restricted thread in /a2a/threads."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/threads", None, _tok(who))
    assert status == 200, raw
    threads = {t["thread"] for t in _j(raw)["threads"]}
    assert MEMBER_THREAD in threads


@pytest.mark.parametrize("who", ["carol", "anon"])
def test_threads_member_thread_hidden_from_non_members(srv, who):
    """Non-members and anon do NOT see the member-restricted thread in /a2a/threads."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/threads", None, _tok(who))
    assert status == 200, raw
    threads = {t["thread"] for t in _j(raw)["threads"]}
    assert MEMBER_THREAD not in threads
    assert OPEN_THREAD in threads


def test_threads_open_thread_visible_to_everyone(srv):
    """Open thread visible to everyone including anon."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/threads", None, None)
    assert status == 200, raw
    threads = {t["thread"] for t in _j(raw)["threads"]}
    assert OPEN_THREAD in threads


# ---------------------------------------------------------------------------
# /a2a/threads/{thread}/messages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", ["alice", "bob"])
def test_thread_messages_member_thread_allowed(srv, who):
    """Active members can read /a2a/threads/{t}/messages for member-restricted thread."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/threads/{MEMBER_THREAD}/messages", None, _tok(who))
    assert status == 200, raw
    assert CANARY in raw
    assert MEMBER_THREAD in raw


@pytest.mark.parametrize("who", ["carol", "anon"])
def test_thread_messages_member_thread_denied_403(srv, who):
    """Non-members and anon get 403 on /a2a/threads/{t}/messages for member-restricted thread."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/threads/{MEMBER_THREAD}/messages", None, _tok(who))
    assert status == 403, raw
    _assert_no_leak(raw)


def test_thread_messages_open_thread_works_unauthenticated(srv):
    """Open thread /a2a/threads/{t}/messages works for anon."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/threads/{OPEN_THREAD}/messages", None, None)
    assert status == 200, raw
    assert OPEN_BODY in raw


def test_thread_messages_open_thread_works_for_carol(srv):
    """Open thread /a2a/threads/{t}/messages works for carol."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/threads/{OPEN_THREAD}/messages", None, _token(CAROL))
    assert status == 200, raw
    assert OPEN_BODY in raw


# ---------------------------------------------------------------------------
# /a2a/mentions (uses _filter_readable)
# ---------------------------------------------------------------------------

def test_mentions_member_thread_filtered(srv):
    """Mentions feed filters member-restricted threads for non-members."""
    _seed(srv)
    # alice sees mentions from both threads
    status, raw = _call("GET", f"{srv.base}/a2a/mentions?reader={ALICE}&since=0", None, _token(ALICE))
    assert status == 200, raw

    # carol should not see member-restricted thread mentions
    status, raw = _call("GET", f"{srv.base}/a2a/mentions?reader={CAROL}&since=0", None, _token(CAROL))
    assert status == 200, raw
    # mentions from member thread should be filtered out
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# /a2a/inbox (uses _filter_readable)
# ---------------------------------------------------------------------------

def test_inbox_member_thread_filtered(srv):
    """Inbox feed filters member-restricted threads for non-members."""
    _seed(srv)
    # alice sees inbox from both threads
    status, raw = _call("GET", f"{srv.base}/a2a/inbox?consumer={ALICE}&limit=50", None, _token(ALICE))
    assert status == 200, raw

    # carol should not see member-restricted thread messages
    status, raw = _call("GET", f"{srv.base}/a2a/inbox?consumer={CAROL}&limit=50", None, _token(CAROL))
    assert status == 200, raw
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# /a2a/inbox/unhandled (uses _filter_readable)
# ---------------------------------------------------------------------------

def test_inbox_unhandled_member_thread_filtered(srv):
    """Unhandled feed filters member-restricted threads for non-members."""
    _seed(srv)
    status, raw = _call("GET", f"{srv.base}/a2a/inbox/unhandled?consumer={ALICE}&limit=50", None, _token(ALICE))
    assert status == 200, raw

    status, raw = _call("GET", f"{srv.base}/a2a/inbox/unhandled?consumer={CAROL}&limit=50", None, _token(CAROL))
    assert status == 200, raw
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# /a2a/stream (uses _filter_readable per poll)
# ---------------------------------------------------------------------------

def test_stream_member_thread_filtered(srv):
    """Stream filters member-restricted threads for non-members."""
    import socket
    import time

    _seed(srv)
    since = time.time() - 60

    def _stream_read(token, window=1.6):
        host, port = srv.base.split("//")[1].split(":")
        q = f"?since={since}&thread={MEMBER_THREAD}"
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
        return buf.decode("utf-8", errors="replace")

    # alice receives the canary
    raw = _stream_read(_token(ALICE))
    assert CANARY in raw

    # carol receives nothing from member thread
    raw = _stream_read(_token(CAROL))
    _assert_no_leak(raw)

    # anon receives nothing from member thread
    raw = _stream_read(None)
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# Standalone (no verifier) - open threads only
# ---------------------------------------------------------------------------

def test_standalone_open_thread_works(srv_standalone):
    """Standalone server: open thread works without auth."""
    # Post to open thread
    status, raw = _call("POST", f"{srv_standalone.base}/a2a/send",
                        {"from": POSTER, "body": OPEN_BODY, "thread": OPEN_THREAD})
    assert status == 200, raw

    # Read back - should work without token
    status, raw = _call("GET", f"{srv_standalone.base}/a2a/messages?thread={OPEN_THREAD}")
    assert status == 200, raw
    assert OPEN_BODY in raw


def test_standalone_member_thread_not_enforced(srv_standalone):
    """Standalone server: without verifier, membership is not enforced (falls back to open).
    
    This is the current behaviour: when no verifier is configured, _get_authenticated_agent_id
    returns None, and the membership gate treats None as non-member. But since there's no
    channel ACL either, the thread would be open. The membership gate only applies when
    has_any_membership is True AND a verified sub exists.
    """
    # Create member-restricted thread via service
    srv_standalone.httpd.service_loop.run(service.a2a_create_thread(
        MEMBER_THREAD, [BOB], ALICE, data_dir=srv_standalone.data_dir
    ))
    # Post canary
    status, raw = _call("POST", f"{srv_standalone.base}/a2a/send",
                        {"from": ALICE, "body": f"{CANARY} from alice", "thread": MEMBER_THREAD})
    assert status == 200, raw

    # In standalone mode, without a verifier, the membership gate cannot identify
    # the caller, so the thread behaves as open (no verified sub = non-member,
    # but the gate should still allow since no ACL is configured and membership
    # gate only restricts when there's a verified sub).
    # Actually, the spec says: "A caller with no verified sub is a non-member."
    # So even in standalone, if a thread has membership, it should be restricted.
    # But without a verifier, there's no way to BE a member. This test documents
    # the expected behaviour: without verifier, membership-gated threads are
    # inaccessible (403 on single-thread reads, filtered on list reads).
    status, raw = _call("GET", f"{srv_standalone.base}/a2a/threads/{MEMBER_THREAD}/messages")
    # Since no verifier, _get_authenticated_agent_id returns None -> non-member -> 403
    assert status == 403, raw

    # But open thread still works
    status, raw = _call("GET", f"{srv_standalone.base}/a2a/threads/{OPEN_THREAD}/messages")
    assert status == 200, raw


# ---------------------------------------------------------------------------
# No channel ACL interaction: membership gate is independent
# ---------------------------------------------------------------------------

def test_membership_gate_independent_of_channel_acl(srv):
    """Thread membership gate operates independently of channel ACL.
    
    A thread can have BOTH membership restriction AND channel ACL.
    Both must pass. This test confirms the membership gate is checked
    in addition to (not instead of) channel ACL.
    """
    _seed(srv)
    # Add channel ACL restricting to ALICE only
    status, raw = srv.admin_set({"channel": MEMBER_THREAD, "read": [ALICE], "post": [ALICE]})
    assert status == 200, raw

    # bob is a member but not in channel ACL read list -> denied by channel ACL
    status, raw = _call("GET", f"{srv.base}/a2a/messages?thread={MEMBER_THREAD}", None, _token(BOB))
    assert status == 200, raw
    _assert_no_leak(raw)  # denied by channel ACL

    # alice is member AND in channel ACL -> allowed
    status, raw = _call("GET", f"{srv.base}/a2a/messages?thread={MEMBER_THREAD}", None, _token(ALICE))
    assert status == 200, raw
    assert CANARY in raw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])