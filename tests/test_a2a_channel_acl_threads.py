"""Slice 4 of the channel-ACL work: threads, thread messages, thread members, receipts.

Covers GET /a2a/threads, /a2a/threads/{t}/messages, /a2a/threads/{t}/members,
GET /a2a/messages/{id}/receipts and GET /a2a/receipts. Identity is the verified registry token ``sub``
only. Every deny test uses a restricted channel, a denied verified caller,
an unauthenticated caller, a canary body and unique channel and sender
names, and asserts none of them appear anywhere in the response body.
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

ADMIN = "admin-token-for-acl-tests"
CANARY = "CANARY-9f3b1c-do-not-leak"
CHAN = "vault-chan-7d2e"
SENDER = "sender-zq81"
MEMBER = "member-canary-p3t7"
OPEN_THREAD = "open-thread-m2k8"
OPEN_MEMBER = "open-member-c5d1"
ALICE = "alice-acl3"
POSTER = "poster-k4x9"
DUAL = "dual-w9m2"
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
        with urllib.request.urlopen(req, timeout=10) as resp:
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


def _build(tmp_path, monkeypatch, enforce, with_verifier=True):
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




def _restrict(srv, channel=CHAN):
    members = [ALICE, POSTER, DUAL]
    status, raw = srv.admin_set({"channel": channel, "read": members, "post": members})
    assert status == 200, raw


def _forged_token(sub):
    other_priv, _ = _keypair()
    return pyjwt.encode({"sub": sub}, other_priv, algorithm="EdDSA")


def _seed(srv):
    """Restricted channel (alice, poster, dual) with a canary message, receipts
    and members, plus an open channel and an open thread with members.

    Sets ``srv.cmsg`` (restricted message id) and ``srv.omsg`` (open message id).
    """
    _restrict(srv)
    status, raw = srv.send(POSTER, CHAN, body=f"{CANARY} {SENDER}", token=_token(POSTER))
    assert status == 200, raw
    srv.cmsg = _j(raw)["id"]
    status, raw = srv.send(DUAL, "general", body=OPEN_BODY)
    assert status == 200, raw
    srv.omsg = _j(raw)["id"]
    for thread, member in ((CHAN, MEMBER), (OPEN_THREAD, OPEN_MEMBER)):
        status, raw = _call(
            "POST", f"{srv.base}/a2a/threads",
            {"thread": thread, "participants": [member], "agent": ALICE},
        )
        assert status == 200, raw
    # Receipts: alice recorded delivered + seen for both messages (needs a token).
    for mid in (srv.cmsg, srv.omsg):
        assert _call("POST", f"{srv.base}/a2a/receipts", {"message_id": mid}, _token(ALICE))[0] == 200
        assert _call("PATCH", f"{srv.base}/a2a/receipts", {"message_id": mid}, _token(ALICE))[0] == 200


WHO = ["bob", "anon", "forged-alice"]


def _tok(who):
    return {
        "bob": _token("bob"), "anon": None,
        "forged-alice": _forged_token(ALICE), "alice": _token(ALICE),
    }[who]


def _assert_no_leak(raw):
    for needle in (CANARY, CHAN, POSTER, SENDER, MEMBER):
        assert needle not in raw, needle


def _get(srv, path, token=None):
    return _call("GET", f"{srv.base}{path}", None, token)


def _corrupt(srv):
    with open(f"{srv.data_dir}/config.json") as fh:
        conf = json.load(fh)
    conf["channel_acl"][CHAN] = [ALICE]  # entry is a list, not a dict
    with open(f"{srv.data_dir}/config.json", "w") as fh:
        json.dump(conf, fh)


# ---------------------------------------------------------------------------
# GET /a2a/threads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", WHO)
def test_threads_restricted_omitted(srv, who):
    _seed(srv)
    status, raw = _get(srv, "/a2a/threads", _tok(who))
    assert status == 200, raw
    _assert_no_leak(raw)
    names = [t["thread"] for t in _j(raw)["threads"]]
    assert "general" in names


def test_threads_principal_filter_cannot_widen(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/threads?principal={POSTER}", _tok("bob"))
    assert status == 200, raw
    _assert_no_leak(raw)


def test_threads_restricted_allowed_sees_everything(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/threads", _tok("alice"))
    assert status == 200, raw
    threads = {t["thread"]: t for t in _j(raw)["threads"]}
    assert {"general", CHAN} <= set(threads)
    assert threads[CHAN]["participants"] == [POSTER]
    assert CANARY in threads[CHAN]["last_message"]["body_preview"]


def test_threads_open_works_unauthenticated(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/threads")
    assert status == 200, raw
    assert "general" in [t["thread"] for t in _j(raw)["threads"]]


def test_threads_malformed_acl_entry_denies(srv):
    _seed(srv)
    _corrupt(srv)
    status, raw = _get(srv, "/a2a/threads", _tok("alice"))
    assert status == 200, raw
    _assert_no_leak(raw)


def test_threads_renamed_channel_stays_hidden(srv):
    _seed(srv)
    status, raw = _call(
        "POST", f"{srv.base}/a2a/admin/rename-channel",
        {"from": CHAN, "to": "renamed-thread-chan-77"}, ADMIN,
    )
    assert status == 200, raw
    for who in WHO:
        status, raw = _get(srv, "/a2a/threads", _tok(who))
        assert status == 200, raw
        assert CANARY not in raw and "renamed-thread-chan-77" not in raw
    status, raw = _get(srv, "/a2a/threads", _tok("alice"))
    assert "renamed-thread-chan-77" in raw


# ---------------------------------------------------------------------------
# GET /a2a/threads/{t}/messages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", WHO)
def test_thread_messages_restricted_denied(srv, who):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/threads/{CHAN}/messages", _tok(who))
    assert status == 403, raw
    _assert_no_leak(raw)


@pytest.mark.parametrize("who", WHO)
def test_thread_messages_bare_path_denied(srv, who):
    """``/a2a/threads/{t}`` (no ``/messages`` suffix) reads the same data."""
    _seed(srv)
    status, raw = _get(srv, f"/a2a/threads/{CHAN}", _tok(who))
    assert status == 403, raw
    _assert_no_leak(raw)


def test_thread_messages_restricted_allowed_sees_everything(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/threads/{CHAN}/messages", _tok("alice"))
    assert status == 200, raw
    msgs = _j(raw)["messages"]
    assert len(msgs) == 1 and CANARY in msgs[0]["body"]


def test_thread_messages_open_works_unauthenticated(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/threads/general/messages")
    assert status == 200, raw
    assert OPEN_BODY in raw


def test_thread_messages_malformed_acl_entry_denies(srv):
    _seed(srv)
    _corrupt(srv)
    status, raw = _get(srv, f"/a2a/threads/{CHAN}/messages", _tok("alice"))
    assert status == 403, raw
    _assert_no_leak(raw)


def test_thread_messages_renamed_old_name_denied(srv):
    _seed(srv)
    status, raw = _call(
        "POST", f"{srv.base}/a2a/admin/rename-channel",
        {"from": CHAN, "to": "renamed-thread-chan-77"}, ADMIN,
    )
    assert status == 200, raw
    for name in (CHAN, "renamed-thread-chan-77"):
        for who in WHO:
            status, raw = _get(srv, f"/a2a/threads/{name}/messages", _tok(who))
            assert status == 403, raw
            assert CANARY not in raw


# ---------------------------------------------------------------------------
# GET /a2a/threads/{t}/members
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", WHO)
def test_thread_members_restricted_denied(srv, who):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/threads/{CHAN}/members", _tok(who))
    assert status == 403, raw
    _assert_no_leak(raw)
    assert ALICE not in raw


def test_thread_members_restricted_allowed_sees_everything(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/threads/{CHAN}/members", _tok("alice"))
    assert status == 200, raw
    assert sorted(m["principal_id"] for m in _j(raw)["members"]) == sorted([ALICE, MEMBER])


def test_thread_members_open_works_unauthenticated(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/threads/{OPEN_THREAD}/members")
    assert status == 200, raw
    assert OPEN_MEMBER in raw


def test_thread_members_malformed_acl_entry_denies(srv):
    _seed(srv)
    _corrupt(srv)
    status, raw = _get(srv, f"/a2a/threads/{CHAN}/members", _tok("alice"))
    assert status == 403, raw
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# GET /a2a/messages/{id}/receipts
# ---------------------------------------------------------------------------

def test_message_receipts_restricted_denied_verified(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/messages/{srv.cmsg}/receipts", _tok("bob"))
    assert status == 403, raw
    _assert_no_leak(raw)
    assert ALICE not in raw


@pytest.mark.parametrize("who", ["anon", "forged-alice"])
def test_message_receipts_restricted_unverified_gets_nothing(srv, who):
    """With a verifier configured the route's own auth gate answers 401 first."""
    _seed(srv)
    status, raw = _get(srv, f"/a2a/messages/{srv.cmsg}/receipts", _tok(who))
    assert status == 401, raw
    _assert_no_leak(raw)
    assert ALICE not in raw


def test_message_receipts_standalone_unauthenticated_denied(srv_standalone):
    srv = srv_standalone
    _seed_standalone(srv)
    status, raw = _get(srv, f"/a2a/messages/{srv.cmsg}/receipts")
    assert status == 403, raw
    _assert_no_leak(raw)
    assert ALICE not in raw


def test_message_receipts_restricted_allowed_sees_everything(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/messages/{srv.cmsg}/receipts", _tok("alice"))
    assert status == 200, raw
    body = _j(raw)
    assert [r["agent_id"] for r in body["delivered"]] == [ALICE]
    assert [r["agent_id"] for r in body["read"]] == [ALICE]


def test_message_receipts_open_works_verified(srv):
    """Open channel: the route itself needs a verified caller, then no ACL applies."""
    _seed(srv)
    status, raw = _get(srv, f"/a2a/messages/{srv.omsg}/receipts", _tok("bob"))
    assert status == 200, raw
    assert _j(raw)["delivered"] == []  # bob has no receipt of his own
    status, raw = _get(srv, f"/a2a/messages/{srv.omsg}/receipts", _tok("alice"))
    assert status == 200, raw


def test_message_receipts_open_works_standalone_unauthenticated(srv_standalone):
    srv = srv_standalone
    _seed_standalone(srv)
    status, raw = _get(srv, f"/a2a/messages/{srv.omsg}/receipts")
    assert status == 200, raw
    assert [r["agent_id"] for r in _j(raw)["delivered"]] == [ALICE]


def test_message_receipts_malformed_acl_entry_denies(srv):
    _seed(srv)
    _corrupt(srv)
    status, raw = _get(srv, f"/a2a/messages/{srv.cmsg}/receipts", _tok("alice"))
    assert status == 403, raw
    _assert_no_leak(raw)


def test_message_receipts_unknown_id_keeps_master_behaviour(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/messages/987654/receipts", _tok("bob"))
    assert status == 200, raw
    assert _j(raw) == {"delivered": [], "read": []}


# ---------------------------------------------------------------------------
# GET /a2a/receipts?message_id=&agent=
# ---------------------------------------------------------------------------

def test_receipt_restricted_denied_verified(srv):
    _seed(srv)
    for agent in (ALICE, "bob"):  # bob asking for his own receipt must not bypass the gate
        status, raw = _get(srv, f"/a2a/receipts?message_id={srv.cmsg}&agent={agent}", _tok("bob"))
        assert status == 403, raw
        _assert_no_leak(raw)
        assert ALICE not in raw


@pytest.mark.parametrize("who", ["anon", "forged-alice"])
def test_receipt_restricted_unverified_gets_nothing(srv, who):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/receipts?message_id={srv.cmsg}&agent={ALICE}", _tok(who))
    assert status == 401, raw
    _assert_no_leak(raw)
    assert ALICE not in raw


def test_receipt_standalone_unauthenticated_denied(srv_standalone):
    srv = srv_standalone
    _seed_standalone(srv)
    status, raw = _get(srv, f"/a2a/receipts?message_id={srv.cmsg}&agent={ALICE}")
    assert status == 403, raw
    _assert_no_leak(raw)
    assert ALICE not in raw


def test_receipt_restricted_allowed_sees_everything(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/receipts?message_id={srv.cmsg}&agent={ALICE}", _tok("alice"))
    assert status == 200, raw
    assert "delivered_at" in _j(raw) and "seen_at" in _j(raw)


def test_receipt_open_works_verified(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/receipts?message_id={srv.omsg}&agent=bob", _tok("bob"))
    assert status == 404, raw  # bob has no receipt: master behaviour on an open channel
    status, raw = _get(srv, f"/a2a/receipts?message_id={srv.omsg}&agent={ALICE}", _tok("alice"))
    assert status == 200, raw


def test_receipt_open_works_standalone_unauthenticated(srv_standalone):
    srv = srv_standalone
    _seed_standalone(srv)
    status, raw = _get(srv, f"/a2a/receipts?message_id={srv.omsg}&agent={ALICE}")
    assert status == 200, raw


def test_receipt_malformed_acl_entry_denies(srv):
    _seed(srv)
    _corrupt(srv)
    status, raw = _get(srv, f"/a2a/receipts?message_id={srv.cmsg}&agent={ALICE}", _tok("alice"))
    assert status == 403, raw
    _assert_no_leak(raw)


def test_receipt_unknown_id_keeps_master_behaviour(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/receipts?message_id=987654&agent={ALICE}", _tok("alice"))
    assert status == 404, raw
    assert "receipt not found" in raw


def _seed_standalone(srv):
    """Standalone server (no verifier): seed with no tokens; receipts are
    written straight through the service since POST /a2a/receipts needs one."""
    _restrict(srv)
    first = srv.httpd.service_loop.run(service.a2a_send(
        POSTER, f"{CANARY} {SENDER}", thread=CHAN, data_dir=srv.data_dir))
    second = srv.httpd.service_loop.run(service.a2a_send(
        DUAL, OPEN_BODY, thread="general", data_dir=srv.data_dir))
    srv.cmsg, srv.omsg = first["id"], second["id"]
    for mid in (srv.cmsg, srv.omsg):
        srv.httpd.service_loop.run(service.a2a_record_delivered(mid, ALICE, data_dir=srv.data_dir))
