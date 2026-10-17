"""Slice 3 of the channel-ACL work: read filtering on the listing routes.

Covers /a2a/channels, /a2a/census, /a2a/members, /a2a/mentions, /a2a/inbox
and /a2a/inbox/unhandled. Identity is the verified registry token ``sub``
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




def _restrict(srv, channel=CHAN):
    members = [ALICE, POSTER, DUAL]
    status, raw = srv.admin_set({"channel": channel, "read": members, "post": members})
    assert status == 200, raw


def _forged_token(sub):
    other_priv, _ = _keypair()
    return pyjwt.encode({"sub": sub}, other_priv, algorithm="EdDSA")


def _seed(srv):
    """Restricted channel (alice, poster, dual) with a canary mention, plus an open one.

    POSTER only ever posts in the restricted channel; DUAL posts in both.
    """
    _restrict(srv)
    body = f"@bob @alice-acl3 {CANARY}"
    assert srv.send(POSTER, CHAN, body=body, token=_token(POSTER))[0] == 200
    assert srv.send(DUAL, CHAN, body=f"dual {CANARY}", token=_token(DUAL))[0] == 200
    assert srv.send(DUAL, "general", body=f"@bob {OPEN_BODY}")[0] == 200
    assert srv.send("agentA", "general", body=f"@alice-acl3 {OPEN_BODY}")[0] == 200


WHO = ["bob", "anon", "forged-alice"]


def _tok(who):
    return {
        "bob": _token("bob"), "anon": None,
        "forged-alice": _forged_token(ALICE), "alice": _token(ALICE),
    }[who]


def _assert_no_leak(raw):
    for needle in (CANARY, CHAN, POSTER):
        assert needle not in raw, needle


def _get(srv, path, token=None):
    return _call("GET", f"{srv.base}{path}", None, token)


# ---------------------------------------------------------------------------
# /a2a/channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", WHO)
def test_channels_restricted_omitted(srv, who):
    _seed(srv)
    status, raw = _get(srv, "/a2a/channels", _tok(who))
    assert status == 200, raw
    _assert_no_leak(raw)
    names = [c["channel"] for c in _j(raw)["channels"]]
    assert "general" in names
    general = [c for c in _j(raw)["channels"] if c["channel"] == "general"][0]
    assert POSTER not in general["members"]


def test_channels_restricted_allowed_sees_everything(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/channels", _tok("alice"))
    assert status == 200, raw
    chans = {c["channel"]: c for c in _j(raw)["channels"]}
    assert {"general", CHAN} <= set(chans)
    assert chans[CHAN]["members"] == sorted([POSTER, DUAL])
    assert chans[CHAN]["message_count"] == 2


def test_channels_open_works_unauthenticated(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/channels")
    assert status == 200, raw
    assert "general" in [c["channel"] for c in _j(raw)["channels"]]


def test_channels_malformed_acl_entry_denies(srv):
    _seed(srv)
    _corrupt(srv)
    status, raw = _get(srv, "/a2a/channels", _tok("alice"))
    assert status == 200, raw
    _assert_no_leak(raw)


def _corrupt(srv):
    with open(f"{srv.data_dir}/config.json") as fh:
        conf = json.load(fh)
    conf["channel_acl"][CHAN] = [ALICE]  # entry is a list, not a dict
    with open(f"{srv.data_dir}/config.json", "w") as fh:
        json.dump(conf, fh)


# ---------------------------------------------------------------------------
# /a2a/census
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", WHO)
def test_census_restricted_excluded(srv, who):
    _seed(srv)
    status, raw = _get(srv, "/a2a/census", _tok(who))
    assert status == 200, raw
    _assert_no_leak(raw)
    census = _j(raw)["census"]
    assert POSTER not in census  # only ever posted in the denied channel
    # DUAL posted in both: only the open contribution may remain.
    assert census[DUAL]["total"] == 1
    assert census[DUAL]["channels"] == {"general": 1}
    assert census["agentA"]["total"] == 1


def test_census_restricted_allowed_sees_everything(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/census", _tok("alice"))
    assert status == 200, raw
    census = _j(raw)["census"]
    assert census[POSTER] == {"total": 1, "channels": {CHAN: 1}}
    assert census[DUAL]["total"] == 2
    assert census[DUAL]["channels"] == {CHAN: 1, "general": 1}


def test_census_open_works_unauthenticated(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/census")
    assert status == 200, raw
    assert _j(raw)["census"]["agentA"]["channels"] == {"general": 1}


def test_census_malformed_acl_entry_denies(srv):
    _seed(srv)
    _corrupt(srv)
    status, raw = _get(srv, "/a2a/census", _tok("alice"))
    assert status == 200, raw
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# /a2a/members
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("who", WHO)
def test_members_restricted_denied(srv, who):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/members?channel={CHAN}", _tok(who))
    assert status == 403, raw
    _assert_no_leak(raw)
    assert DUAL not in raw


def test_members_restricted_allowed_sees_everything(srv):
    _seed(srv)
    status, raw = _get(srv, f"/a2a/members?channel={CHAN}", _tok("alice"))
    assert status == 200, raw
    assert _j(raw)["members"] == sorted([POSTER, DUAL])


def test_members_open_works_unauthenticated(srv):
    _seed(srv)
    status, raw = _get(srv, "/a2a/members?channel=general")
    assert status == 200, raw
    assert "agentA" in _j(raw)["members"]


def test_members_malformed_acl_entry_denies(srv):
    _seed(srv)
    _corrupt(srv)
    status, raw = _get(srv, f"/a2a/members?channel={CHAN}", _tok("alice"))
    assert status == 403, raw
    _assert_no_leak(raw)


# ---------------------------------------------------------------------------
# /a2a/mentions, /a2a/inbox, /a2a/inbox/unhandled
# ---------------------------------------------------------------------------

def _mentions(srv, who, reader):
    return _get(srv, f"/a2a/mentions?reader={reader}", _tok(who))


def _inbox(srv, who, consumer):
    return _get(srv, f"/a2a/inbox?consumer={consumer}", _tok(who))


def _unhandled(srv, who, consumer):
    return _get(srv, f"/a2a/inbox/unhandled?consumer={consumer}", _tok(who))


ROUTES = {"mentions": _mentions, "inbox": _inbox, "unhandled": _unhandled}


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("who", WHO)
def test_mention_routes_restricted_filtered(srv, route, who):
    _seed(srv)
    # bob is mentioned in the restricted channel; an anonymous or forged
    # caller is turned away before any row is produced. Either way no leak.
    status, raw = ROUTES[route](srv, who, "bob")
    assert status in (200, 401, 403), raw
    _assert_no_leak(raw)
    if status == 200:
        bodies = [m["body"] for m in _j(raw)["messages"]]
        assert any(OPEN_BODY in b for b in bodies)  # open-channel mention kept


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_mention_routes_open_channel_visible_to_verified_caller(srv, route):
    _seed(srv)
    status, raw = ROUTES[route](srv, "bob", "bob")
    assert status == 200, raw
    assert OPEN_BODY in raw
    _assert_no_leak(raw)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_mention_routes_restricted_allowed_sees_everything(srv, route):
    _seed(srv)
    status, raw = ROUTES[route](srv, "alice", ALICE)
    assert status == 200, raw
    assert CANARY in raw and CHAN in raw
    assert OPEN_BODY in raw


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_mention_routes_query_param_does_not_grant_access(srv, route):
    _seed(srv)
    # bob asking for alice's inbox is refused outright; either way no leak.
    status, raw = ROUTES[route](srv, "bob", ALICE)
    assert status in (200, 403), raw
    _assert_no_leak(raw)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_mention_routes_malformed_acl_entry_denies(srv, route):
    _seed(srv)
    _corrupt(srv)
    status, raw = ROUTES[route](srv, "alice", ALICE)
    assert status == 200, raw
    _assert_no_leak(raw)
