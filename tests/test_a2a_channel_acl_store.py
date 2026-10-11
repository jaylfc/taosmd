"""Slice 1 of the channel-ACL work: ACL store, admin API and the post gate.

Identity for every ACL decision is the verified registry token ``sub`` and
nothing else (never the POST ``from`` field). A restricted channel with no
verified caller is DENIED; a channel with no ACL stays open to everyone.
All state lives under ``tmp_path``.
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


# ---------------------------------------------------------------------------
# config store
# ---------------------------------------------------------------------------

def test_get_acl_defaults_open(tmp_path):
    assert cfg.get_acl("nope", data_dir=str(tmp_path)) == {"read": ["*"], "post": ["*"]}


def test_set_acl_missing_dimension_defaults_open(tmp_path):
    d = str(tmp_path)
    cfg.set_acl("c", post=["a"], data_dir=d)
    assert cfg.get_acl("c", data_dir=d) == {"read": ["*"], "post": ["a"]}


@pytest.mark.parametrize("bad", [
    ["alice"],                      # entry is a list
    "alice",                        # entry is a string
    7,                              # entry is a number
    {"read": "alice"},              # dimension is a string
    {"read": ["alice"], "post": "alice"},
    {"read": [1, 2], "post": ["a"]},  # not a list of strings
    {"read": None},
])
def test_malformed_entry_is_closed_in_store(tmp_path, bad):
    d = tmp_path
    (d / "config.json").write_text(json.dumps({"channel_acl": {"c": bad}}))
    assert cfg.get_acl("c", data_dir=str(d)) == {"read": [], "post": []}


def test_acl_read_is_memoized_until_file_changes(tmp_path, monkeypatch):
    d = str(tmp_path)
    cfg.set_acl("c", read=["a"], data_dir=d)
    calls = []
    real = cfg._read

    def counting(data_dir=None):
        calls.append(1)
        return real(data_dir)

    monkeypatch.setattr(cfg, "_read", counting)
    for _ in range(50):
        cfg.get_acl("c", data_dir=d)
    assert len(calls) <= 1
    cfg.set_acl("c", read=["b"], data_dir=d)
    assert cfg.get_acl("c", data_dir=d)["read"] == ["b"]


# ---------------------------------------------------------------------------
# admin API
# ---------------------------------------------------------------------------

def test_admin_routes_require_admin_token(srv):
    for tok in (None, "wrong", _token("alice")):
        status, _ = srv.admin_set({"channel": "secret", "read": ["alice"]}, token=tok)
        assert status in (401, 403)
        status, raw = srv.admin_get("secret", token=tok)
        assert status in (401, 403)
    assert cfg.get_acl("secret", data_dir=srv.data_dir) == {"read": ["*"], "post": ["*"]}


def test_is_admin_route_lists_new_routes(srv):
    handler = srv.httpd.RequestHandlerClass
    assert handler._is_admin_route("POST", "/a2a/admin/set-channel-acl")
    assert handler._is_admin_route("GET", "/a2a/admin/channel-acl")
    assert not handler._is_admin_route("GET", "/a2a/channels")


def test_admin_set_and_get_roundtrip(srv):
    status, raw = srv.admin_set({"channel": "secret", "read": ["alice"], "post": ["alice", "bob"]})
    assert status == 200
    status, raw = srv.admin_get("secret")
    assert status == 200
    body = _j(raw)
    assert body["acl"] == {"read": ["alice"], "post": ["alice", "bob"]}


def test_set_post_only_keeps_existing_read(srv):
    _restrict(srv, read=["alice"], post=["alice"])
    status, _ = srv.admin_set({"channel": "secret", "post": ["alice", "bob"]})
    assert status == 200
    acl = _j(srv.admin_get("secret")[1])["acl"]
    assert acl["read"] == ["alice"]
    assert acl["post"] == ["alice", "bob"]


def test_clear_must_be_real_bool(srv):
    _restrict(srv)
    for bad in ("false", "true", 1, 0, "yes"):
        status, _ = srv.admin_set({"channel": "secret", "clear": bad})
        assert status == 400
    assert cfg.get_acl("secret", data_dir=srv.data_dir) == {"read": ["alice"], "post": ["alice"]}
    status, _ = srv.admin_set({"channel": "secret", "clear": True})
    assert status == 200
    assert cfg.get_acl("secret", data_dir=srv.data_dir) == {"read": ["*"], "post": ["*"]}


def test_admin_set_validates_input(srv):
    assert srv.admin_set({"read": ["a"]})[0] == 400
    assert srv.admin_set({"channel": "c"})[0] == 400
    assert srv.admin_set({"channel": "c", "read": "alice"})[0] == 400
    assert srv.admin_set({"channel": "c", "post": [1]})[0] == 400


# ---------------------------------------------------------------------------
# post gate
# ---------------------------------------------------------------------------

def test_denied_post_is_403_and_canary_not_archived(srv):
    _restrict(srv)
    status, raw = srv.send("bob", "secret", body=CANARY, token=_token("bob"))
    assert status == 403
    assert CANARY not in raw
    assert CANARY not in srv.archive_dump()


def test_denied_import_is_403_and_canary_not_archived(srv):
    _restrict(srv)
    status, raw = srv.imp("bob", "secret", body=CANARY, token=_token("bob"))
    assert status == 403
    assert CANARY not in raw
    assert CANARY not in srv.archive_dump()


def test_import_batch_denied_if_any_channel_denied(srv):
    _restrict(srv)
    payload = {"envelopes": [
        {"from": "bob", "body": "open-one", "thread": "general"},
        {"from": "bob", "body": CANARY, "thread": "secret"},
    ]}
    status, raw = _call("POST", f"{srv.base}/a2a/import", payload, _token("bob"))
    assert status == 403
    assert "open-one" not in srv.archive_dump()
    assert CANARY not in srv.archive_dump()


def test_unauthenticated_post_to_restricted_channel_is_403(srv):
    _restrict(srv)
    status, raw = srv.send("alice", "secret", body=CANARY)  # claims to be alice, no token
    assert status == 403
    assert CANARY not in raw
    assert CANARY not in srv.archive_dump()
    status, raw = srv.imp("alice", "secret", body=CANARY)
    assert status == 403
    assert CANARY not in srv.archive_dump()


def test_unauthenticated_restricted_is_403_not_401_in_enforce_mode(srv_enforce):
    _restrict(srv_enforce)
    status, raw = srv_enforce.send("alice", "secret", body=CANARY)
    assert status == 403
    assert CANARY not in srv_enforce.archive_dump()


def test_bad_token_on_restricted_channel_is_403(srv):
    _restrict(srv)
    status, raw = srv.send("alice", "secret", body=CANARY, token="not.a.jwt")
    assert status == 403
    assert CANARY not in srv.archive_dump()


def test_post_from_field_cannot_impersonate_allowed_sender(srv):
    """R-ID: bob's verified sub is bob even if the body claims from=alice."""
    _restrict(srv)
    status, raw = srv.send("alice", "secret", body=CANARY, token=_token("bob"))
    assert status == 403
    assert CANARY not in srv.archive_dump()


def test_allowed_caller_can_post(srv):
    _restrict(srv)
    status, raw = srv.send("alice", "secret", body=CANARY, token=_token("alice"))
    assert status == 200, raw
    assert CANARY in srv.archive_dump()
    status, raw = srv.imp("alice", "secret", body=CANARY + "-imp", token=_token("alice"))
    assert status == 200, raw
    assert CANARY + "-imp" in srv.archive_dump()


def test_post_only_restriction_leaves_read_dimension_alone(srv):
    """Restricting post does not make post depend on the read list."""
    srv.admin_set({"channel": "ro", "read": ["*"], "post": ["alice"]})
    assert srv.send("bob", "ro", body=CANARY, token=_token("bob"))[0] == 403
    assert srv.send("alice", "ro", body="fine", token=_token("alice"))[0] == 200


@pytest.mark.parametrize("bad", [["alice"], "alice", 7, {"read": "alice", "post": "alice"}])
def test_malformed_entry_denies_post_over_http(srv, bad):
    """List/string/number entries and non-list dimensions deny read AND post."""
    d = srv.data_dir
    with open(f"{d}/config.json") as fh:
        conf = json.load(fh)
    conf["channel_acl"] = {"broken": bad}
    with open(f"{d}/config.json", "w") as fh:
        json.dump(conf, fh)
    # Even the name that would be "allowed" if the entry were trusted.
    for who in ("alice", "bob"):
        status, raw = srv.send(who, "broken", body=CANARY, token=_token(who))
        assert status == 403
        assert CANARY not in raw
    assert CANARY not in srv.archive_dump()
    # The resolved ACL the admin API reports is closed for both dimensions.
    acl = _j(srv.admin_get("broken")[1])["acl"]
    assert acl == {"read": [], "post": []}


def test_open_channel_unchanged_for_unauthenticated_caller(srv):
    status, raw = srv.send("anyone", "general", body="plain hello")
    assert status == 200, raw
    assert "plain hello" in srv.archive_dump()
    # An explicit ["*"] ACL is also open.
    srv.admin_set({"channel": "open2", "read": ["*"], "post": ["*"]})
    status, raw = srv.send("anyone", "open2", body="still open")
    assert status == 200, raw


def test_clearing_acl_reopens_channel(srv):
    _restrict(srv)
    assert srv.send("bob", "secret", token=_token("bob"))[0] == 403
    srv.admin_set({"channel": "secret", "clear": True})
    assert srv.send("bob", "secret", body="now ok", token=_token("bob"))[0] == 200


def test_renamed_channel_keeps_its_acl(srv):
    """R-ALIAS: an ACL set on a channel survives an admin rename."""
    _restrict(srv, channel="secret")
    status, raw = _call(
        "POST", f"{srv.base}/a2a/admin/rename-channel",
        {"from": "secret", "to": "vault"}, ADMIN,
    )
    assert status == 200, raw
    for name in ("secret", "vault"):
        status, raw = srv.send("bob", name, body=CANARY, token=_token("bob"))
        assert status == 403, name
    assert CANARY not in srv.archive_dump()
    assert srv.send("alice", "vault", body="ok", token=_token("alice"))[0] == 200


def test_set_acl_runs_on_the_service_loop(srv, monkeypatch):
    """The config.json read-modify-write must serialise on the service loop,
    not on the per-request thread (lost-update race, audit M3)."""
    seen = []
    real = cfg.set_acl

    def spy(*args, **kwargs):
        seen.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(cfg, "set_acl", spy)
    status, raw = srv.admin_set({"channel": "secret", "read": ["alice"]})
    assert status == 200, raw
    assert seen == ["taosmd-service-loop"], seen


def test_concurrent_admin_writes_do_not_lose_updates(srv, monkeypatch):
    """Two concurrent set-channel-acl calls for different channels must both
    survive. A widened read-modify-write window makes a lost update
    deterministic when the write is not serialised on the service loop."""
    import time

    real_read = cfg._read

    def slow_read(data_dir=None):
        data = real_read(data_dir)
        time.sleep(0.3)
        return data

    monkeypatch.setattr(cfg, "_read", slow_read)
    results = {}

    def fire(channel, who):
        results[channel] = srv.admin_set({"channel": channel, "post": [who]})

    threads = [
        threading.Thread(target=fire, args=("alpha", "a")),
        threading.Thread(target=fire, args=("beta", "b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert results["alpha"][0] == 200 and results["beta"][0] == 200
    monkeypatch.setattr(cfg, "_read", real_read)
    assert _j(srv.admin_get("alpha")[1])["acl"]["post"] == ["a"]
    assert _j(srv.admin_get("beta")[1])["acl"]["post"] == ["b"]
