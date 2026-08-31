"""Registry auth guards for the A2A inbox, advance, ack, and unhandled HTTP endpoints.

Covers:
  (a) missing Bearer token -> 401 on all four endpoints
  (b) token ``sub`` mismatch against the requested consumer -> 403 on inbox and unhandled
  (c) bad-signature token -> 401 on advance, ack, and unhandled
  (d) valid matching token -> 200 on all four endpoints
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from taosmd import api as taosmd_api
from taosmd import config as cfg, http_server, registry_auth
from taosmd.registry_auth import REGISTRY_ISS


# ---------------------------------------------------------------------------
# Keypair / token helpers
# ---------------------------------------------------------------------------

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


REG_PRIV_PEM, REG_PUB_PEM = _keypair()
OTHER_PRIV_PEM, _OTHER_PUB_PEM = _keypair()


def _make_token(sub, priv_pem=REG_PRIV_PEM, iss=REGISTRY_ISS):
    claims = {"sub": sub, "iss": iss}
    return pyjwt.encode(claims, priv_pem, algorithm="EdDSA")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request(url, method, payload=None, token=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"error": body}


def _get(url, token=None):
    return _request(url, "GET", None, token)


def _post(url, payload, token=None):
    return _request(url, "POST", payload, token)


# ---------------------------------------------------------------------------
# Server fixture with registry verifier (enforce mode)
# ---------------------------------------------------------------------------

@pytest.fixture
def authed_server(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-inbox-auth"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    cfg.set_a2a_auth_enforce(True, str(data_dir))

    def fake_opener(url, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": REG_PUB_PEM})
        return json.dumps([])

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=registry_auth.REGISTRY_ISS,
    )
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir), verifier=verifier,
    )
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Gate (a): missing token -> 401
# ---------------------------------------------------------------------------

def test_inbox_without_token_is_rejected(authed_server):
    status, _ = _get(f"{authed_server}/a2a/inbox")
    assert status == 401


def test_inbox_advance_without_token_is_rejected(authed_server):
    status, _ = _post(f"{authed_server}/a2a/inbox/advance", {"to_id": 5})
    assert status == 401


def test_ack_without_token_is_rejected(authed_server):
    status, _ = _post(f"{authed_server}/a2a/ack", {"message_id": 1})
    assert status == 401


def test_inbox_unhandled_without_token_is_rejected(authed_server):
    status, _ = _get(f"{authed_server}/a2a/inbox/unhandled")
    assert status == 401


# ---------------------------------------------------------------------------
# Gate (b): token sub mismatch against requested consumer -> 403 on inbox and unhandled
# ---------------------------------------------------------------------------

def test_inbox_consumer_mismatch_is_rejected(authed_server):
    token = _make_token("agent-A")
    status, _ = _get(f"{authed_server}/a2a/inbox?consumer=agent-B", token=token)
    assert status == 403


def test_inbox_unhandled_consumer_mismatch_is_rejected(authed_server):
    token = _make_token("agent-A")
    status, _ = _get(f"{authed_server}/a2a/inbox/unhandled?consumer=agent-B", token=token)
    assert status == 403


# ---------------------------------------------------------------------------
# Gate (c): bad-signature token -> 401 on advance, ack, and unhandled
# ---------------------------------------------------------------------------

def test_inbox_advance_with_bad_token_is_rejected(authed_server):
    bad_token = _make_token("agent-1", priv_pem=OTHER_PRIV_PEM)
    status, _ = _post(f"{authed_server}/a2a/inbox/advance", {"to_id": 5}, token=bad_token)
    assert status == 401


def test_ack_with_bad_token_is_rejected(authed_server):
    bad_token = _make_token("agent-1", priv_pem=OTHER_PRIV_PEM)
    status, _ = _post(f"{authed_server}/a2a/ack", {"message_id": 1}, token=bad_token)
    assert status == 401


def test_inbox_unhandled_with_bad_token_is_rejected(authed_server):
    bad_token = _make_token("agent-1", priv_pem=OTHER_PRIV_PEM)
    status, _ = _get(f"{authed_server}/a2a/inbox/unhandled", token=bad_token)
    assert status == 401


# ---------------------------------------------------------------------------
# Gate (d): valid matching token -> 200 on all four endpoints
# ---------------------------------------------------------------------------

def test_inbox_with_valid_matching_token_succeeds(authed_server):
    token = _make_token("agent-1")
    status, body = _get(f"{authed_server}/a2a/inbox", token=token)
    assert status == 200
    assert "messages" in body


def test_inbox_advance_with_valid_matching_token_succeeds(authed_server, tmp_path, monkeypatch):
    dd = tmp_path / "inbox-advance-ok"
    dd.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    asyncio_run = __import__("asyncio").run
    stores = asyncio_run(taosmd_api._ensure_stores(str(dd)))
    vmem = stores["vector"]

    async def _fake_embed(text, task="search_document"):
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]
    vmem.embed = _fake_embed  # type: ignore[assignment]

    token = _make_token("agent-1")
    send_status, send_body = _post(
        f"{authed_server}/a2a/send",
        {"from": "agent-1", "body": "hello @agent-1", "thread": "general"},
        token=token,
    )
    assert send_status == 200
    msg_id = send_body["id"]

    inbox_status, inbox_body = _get(f"{authed_server}/a2a/inbox", token=token)
    assert inbox_status == 200
    assert inbox_body.get("messages") == []

    status, body = _post(
        f"{authed_server}/a2a/inbox/advance",
        {"to_id": msg_id},
        token=token,
    )
    assert status == 200
    assert body.get("ok") is True


def test_ack_with_valid_matching_token_succeeds(authed_server, tmp_path, monkeypatch):
    dd = tmp_path / "ack-ok"
    dd.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    asyncio_run = __import__("asyncio").run
    stores = asyncio_run(taosmd_api._ensure_stores(str(dd)))
    vmem = stores["vector"]

    async def _fake_embed(text, task="search_document"):
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]
    vmem.embed = _fake_embed  # type: ignore[assignment]

    token = _make_token("agent-1")
    send_status, send_body = _post(
        f"{authed_server}/a2a/send",
        {"from": "agent-1", "body": "ack me", "thread": "acks"},
        token=token,
    )
    assert send_status == 200
    msg_id = send_body["id"]

    status, body = _post(
        f"{authed_server}/a2a/ack",
        {"message_id": msg_id},
        token=token,
    )
    assert status == 200
    assert body.get("ok") is True
    assert body.get("id") == msg_id


def test_inbox_unhandled_with_valid_matching_token_succeeds(authed_server, tmp_path, monkeypatch):
    dd = tmp_path / "unhandled-ok"
    dd.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    asyncio_run = __import__("asyncio").run
    stores = asyncio_run(taosmd_api._ensure_stores(str(dd)))
    vmem = stores["vector"]

    async def _fake_embed(text, task="search_document"):
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]
    vmem.embed = _fake_embed  # type: ignore[assignment]

    token = _make_token("agent-1")
    send_status, send_body = _post(
        f"{authed_server}/a2a/send",
        {"from": "agent-1", "body": "hello @agent-1", "thread": "general"},
        token=token,
    )
    assert send_status == 200

    status, body = _get(
        f"{authed_server}/a2a/inbox/unhandled",
        token=token,
    )
    assert status == 200
    assert "messages" in body


# ---------------------------------------------------------------------------
# Gate (e): wrong issuer -> 401 on all endpoints
# ---------------------------------------------------------------------------

def test_inbox_with_wrong_issuer_is_rejected(authed_server):
    bad_token = _make_token("agent-1", iss="wrong-issuer")
    status, _ = _get(f"{authed_server}/a2a/inbox", token=bad_token)
    assert status == 401


def test_inbox_advance_with_wrong_issuer_is_rejected(authed_server):
    bad_token = _make_token("agent-1", iss="wrong-issuer")
    status, _ = _post(f"{authed_server}/a2a/inbox/advance", {"to_id": 5}, token=bad_token)
    assert status == 401


def test_ack_with_wrong_issuer_is_rejected(authed_server):
    bad_token = _make_token("agent-1", iss="wrong-issuer")
    status, _ = _post(f"{authed_server}/a2a/ack", {"message_id": 1}, token=bad_token)
    assert status == 401


def test_inbox_unhandled_with_wrong_issuer_is_rejected(authed_server):
    bad_token = _make_token("agent-1", iss="wrong-issuer")
    status, _ = _get(f"{authed_server}/a2a/inbox/unhandled", token=bad_token)
    assert status == 401


# ---------------------------------------------------------------------------
# Gate (f): unknown query parameter -> 400 on unhandled
# ---------------------------------------------------------------------------

def test_inbox_unhandled_unknown_param_is_rejected(authed_server):
    token = _make_token("agent-1")
    status, _ = _get(f"{authed_server}/a2a/inbox/unhandled?foo=bar", token=token)
    assert status == 400
