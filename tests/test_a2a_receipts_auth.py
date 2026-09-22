"""Tests for A2A receipt READ endpoint authentication.

Covers tsk-kscoet: gate the two receipt READ endpoints on a verified registry
identity, with three-way behaviour:

- registry verifier configured, no or bad token -> 401
- registry verifier configured, valid token -> 200
- no registry verifier configured (standalone) -> 200, behaviour unchanged
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt as pyjwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from taosmd import api as taosmd_api
from taosmd import http_server, registry_auth, receipts
from taosmd.registry_auth import REGISTRY_ISS


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
FORGED_PRIV_PEM, _FORGED_PUB_PEM = _keypair()


def _make_token(sub, priv_pem=REG_PRIV_PEM, iss=None):
    claims = {"sub": sub}
    if iss is not None:
        claims["iss"] = iss
    return pyjwt.encode(claims, priv_pem, algorithm="EdDSA")


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


@pytest.fixture
def authed_server(tmp_path, monkeypatch):
    """Live server with a registry verifier, no server token."""
    data_dir = tmp_path / "taosmd-receipts-auth"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    def fake_opener(url, token=None):
        if url.endswith("pubkey"):
            return json.dumps({"pubkey": REG_PUB_PEM})
        return json.dumps([])

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=None,
    )
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir), verifier=verifier,
    )
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))

    async def _fake_embed(text, task="search_document"):
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]
    stores["vector"].embed = _fake_embed  # type: ignore[assignment]

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        httpd.service_loop.close()


@pytest.fixture
def standalone_server(tmp_path, monkeypatch):
    """Live server without a registry verifier (standalone mode)."""
    data_dir = tmp_path / "taosmd-receipts-standalone"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir), verifier=None,
    )
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))

    async def _fake_embed(text, task="search_document"):
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]
    stores["vector"].embed = _fake_embed  # type: ignore[assignment]

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}", str(data_dir)
    finally:
        httpd.shutdown()
        httpd.server_close()
        httpd.service_loop.close()


def _seed_receipt(server_url, token):
    """POST a delivered receipt so there is something to read."""
    _post(f"{server_url}/a2a/receipts", {"message_id": 42}, token=token)


# ---------------------------------------------------------------------------
# Registry-verifier-configured server: six cases, two routes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "route,make_url",
    [
        (
            "message_receipts",
            lambda base: f"{base}/a2a/messages/42/receipts",
        ),
        (
            "single_receipt",
            lambda base: f"{base}/a2a/receipts?message_id=42&agent=alice",
        ),
    ],
    ids=["message_receipts", "single_receipt"],
)
class TestReceiptReadAuth:
    """Each route is tested independently for all three auth arms."""

    def test_missing_token_returns_401(self, authed_server, route, make_url):
        """Without a registry token the read returns 401."""
        status, body = _get(make_url(authed_server))
        assert status == 401, body
        assert "error" in body

    def test_forged_token_returns_401(self, authed_server, route, make_url):
        """A forged (badly signed) token is rejected with 401."""
        forged = _make_token("alice", priv_pem=FORGED_PRIV_PEM, iss=REGISTRY_ISS)
        status, body = _get(make_url(authed_server), token=forged)
        assert status == 401, body
        assert "error" in body

    def test_valid_token_returns_200(self, authed_server, route, make_url):
        """A valid registry token returns 200 and receipt data."""
        token = _make_token("alice", iss=REGISTRY_ISS)
        _seed_receipt(authed_server, token)
        status, body = _get(make_url(authed_server), token=token)
        assert status == 200, body


# ---------------------------------------------------------------------------
# Standalone server: both reads must still answer 200
# ---------------------------------------------------------------------------

class TestReceiptReadStandalone:
    """Without a registry verifier, reads remain open."""

    def test_message_receipts_returns_200(self, standalone_server):
        """GET /a2a/messages/{id}/receipts returns 200 in standalone mode."""
        base_url, _ = standalone_server
        status, body = _get(f"{base_url}/a2a/messages/42/receipts")
        assert status == 200, body
        assert "delivered" in body
        assert "read" in body

    def test_single_receipt_returns_200(self, standalone_server):
        """GET /a2a/receipts?message_id=X&agent=Y returns 200 in standalone mode."""
        import asyncio
        base_url, data_dir = standalone_server
        db_path = str(Path(data_dir) / "a2a-receipts.db")
        store = receipts.ReceiptStore(db_path=db_path)
        asyncio.run(store.init())
        try:
            asyncio.run(store.record_delivered(42, "alice", 100.0))
            asyncio.run(store.record_seen(42, "alice", 101.0))
        finally:
            asyncio.run(store.close())
        status, body = _get(
            f"{base_url}/a2a/receipts?message_id=42&agent=alice"
        )
        assert status == 200, body
        assert "delivered_at" in body
