"""Tests for A2A receipt authorization (tsk-e6pjg2).

A receipt is private to the agent it is about, except that the message's
SENDER may see all receipts for their own message.

Covers:
- /a2a/receipts: cross-agent read returns 403; sender can read any receipt;
  agent can read own receipt.
- /a2a/messages/{id}/receipts: sender sees all rows; non-sender sees only own.
- Standalone mode (no registry verifier): behaviour unchanged.
"""

from __future__ import annotations

import asyncio
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
    data_dir = tmp_path / "taosmd-receipts-authz"
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


def _seed_message(server_url, sender_token):
    """POST a message from sender and return its id."""
    status, body = _post(
        f"{server_url}/a2a/send",
        {"from": "agent-a", "body": "hello from A", "thread": "general"},
        token=sender_token,
    )
    assert status == 200, body
    return body["id"]


def _seed_receipt(server_url, token, message_id):
    """POST a delivered receipt for the token's sub on message_id."""
    _post(f"{server_url}/a2a/receipts", {"message_id": message_id}, token=token)


# ---------------------------------------------------------------------------
# /a2a/receipts: single-receipt authorization
# ---------------------------------------------------------------------------


class TestSingleReceiptAuthorization:
    """GET /a2a/receipts?message_id=X&agent=Y authorization rules."""

    def test_cross_agent_read_returns_403(self, authed_server):
        """C asking for B's receipt on A's message returns 403."""
        token_a = _make_token("agent-a")
        token_b = _make_token("agent-b")
        token_c = _make_token("agent-c")

        msg_id = _seed_message(authed_server, token_a)
        _seed_receipt(authed_server, token_b, msg_id)

        status, body = _get(
            f"{authed_server}/a2a/receipts?message_id={msg_id}&agent=agent-b",
            token=token_c,
        )
        assert status == 403, body

    def test_own_receipt_read_returns_200(self, authed_server):
        """B can read B's own receipt."""
        token_b = _make_token("agent-b")

        msg_id = _seed_message(authed_server, _make_token("agent-a"))
        _seed_receipt(authed_server, token_b, msg_id)

        status, body = _get(
            f"{authed_server}/a2a/receipts?message_id={msg_id}&agent=agent-b",
            token=token_b,
        )
        assert status == 200, body
        assert "delivered_at" in body

    def test_sender_can_read_any_receipt(self, authed_server):
        """A (sender) can read B's receipt."""
        token_a = _make_token("agent-a")
        token_b = _make_token("agent-b")

        msg_id = _seed_message(authed_server, token_a)
        _seed_receipt(authed_server, token_b, msg_id)

        status, body = _get(
            f"{authed_server}/a2a/receipts?message_id={msg_id}&agent=agent-b",
            token=token_a,
        )
        assert status == 200, body
        assert "delivered_at" in body


# ---------------------------------------------------------------------------
# /a2a/messages/{id}/receipts: listing authorization
# ---------------------------------------------------------------------------


class TestMessageReceiptsAuthorization:
    """GET /a2a/messages/{id}/receipts authorization rules."""

    def test_sender_sees_all_receipts(self, authed_server):
        """A (sender) listing receipts for A's message sees all rows."""
        token_a = _make_token("agent-a")
        token_b = _make_token("agent-b")
        token_c = _make_token("agent-c")

        msg_id = _seed_message(authed_server, token_a)
        _seed_receipt(authed_server, token_b, msg_id)
        _seed_receipt(authed_server, token_c, msg_id)

        status, body = _get(
            f"{authed_server}/a2a/messages/{msg_id}/receipts",
            token=token_a,
        )
        assert status == 200, body
        assert len(body.get("delivered", [])) == 2
        agent_ids = {r["agent_id"] for r in body["delivered"]}
        assert agent_ids == {"agent-b", "agent-c"}

    def test_non_sender_sees_own_row_only(self, authed_server):
        """C listing receipts for A's message sees only C's row."""
        token_a = _make_token("agent-a")
        token_b = _make_token("agent-b")
        token_c = _make_token("agent-c")

        msg_id = _seed_message(authed_server, token_a)
        _seed_receipt(authed_server, token_b, msg_id)
        _seed_receipt(authed_server, token_c, msg_id)

        status, body = _get(
            f"{authed_server}/a2a/messages/{msg_id}/receipts",
            token=token_c,
        )
        assert status == 200, body
        assert len(body.get("delivered", [])) == 1
        assert body["delivered"][0]["agent_id"] == "agent-c"


# ---------------------------------------------------------------------------
# Standalone mode: behaviour unchanged
# ---------------------------------------------------------------------------


class TestStandaloneModeUnchanged:
    """Without a registry verifier, cross-agent reads still work."""

    def test_cross_agent_single_receipt_200(self, standalone_server):
        """In standalone mode, C can read B's receipt (no auth enforced)."""
        base_url, data_dir = standalone_server

        db_path = str(Path(data_dir) / "a2a-receipts.db")
        store = receipts.ReceiptStore(db_path=db_path)
        asyncio.run(store.init())
        try:
            asyncio.run(store.record_delivered(42, "agent-b", 100.0))
            asyncio.run(store.record_seen(42, "agent-b", 101.0))
        finally:
            asyncio.run(store.close())

        status, body = _get(
            f"{base_url}/a2a/receipts?message_id=42&agent=agent-b"
        )
        assert status == 200, body
        assert "delivered_at" in body

    def test_cross_agent_message_receipts_200(self, standalone_server):
        """In standalone mode, listing receipts returns all rows."""
        base_url, data_dir = standalone_server

        db_path = str(Path(data_dir) / "a2a-receipts.db")
        store = receipts.ReceiptStore(db_path=db_path)
        asyncio.run(store.init())
        try:
            asyncio.run(store.record_delivered(42, "agent-b", 100.0))
            asyncio.run(store.record_delivered(42, "agent-c", 200.0))
            asyncio.run(store.record_seen(42, "agent-b", 101.0))
        finally:
            asyncio.run(store.close())

        status, body = _get(
            f"{base_url}/a2a/messages/42/receipts"
        )
        assert status == 200, body
        assert len(body.get("delivered", [])) == 2
