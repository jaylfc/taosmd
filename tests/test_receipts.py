"""Tests for the A2A receipts system: record, get, and prune.

Covers tsk-r4j272 (fourth pass on receipts), which restores the dropped
``receipts.py`` / migration / service wrappers / ``_ensure_stores`` integration
and fixes four blockers:

* BLOCKER 1 -- suite collection: ``from taosmd import receipts`` must succeed.
* BLOCKER 2 -- all five endpoints reachable (POST /a2a/receipts,
  PATCH /a2a/receipts, GET /a2a/receipts, GET /a2a/messages/{id}/receipts,
  POST /a2a/admin/prune-receipts).
* BLOCKER 3 -- identity: receipt writes derive ``agent_id`` from the
  verified registry token, so a forged token produces no receipt row.
* BLOCKER 4 -- ``ttl_days`` is converted to an absolute timestamp, not
  passed through as epoch seconds.
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

import jwt as pyjwt  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from taosmd import api as taosmd_api  # noqa: E402
from taosmd import http_server, registry_auth, receipts  # noqa: E402
from taosmd.registry_auth import REGISTRY_ISS  # noqa: E402

# ---------------------------------------------------------------------------
# Token / keypair helpers
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

# Registry key pair -- the only one the verifier knows.
REG_PRIV_PEM, REG_PUB_PEM = _keypair()

# A second key pair that the verifier does NOT know -- used to forge tokens.
FORGED_PRIV_PEM, _FORGED_PUB_PEM = _keypair()

ADMIN_TOKEN = "test-admin-token-abc123"


def _make_token(sub, priv_pem=REG_PRIV_PEM, iss=None):
    claims = {"sub": sub}
    if iss is not None:
        claims["iss"] = iss
    return pyjwt.encode(claims, priv_pem, algorithm="EdDSA")


# ---------------------------------------------------------------------------
# HTTP helpers (POST, PATCH, GET)
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

def _post(url, payload, token=None):
    return _request(url, "POST", payload, token)

def _patch(url, payload, token=None):
    return _request(url, "PATCH", payload, token)

def _get(url, token=None):
    return _request(url, "GET", None, token)


# ---------------------------------------------------------------------------
# Server fixture with registry verifier (enforce mode)
# ---------------------------------------------------------------------------

@pytest.fixture
def authed_server(tmp_path, monkeypatch):
    """Live server with a registry verifier and admin token, no server token."""
    data_dir = tmp_path / "taosmd-receipts"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    monkeypatch.setenv("TAOSMD_ADMIN_TOKEN", ADMIN_TOKEN)

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


# ---------------------------------------------------------------------------
# ReceiptStore direct tests (no server)
# ---------------------------------------------------------------------------

def test_receipt_store_importable():
    """BLOCKER 1 guard: ``from taosmd import receipts`` succeeds."""
    from taosmd import receipts
    assert hasattr(receipts, "ReceiptStore")
    assert hasattr(receipts, "SCHEMA")


def test_receipt_store_basic():
    """ReceiptStore record/get/prune work correctly."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "receipt-test-data"
        data_dir.mkdir()
        store = receipts.ReceiptStore(db_path=str(data_dir / "a2a-receipts.db"))
        asyncio.run(store.init())
        try:
            asyncio.run(store.record_delivered(42, "agent-a", 100.0))
            asyncio.run(store.record_seen(42, "agent-a", 101.0))
            result = asyncio.run(store.get_receipts_for_message(42))
            assert "delivered" in result
            assert "read" in result
            assert len(result["delivered"]) == 1
            assert result["delivered"][0]["agent_id"] == "agent-a"
            assert len(result["read"]) == 1
            assert result["read"][0]["agent_id"] == "agent-a"
            receipt = asyncio.run(store.get_receipt(42, "agent-a"))
            assert receipt is not None
            assert "delivered_at" in receipt
            assert "seen_at" in receipt
            n = asyncio.run(store.prune(99.0))
            assert n == 0
            n = asyncio.run(store.prune(999.0))
            assert n == 1
        finally:
            asyncio.run(store.close())


def test_receipt_store_get_receipt_missing():
    """get_receipt with no matching row returns None."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "receipt-test-data"
        data_dir.mkdir()
        store = receipts.ReceiptStore(db_path=str(data_dir / "a2a-receipts.db"))
        asyncio.run(store.init())
        try:
            receipt = asyncio.run(store.get_receipt(42, ""))
            assert receipt is None
        finally:
            asyncio.run(store.close())


def test_busy_timeout_is_set():
    """Verify that _db.connect sets busy_timeout to 5000 ms."""
    import tempfile
    from taosmd import _db
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = _db.connect(str(db_path))
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert int(timeout) == 5000, f"Expected 5000, got {timeout}"
        conn.close()


def test_receipt_store_seen_idempotent():
    """Seen timestamp is monotonic -- a second call does not move it back."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "receipt-test-data"
        data_dir.mkdir()
        store = receipts.ReceiptStore(db_path=str(data_dir / "a2a-receipts.db"))
        asyncio.run(store.init())
        try:
            asyncio.run(store.record_delivered(1, "alice", 100.0))
            asyncio.run(store.record_seen(1, "alice", 101.0))
            asyncio.run(store.record_seen(1, "alice", 99.0))  # earlier -- should be ignored
            receipt = asyncio.run(store.get_receipt(1, "alice"))
            assert receipt["seen_at"] == 101.0
        finally:
            asyncio.run(store.close())


# ---------------------------------------------------------------------------
# HTTP endpoint tests with valid registry tokens
# ---------------------------------------------------------------------------

def test_post_receipts_delivered(authed_server):
    """POST /a2a/receipts records delivery under the verified identity."""
    token = _make_token("alice", iss=REGISTRY_ISS)
    status, body = _post(
        f"{authed_server}/a2a/receipts",
        {"message_id": 42},
        token=token,
    )
    assert status == 200, body
    assert body["ok"] is True


def test_patch_receipts_seen(authed_server):
    """PATCH /a2a/receipts records seen under the verified identity."""
    token = _make_token("alice", iss=REGISTRY_ISS)
    status, body = _patch(
        f"{authed_server}/a2a/receipts",
        {"message_id": 42},
        token=token,
    )
    assert status == 200, body
    assert body["ok"] is True


def test_post_receipts_no_token_rejected(authed_server):
    """Without a registry token, POST /a2a/receipts returns 401."""
    status, body = _post(
        f"{authed_server}/a2a/receipts",
        {"message_id": 42},
    )
    assert status == 401, body


def test_patch_receipts_no_token_rejected(authed_server):
    """Without a registry token, PATCH /a2a/receipts returns 401."""
    status, body = _patch(
        f"{authed_server}/a2a/receipts",
        {"message_id": 42},
    )
    assert status == 401, body


def test_get_message_receipts(authed_server):
    """GET /a2a/messages/{id}/receipts returns delivered + read lists."""
    token = _make_token("alice", iss=REGISTRY_ISS)
    _post(f"{authed_server}/a2a/receipts", {"message_id": 42}, token=token)
    _patch(f"{authed_server}/a2a/receipts", {"message_id": 42}, token=token)
    status, body = _get(f"{authed_server}/a2a/messages/42/receipts")
    assert status == 200, body
    assert "delivered" in body
    assert "read" in body
    assert len(body["delivered"]) == 1
    assert body["delivered"][0]["agent_id"] == "alice"
    assert len(body["read"]) == 1
    assert body["read"][0]["agent_id"] == "alice"


def test_get_single_receipt(authed_server):
    """GET /a2a/receipts?message_id=X&agent=Y returns one receipt."""
    token = _make_token("alice", iss=REGISTRY_ISS)
    _post(f"{authed_server}/a2a/receipts", {"message_id": 42}, token=token)
    status, body = _get(
        f"{authed_server}/a2a/receipts?message_id=42&agent=alice"
    )
    assert status == 200, body
    assert "delivered_at" in body
    assert "seen_at" in body


def test_get_single_receipt_not_found(authed_server):
    """GET /a2a/receipts for a nonexistent receipt returns 404."""
    status, body = _get(
        f"{authed_server}/a2a/receipts?message_id=999&agent=nobody"
    )
    assert status == 404, body


# ---------------------------------------------------------------------------
# BLOCKER 3: forged-token rejects must leave no receipt row
# ---------------------------------------------------------------------------

def test_forged_token_post_writes_no_receipt(authed_server):
    """BLOCKER 3: a forged (badly signed) token must NOT write a receipt row.

    The token carries ``sub=alice`` but is signed with a key the verifier
    does not know.  ``authorize`` fails the signature check, so
    ``_get_authenticated_agent_id`` returns ``None`` and the handler replies
    401 with no row inserted.
    """
    forged = _make_token("alice", priv_pem=FORGED_PRIV_PEM, iss=REGISTRY_ISS)
    status, body = _post(
        f"{authed_server}/a2a/receipts",
        {"message_id": 77},
        token=forged,
    )
    assert status == 401, body

    # Confirm the DB stays empty for message 77.
    status2, body2 = _get(f"{authed_server}/a2a/messages/77/receipts")
    assert status2 == 200, body2
    assert body2["delivered"] == []
    assert body2["read"] == []


def test_forged_token_patch_writes_no_receipt(authed_server):
    """BLOCKER 3: forged token on PATCH must not write a seen receipt."""
    forged = _make_token("alice", priv_pem=FORGED_PRIV_PEM, iss=REGISTRY_ISS)
    status, body = _patch(
        f"{authed_server}/a2a/receipts",
        {"message_id": 77},
        token=forged,
    )
    assert status == 401, body

    status2, body2 = _get(f"{authed_server}/a2a/messages/77/receipts")
    assert status2 == 200, body2
    assert body2["delivered"] == []
    assert body2["read"] == []


def test_forged_token_does_not_impersonate(authed_server):
    """BLOCKER 3: a forged token claiming sub=bob must not attribute to bob."""
    forged = _make_token("bob", priv_pem=FORGED_PRIV_PEM, iss=REGISTRY_ISS)
    status, body = _post(
        f"{authed_server}/a2a/receipts",
        {"message_id": 88},
        token=forged,
    )
    assert status == 401, body

    # Bob should have no receipts.
    status2, body2 = _get(f"{authed_server}/a2a/messages/88/receipts")
    assert status2 == 200, body2
    assert len(body2["delivered"]) == 0


# ---------------------------------------------------------------------------
# BLOCKER 4: admin prune (ttl_days units + admin gate)
# ---------------------------------------------------------------------------

def test_admin_prune_no_token_returns_rejected(authed_server):
    """Prune without an admin token is rejected."""
    status, body = _post(
        f"{authed_server}/a2a/admin/prune-receipts",
        {"ttl_days": 0},
    )
    assert status in (401, 403), body


def test_admin_prune_with_token(authed_server):
    """POST /a2a/admin/prune-receipts prunes receipts older than ttl_days."""
    token = _make_token("alice", iss=REGISTRY_ISS)
    _post(f"{authed_server}/a2a/receipts", {"message_id": 42}, token=token)

    status, body = _post(
        f"{authed_server}/a2a/admin/prune-receipts",
        {"ttl_days": 0},
        token=ADMIN_TOKEN,
    )
    assert status == 200, body
    assert "pruned" in body
    assert body["pruned"] >= 1

    # Receipt should be gone.
    status2, body2 = _get(f"{authed_server}/a2a/receipts?message_id=42&agent=alice")
    assert status2 == 404, body2


def test_admin_prune_default_ttl(authed_server):
    """Admin prune with no body uses the 30-day default."""
    token = _make_token("alice", iss=REGISTRY_ISS)
    _post(f"{authed_server}/a2a/receipts", {"message_id": 50}, token=token)

    status, body = _post(
        f"{authed_server}/a2a/admin/prune-receipts",
        {},
        token=ADMIN_TOKEN,
    )
    assert status == 200, body
    assert "pruned" in body
    assert int(body["pruned"]) >= 0


# ---------------------------------------------------------------------------
# Delivered/seen round-trip
# ---------------------------------------------------------------------------

def test_delivered_seen_round_trip(authed_server):
    """Record delivered, then seen, then read back the full receipt."""
    token = _make_token("carol", iss=REGISTRY_ISS)
    msg_id = 100

    # Deliver
    s, b = _post(f"{authed_server}/a2a/receipts", {"message_id": msg_id}, token=token)
    assert s == 200, b
    assert b["ok"] is True

    # Read back: delivered yes, read empty
    s, b = _get(f"{authed_server}/a2a/messages/{msg_id}/receipts")
    assert s == 200, b
    assert len(b["delivered"]) == 1
    assert len(b["read"]) == 0

    # Mark seen
    s, b = _patch(f"{authed_server}/a2a/receipts", {"message_id": msg_id}, token=token)
    assert s == 200, b
    assert b["ok"] is True

    # Read back: both delivered and read populated
    s, b = _get(f"{authed_server}/a2a/messages/{msg_id}/receipts")
    assert s == 200, b
    assert len(b["delivered"]) == 1
    assert len(b["read"]) == 1
    assert b["delivered"][0]["agent_id"] == "carol"
    assert b["read"][0]["agent_id"] == "carol"

    # Single receipt
    s, b = _get(f"{authed_server}/a2a/receipts?message_id={msg_id}&agent=carol")
    assert s == 200, b
    assert "delivered_at" in b
    assert b["seen_at"] is not None
