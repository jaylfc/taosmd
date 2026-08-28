"""Tests that the A2A inbox service layer forwards to the remote server.

When ``server_url`` is configured in the data_dir's ``config.json``, the
service functions ``a2a_inbox``, ``a2a_inbox_advance``, and ``a2a_ack`` must
reach the remote HTTP server, not the local archive.

The live server runs in verify-and-warn mode so unauthenticated ``a2a_send``
pre-seeding is accepted; the caller sends the registry bearer token for the
mutating endpoints.
"""
from __future__ import annotations

import asyncio
import json
import threading

import pytest

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from taosmd import api as taosmd_api
from taosmd import config as taosmd_config
from taosmd import http_server, registry_auth, service as taosmd_service


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


def _make_token(sub, priv_pem=REG_PRIV_PEM):
    return pyjwt.encode({"sub": sub}, priv_pem, algorithm="EdDSA")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixture: live server with registry verifier (warn mode)
# ---------------------------------------------------------------------------

@pytest.fixture
def authed_live_server(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-remote-fwd"
    data_dir.mkdir()

    # Warn mode so unauthenticated a2a_send pre-seeding is accepted.
    taosmd_config.set_a2a_auth_enforce(False, str(data_dir))

    # Write server_token as a valid registry JWT so both _check_token and
    # _get_authenticated_agent_id() accept the same bearer credential.
    valid_token = _make_token("agent-1")
    cfg_file = data_dir / "config.json"
    cfg_file.write_text(json.dumps({"server_token": valid_token}))

    def fake_opener(url, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": REG_PUB_PEM})
        return json.dumps([])

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=None,
    )

    # Pre-seed via local service layer before the server starts, then clear
    # the stores cache so the server creates fresh connections in its thread.
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    asyncio_run = __import__("asyncio").run
    stores = asyncio_run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    asyncio_run(taosmd_service.a2a_send(
        "bob", "remote inbox hello @agent-1", thread="general", data_dir=str(data_dir),
    ))
    asyncio_run(taosmd_service.a2a_send(
        "bob", "advance me @agent-1", thread="general", data_dir=str(data_dir),
    ))
    asyncio_run(taosmd_service.a2a_send(
        "bob", "ack me remote @agent-1", thread="acks", data_dir=str(data_dir),
    ))
    taosmd_api._stores_cache.clear()

    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir), verifier=verifier,
    )
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}", str(data_dir)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Fixture: caller data_dir pointing at the live server
# ---------------------------------------------------------------------------

@pytest.fixture
def caller_data_dir(tmp_path, monkeypatch, authed_live_server):
    base_url, server_data_dir = authed_live_server
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    valid_token = _make_token("agent-1")
    cfg_file = caller_dir / "config.json"
    cfg_file.write_text(json.dumps({"server_url": base_url, "server_token": valid_token}))
    taosmd_service._remote_cache.clear()
    yield str(caller_dir)
    taosmd_service._remote_cache.clear()


# ---------------------------------------------------------------------------
# Tests: a2a_inbox forwards to remote
# ---------------------------------------------------------------------------

def test_remote_inbox_reaches_remote_server(caller_data_dir, authed_live_server):
    base_url, server_data_dir = authed_live_server
    # Verify via the caller path: the remote server's seeded message is visible.
    msgs = asyncio.run(taosmd_service.a2a_inbox(
        "agent-1", data_dir=caller_data_dir,
    ))
    bodies = {m["body"] for m in msgs}
    assert "remote inbox hello @agent-1" in bodies


# ---------------------------------------------------------------------------
# Tests: a2a_inbox_advance forwards to remote
# ---------------------------------------------------------------------------

def test_remote_inbox_advance_reaches_remote_server(caller_data_dir, authed_live_server):
    base_url, server_data_dir = authed_live_server
    asyncio_run = __import__("asyncio").run

    # Read the seeded message via caller path to get its id.
    msgs = asyncio_run(taosmd_service.a2a_inbox(
        "agent-1", data_dir=caller_data_dir,
    ))
    advance_msg = [m for m in msgs if m["body"] == "advance me @agent-1"]
    assert advance_msg, "seeded advance-me message not visible remotely"
    msg_id = advance_msg[0]["id"]

    result = asyncio_run(taosmd_service.a2a_inbox_advance(
        "agent-1", msg_id, data_dir=caller_data_dir,
    ))
    assert result.get("ok") is True

    # Verify via the caller path: the remote server's cursor moved.
    msgs = asyncio_run(taosmd_service.a2a_inbox(
        "agent-1", data_dir=caller_data_dir,
    ))
    assert not any(m["id"] == msg_id for m in msgs)


# ---------------------------------------------------------------------------
# Tests: a2a_ack forwards to remote
# ---------------------------------------------------------------------------

def test_remote_ack_reaches_remote_server(caller_data_dir, authed_live_server):
    base_url, server_data_dir = authed_live_server
    asyncio_run = __import__("asyncio").run

    # Read the seeded message via caller path to get its id.
    msgs = asyncio_run(taosmd_service.a2a_inbox(
        "agent-1", data_dir=caller_data_dir,
    ))
    ack_msg = [m for m in msgs if m["body"] == "ack me remote @agent-1"]
    assert ack_msg, "seeded ack-me message not visible remotely"
    msg_id = ack_msg[0]["id"]

    result = asyncio_run(taosmd_service.a2a_ack(
        msg_id, "agent-1", data_dir=caller_data_dir,
    ))
    assert result.get("ok") is True
    assert result.get("id") == msg_id
    assert "agent-1" in result.get("acked_by", [])
