"""Tests that the A2A inbox service layer forwards to the remote server.

When ``server_url`` is configured in the data_dir's ``config.json``, the
service functions ``a2a_inbox``, ``a2a_inbox_advance``, ``a2a_ack``, and
``a2a_inbox_unhandled`` must reach the remote HTTP server, not the local
archive.

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


def _make_token(sub, priv_pem=REG_PRIV_PEM, iss=REGISTRY_ISS):
    return pyjwt.encode({"sub": sub, "iss": iss}, priv_pem, algorithm="EdDSA")


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
        "http://reg.test", opener=fake_opener, expected_iss=registry_auth.REGISTRY_ISS,
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
    asyncio_run(taosmd_service.a2a_send(
        "bob", "unhandled @agent-1", thread="general", data_dir=str(data_dir),
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

    msgs = asyncio_run(taosmd_service.a2a_inbox(
        "agent-1", data_dir=caller_data_dir,
    ))
    ack_msg = [m for m in msgs if m["body"] == "ack me remote @agent-1"]
    assert ack_msg, "seeded ack message not visible remotely"
    msg_id = ack_msg[0]["id"]

    result = asyncio_run(taosmd_service.a2a_ack(
        msg_id, "agent-1", data_dir=caller_data_dir,
    ))
    assert result.get("ok") is True
    assert result.get("id") == msg_id


# ---------------------------------------------------------------------------
# Tests: a2a_inbox_unhandled forwards to remote
# ---------------------------------------------------------------------------

def test_remote_inbox_unhandled_reaches_remote_server(caller_data_dir, authed_live_server):
    base_url, server_data_dir = authed_live_server
    asyncio_run = __import__("asyncio").run

    msgs = asyncio_run(taosmd_service.a2a_inbox_unhandled(
        "agent-1", data_dir=caller_data_dir,
    ))
    bodies = {m["body"] for m in msgs}
    assert "unhandled @agent-1" in bodies


# ---------------------------------------------------------------------------
# Tests: wrong issuer is rejected on the remote path
# ---------------------------------------------------------------------------

def test_remote_inbox_rejects_wrong_issuer(tmp_path, monkeypatch):
    """A token with the wrong issuer is rejected by the remote registry verifier."""
    server_dir = tmp_path / "taosmd-remote-wrong-iss-server"
    server_dir.mkdir()
    taosmd_config.set_a2a_auth_enforce(True, str(server_dir))
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    def fake_opener(url, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": REG_PUB_PEM})
        return json.dumps([])

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=registry_auth.REGISTRY_ISS,
    )

    wrong_token = _make_token("agent-1", iss="wrong-issuer")
    cfg_file = server_dir / "config.json"
    cfg_file.write_text(json.dumps({"server_token": wrong_token}))

    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(server_dir), verifier=verifier,
    )
    asyncio_run = __import__("asyncio").run
    asyncio_run(taosmd_api._ensure_stores(str(server_dir)))
    _patch_embedder(asyncio_run(taosmd_api._ensure_stores(str(server_dir))))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        caller_dir = tmp_path / "taosmd-remote-wrong-iss-caller"
        caller_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        caller_cfg = caller_dir / "config.json"
        caller_cfg.write_text(json.dumps({"server_url": f"http://{host}:{port}", "server_token": wrong_token}))
        taosmd_service._remote_cache.clear()
        with pytest.raises(RuntimeError, match="HTTP 401"):
            asyncio_run(taosmd_service.a2a_inbox("agent-1", data_dir=str(caller_dir)))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()
        taosmd_service._remote_cache.clear()


def test_remote_inbox_exclude_acked_by_applied(caller_data_dir, authed_live_server):
    base_url, server_data_dir = authed_live_server
    asyncio_run = __import__("asyncio").run

    # All messages are visible without the filter.
    all_msgs = asyncio_run(taosmd_service.a2a_inbox(
        "agent-1", data_dir=caller_data_dir,
    ))
    assert len(all_msgs) == 4

    # With exclude_acked_by="agent-1", messages acked by agent-1 are omitted.
    filtered = asyncio_run(taosmd_service.a2a_inbox(
        "agent-1", exclude_acked_by="agent-1", data_dir=caller_data_dir,
    ))
    assert len(filtered) == 4

    # None of the pre-seeded messages are acked, so the filter has no effect
    # yet. Ack one via the remote ack endpoint and re-query.
    msg_id = all_msgs[0]["id"]
    ack_result = asyncio_run(taosmd_service.a2a_ack(
        msg_id, "agent-1", data_dir=caller_data_dir,
    ))
    assert ack_result.get("ok") is True

    filtered_after = asyncio_run(taosmd_service.a2a_inbox(
        "agent-1", exclude_acked_by="agent-1", data_dir=caller_data_dir,
    ))
    assert len(filtered_after) == 3
    assert not any(m["id"] == msg_id for m in filtered_after)
