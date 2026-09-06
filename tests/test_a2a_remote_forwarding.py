"""Tests that the A2A inbox service layer forwards to the remote server.

When ``server_url`` is configured in the data_dir's ``config.json``, the
service functions ``a2a_inbox``, ``a2a_inbox_advance``, ``a2a_ack``, and
``a2a_inbox_unhandled`` must reach the remote HTTP server, not the local
archive.

The live server runs in verify-and-warn mode so unauthenticated ``a2a_send``
pre-seeding is accepted; the caller sends the registry bearer token for the
mutating endpoints.

Also contains the RED proof for Defect 1: service.a2a_* functions forward
``data_dir`` positionally to ``RemoteClient``, but the remote methods declare
``**_opts`` which only captures keyword arguments.  Every one of the four
thread-membership operations raises ``TypeError`` whenever a remote server
URL is configured.  Those tests drive a recording fake remote and assert both
that the call arrives AND that it carries ``data_dir`` as a keyword argument.
An existing forwarded function (``a2a_threads``) serves as a positive control
and stays green.
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
from taosmd import http_server, registry_auth, service
from taosmd import service as taosmd_service


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


def test_remote_inbox_unhandled_hits_unhandled_endpoint(
    caller_data_dir, authed_live_server, monkeypatch,
):
    """a2a_inbox_unhandled must hit GET /a2a/inbox/unhandled on the remote
    server, not fall through to a2a_inbox's own remote forward at GET
    /a2a/inbox.

    When the unhandled remote forward in service.a2a_inbox_unhandled is
    neutered, the call falls through to a2a_inbox (which has its own remote
    forward) and the request lands on /a2a/inbox instead.  The two endpoints
    return the same message set when nothing is acked, so a content assertion
    cannot catch the regression -- only the request URL can.
    """
    asyncio_run = asyncio.run

    taosmd_service._remote_cache.clear()
    client = taosmd_service._get_remote(caller_data_dir)

    requested_paths: list[str] = []
    original_run = client._run

    async def _spy_run(method, path, body=None, params=None):
        requested_paths.append(path)
        return await original_run(method, path, body, params)

    monkeypatch.setattr(client, "_run", _spy_run)

    asyncio_run(taosmd_service.a2a_inbox_unhandled(
        "agent-1", data_dir=caller_data_dir,
    ))

    assert "/a2a/inbox/unhandled" in requested_paths
    assert "/a2a/inbox" not in requested_paths


# ---------------------------------------------------------------------------
# Tests: exclude_acked_by forwarded and applied on remote path
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RED proof for Defect 1: thread-membership functions forward ``data_dir``
# positionally to ``RemoteClient``, whose remote methods declare ``**_opts``
# which only captures keyword arguments.
# ---------------------------------------------------------------------------

class RecordingRemote:
    """Minimal stand-in for :class:`~taosmd.remote.RemoteClient`.

    Mirrors the four membership method signatures (each ends in ``**_opts``)
    plus ``a2a_threads`` as a positive control.  Records every call so the
    tests can assert on the received arguments.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def a2a_create_thread(self, thread, participants, agent, **_opts) -> dict:
        self.calls.append(("a2a_create_thread", thread, participants, agent, _opts))
        return {"thread": thread, "created": True, "active_members": []}

    async def a2a_list_members(self, thread, **_opts) -> list[dict]:
        self.calls.append(("a2a_list_members", thread, _opts))
        return []

    async def a2a_add_member(self, thread, principal_id, agent, **_opts) -> dict:
        self.calls.append(("a2a_add_member", thread, principal_id, agent, _opts))
        return {"thread": thread, "principal_id": principal_id, "added": True}

    async def a2a_remove_member(self, thread, principal_id, agent, **_opts) -> dict:
        self.calls.append(("a2a_remove_member", thread, principal_id, agent, _opts))
        return {"thread": thread, "principal_id": principal_id, "removed": True}

    async def a2a_threads(self, *, principal=None, **_opts) -> list[dict]:
        self.calls.append(("a2a_threads", principal, _opts))
        return [{"thread": "t", "kind": "channel", "participants": [],
                 "last_message": {}}]


@pytest.fixture
def patched_remote(monkeypatch):
    """Patch ``_get_remote`` so the service layer always uses our fake remote."""
    remote = RecordingRemote()
    monkeypatch.setattr(service, "_get_remote", lambda data_dir=None: remote)
    return remote


DD = "/fake/data/dir"


def test_a2a_create_thread_forwards_data_dir_by_keyword(patched_remote):
    """``service.a2a_create_thread`` must forward ``data_dir`` as a keyword so
    the remote method (which only accepts ``**_opts`` for extras) receives it."""
    asyncio.run(service.a2a_create_thread(
        "proj-x", ["alice"], "carol", data_dir=DD,
    ))
    assert len(patched_remote.calls) == 1
    name = patched_remote.calls[0][0]
    assert name == "a2a_create_thread"
    opts = patched_remote.calls[0][-1]
    assert opts.get("data_dir") == DD


def test_a2a_list_members_forwards_data_dir_by_keyword(patched_remote):
    """``service.a2a_list_members`` must forward ``data_dir`` as a keyword."""
    asyncio.run(service.a2a_list_members("thread-1", data_dir=DD))
    assert len(patched_remote.calls) == 1
    assert patched_remote.calls[0][0] == "a2a_list_members"
    opts = patched_remote.calls[0][-1]
    assert opts.get("data_dir") == DD


def test_a2a_add_member_forwards_data_dir_by_keyword(patched_remote):
    """``service.a2a_add_member`` must forward ``data_dir`` as a keyword."""
    asyncio.run(service.a2a_add_member("t1", "dave", "carol", data_dir=DD))
    assert len(patched_remote.calls) == 1
    assert patched_remote.calls[0][0] == "a2a_add_member"
    opts = patched_remote.calls[0][-1]
    assert opts.get("data_dir") == DD


def test_a2a_remove_member_forwards_data_dir_by_keyword(patched_remote):
    """``service.a2a_remove_member`` must forward ``data_dir`` as a keyword."""
    asyncio.run(service.a2a_remove_member("t1", "alice", "carol", data_dir=DD))
    assert len(patched_remote.calls) == 1
    assert patched_remote.calls[0][0] == "a2a_remove_member"
    opts = patched_remote.calls[0][-1]
    assert opts.get("data_dir") == DD


def test_a2a_threads_positive_control(patched_remote):
    """``service.a2a_threads`` already forwards ``principal`` by keyword --
    positive control that must stay green while the four membership forwards
    are broken."""
    asyncio.run(service.a2a_threads(principal="carol", data_dir=DD))
    assert len(patched_remote.calls) == 1
    assert patched_remote.calls[0][0] == "a2a_threads"
    principal, opts = patched_remote.calls[0][1], patched_remote.calls[0][2]
    assert principal == "carol"
    assert opts.get("data_dir") == DD
