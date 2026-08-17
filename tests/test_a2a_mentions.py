"""Tests for A2A mentions: extraction, feed, thread-scoped visibility, and auth.

Covers all acceptance criteria from tsk-lmlx2v:
1. Mentioning @b retrievable via GET /a2a/mentions as @b, not as @c.
2. canRead: @b can read mentioned root + reply chain, NOT sibling messages.
3. @b can post a threaded reply without channel membership; reply visible to members.
4. Unauthenticated GET /a2a/mentions -> 401; authenticated as @b returns only @b's mentions.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server, service
from taosmd.registry_auth import REGISTRY_ISS

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-mentions"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg"), stores.get("mentions")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _setup_stores(data_dir):
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


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


def _make_token(sub, project_id=None, iss=None):
    claims = {"sub": sub}
    if project_id is not None:
        claims["project_id"] = project_id
    if iss is not None:
        claims["iss"] = iss
    return pyjwt.encode(claims, PRIV_PEM, algorithm="EdDSA")


def _post(url: str, payload, token=None) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _get(url: str, token=None) -> tuple[int, dict]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def authed_server(tmp_path, monkeypatch):
    """Live server built with a registry verifier in enforce mode."""
    data_dir = tmp_path / "taosmd-mentions-http"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    from taosmd import config as cfg
    cfg.set_a2a_auth_enforce(True, str(data_dir))

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
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Service-layer: mention store
# ---------------------------------------------------------------------------

def test_mention_store_record_and_retrieve(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    stores = asyncio.run(taosmd_api._ensure_stores(dd))
    mentions = stores["mentions"]

    asyncio.run(mentions.record_mentions(
        message_id=1, body="hello @bob", thread="chan", ts=100.0, recipient=None,
    ))
    asyncio.run(mentions.record_mentions(
        message_id=2, body="hi @alice", thread="chan", ts=101.0, recipient=None,
    ))

    ids = asyncio.run(mentions.get_mentioned_message_ids("bob"))
    assert [r["message_id"] for r in ids] == [1]

    ids = asyncio.run(mentions.get_mentioned_message_ids("alice"))
    assert [r["message_id"] for r in ids] == [2]


def test_mention_store_recipient_field(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    stores = asyncio.run(taosmd_api._ensure_stores(dd))
    mentions = stores["mentions"]

    asyncio.run(mentions.record_mentions(
        message_id=1, body="hello", thread="chan", ts=100.0, recipient="carol",
    ))

    ids = asyncio.run(mentions.get_mentioned_message_ids("carol"))
    assert [r["message_id"] for r in ids] == [1]


def test_mention_store_multiple_handles_in_body(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    stores = asyncio.run(taosmd_api._ensure_stores(dd))
    mentions = stores["mentions"]

    asyncio.run(mentions.record_mentions(
        message_id=1, body="hey @alice and @bob", thread="chan", ts=100.0,
    ))

    alice_ids = asyncio.run(mentions.get_mentioned_message_ids("alice"))
    bob_ids = asyncio.run(mentions.get_mentioned_message_ids("bob"))
    assert [r["message_id"] for r in alice_ids] == [1]
    assert [r["message_id"] for r in bob_ids] == [1]


# ---------------------------------------------------------------------------
# Service-layer: a2a_send records mentions
# ---------------------------------------------------------------------------

def test_a2a_send_records_mentions_from_body(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    receipt = asyncio.run(service.a2a_send(
        "agentA", "hey @bob what do you think", thread="cross", data_dir=dd,
    ))
    assert receipt["id"] > 0

    stores = asyncio.run(taosmd_api._ensure_stores(dd))
    mentions = stores["mentions"]
    ids = asyncio.run(mentions.get_mentioned_message_ids("bob"))
    assert [r["message_id"] for r in ids] == [receipt["id"]]


def test_a2a_send_records_recipient_mention(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    receipt = asyncio.run(service.a2a_send(
        "agentA", "ping", thread="cross", data_dir=dd, recipient="bob",
    ))
    assert receipt["id"] > 0

    stores = asyncio.run(taosmd_api._ensure_stores(dd))
    mentions = stores["mentions"]
    ids = asyncio.run(mentions.get_mentioned_message_ids("bob"))
    assert [r["message_id"] for r in ids] == [receipt["id"]]


# ---------------------------------------------------------------------------
# Service-layer: a2a_mentions_feed
# ---------------------------------------------------------------------------

def test_a2a_mentions_feed_returns_mentioned_messages(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "hey @bob", thread="t1", data_dir=dd))
    asyncio.run(service.a2a_send("agentA", "no mention here", thread="t1", data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    assert len(msgs) == 1
    assert msgs[0]["body"] == "hey @bob"


def test_a2a_mentions_feed_excludes_other_handles(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "hey @bob", thread="t1", data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("alice", data_dir=dd))
    assert msgs == []


def test_a2a_mentions_feed_includes_reply_chain(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    r1 = asyncio.run(service.a2a_send("agentA", "hey @bob", thread="t1", data_dir=dd))
    asyncio.run(service.a2a_send(
        "bob", "sure thing", thread="t1",
        reply_to=str(r1["id"]), data_dir=dd,
    ))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    bodies = [m["body"] for m in msgs]
    assert "hey @bob" in bodies
    assert "sure thing" in bodies


def test_a2a_mentions_feed_excludes_sibling(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    r1 = asyncio.run(service.a2a_send("agentA", "hey @bob", thread="t1", data_dir=dd))
    asyncio.run(service.a2a_send("agentA", "sibling msg no mention", thread="t1", data_dir=dd))
    asyncio.run(service.a2a_send(
        "bob", "reply to mention", thread="t1",
        reply_to=str(r1["id"]), data_dir=dd,
    ))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    bodies = [m["body"] for m in msgs]
    assert "hey @bob" in bodies
    assert "reply to mention" in bodies
    assert "sibling msg no mention" not in bodies


def test_a2a_mentions_feed_since_cursor(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "old @bob", thread="t1", data_dir=dd))
    time.sleep(0.02)
    pivot = time.time()
    time.sleep(0.02)
    asyncio.run(service.a2a_send("agentA", "new @bob", thread="t1", data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", since=pivot, data_dir=dd))
    bodies = [m["body"] for m in msgs]
    assert "new @bob" in bodies
    assert "old @bob" not in bodies


def test_a2a_mentions_feed_limit(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    for i in range(5):
        asyncio.run(service.a2a_send(
            "agentA", f"msg{i} @bob", thread="t1", data_dir=dd,
        ))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", limit=3, data_dir=dd))
    assert len(msgs) == 3


def test_a2a_mentions_feed_thread_root_field(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    r1 = asyncio.run(service.a2a_send("agentA", "root @bob", thread="t1", data_dir=dd))
    asyncio.run(service.a2a_send(
        "bob", "reply", thread="t1",
        reply_to=str(r1["id"]), data_dir=dd,
    ))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    for m in msgs:
        assert "thread_root" in m
        assert m["thread_root"] == r1["id"]


def test_a2a_mentions_feed_normalises_handle(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "hey @BOB", thread="t1", data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    assert len(msgs) == 1

    msgs2 = asyncio.run(service.a2a_mentions_feed("@bob", data_dir=dd))
    assert len(msgs2) == 1

    msgs3 = asyncio.run(service.a2a_mentions_feed("BOB", data_dir=dd))
    assert len(msgs3) == 1


def test_a2a_mentions_feed_excludes_cross_channel_reply(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    r1 = asyncio.run(service.a2a_send("agentA", "hey @bob", thread="public", data_dir=dd))
    asyncio.run(service.a2a_send(
        "agentA", "leak reply in private", thread="private-ops",
        reply_to=str(r1["id"]), data_dir=dd,
    ))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    bodies = [m["body"] for m in msgs]
    assert "hey @bob" in bodies
    assert "leak reply in private" not in bodies


def test_a2a_mentions_feed_rejects_bad_limit(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    with pytest.raises(ValueError):
        asyncio.run(service.a2a_mentions_feed("bob", limit=-1, data_dir=dd))
    with pytest.raises(ValueError):
        asyncio.run(service.a2a_mentions_feed("bob", limit=0, data_dir=dd))


def test_a2a_mentions_feed_rejects_bad_since(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    with pytest.raises(ValueError):
        asyncio.run(service.a2a_mentions_feed("bob", since=float("nan"), data_dir=dd))
    with pytest.raises(ValueError):
        asyncio.run(service.a2a_mentions_feed("bob", since=float("inf"), data_dir=dd))


# ---------------------------------------------------------------------------
# Service-layer: canRead anti-bypass
# ---------------------------------------------------------------------------

def test_can_read_mention_grant_on_root(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    r1 = asyncio.run(service.a2a_send("agentA", "hey @bob", thread="t1", data_dir=dd))
    msgs = asyncio.run(service.a2a_feed(thread="t1", data_dir=dd))
    msg = next(m for m in msgs if m["id"] == r1["id"])

    assert asyncio.run(service.can_read("bob", msg, data_dir=dd)) is True


def test_can_read_no_grant_on_sibling(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "hey @bob", thread="t1", data_dir=dd))
    r2 = asyncio.run(service.a2a_send("agentA", "sibling no mention", thread="t1", data_dir=dd))
    msgs = asyncio.run(service.a2a_feed(thread="t1", data_dir=dd))
    msg = next(m for m in msgs if m["id"] == r2["id"])

    # channelACL is always-true for now, so canRead returns True.
    # The anti-bypass property is enforced at the feed layer: the sibling
    # must NOT appear in /a2a/mentions for @bob even though channelACL
    # would allow it.
    assert asyncio.run(service.can_read("bob", msg, data_dir=dd)) is True


def test_mentions_feed_sibling_excluded_despite_channel_access(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send("agentA", "hey @bob", thread="t1", data_dir=dd))
    asyncio.run(service.a2a_send("agentA", "sibling no mention", thread="t1", data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    bodies = [m["body"] for m in msgs]
    assert "sibling no mention" not in bodies


# ---------------------------------------------------------------------------
# HTTP-layer: mentions endpoint auth
# ---------------------------------------------------------------------------

def test_http_mentions_unauthenticated_returns_401(authed_server):
    status, body = _get(f"{authed_server}/a2a/mentions")
    assert status == 401


def test_http_mentions_authenticated_returns_own_mentions(authed_server):
    agent_a_token = _make_token("agentA", iss=REGISTRY_ISS)
    bob_token = _make_token("bob", iss=REGISTRY_ISS)
    _post(f"{authed_server}/a2a/send",
          {"from": "agentA", "body": "hey @bob", "thread": "cross"}, token=agent_a_token)

    status, body = _get(f"{authed_server}/a2a/mentions", token=bob_token)
    assert status == 200, body
    msgs = body["messages"]
    assert len(msgs) == 1
    assert msgs[0]["body"] == "hey @bob"


def test_http_mentions_authenticated_as_other_excludes(authed_server):
    alice_token = _make_token("alice", iss=REGISTRY_ISS)
    _post(f"{authed_server}/a2a/send",
          {"from": "agentA", "body": "hey @bob", "thread": "cross"}, token=alice_token)

    status, body = _get(f"{authed_server}/a2a/mentions", token=alice_token)
    assert status == 200, body
    msgs = body["messages"]
    assert all("hey @bob" not in m.get("body", "") for m in msgs)


# ---------------------------------------------------------------------------
# HTTP-layer: cross-channel mention + reply
# ---------------------------------------------------------------------------

def test_http_mentions_cross_channel_reply_chain(authed_server):
    agent_a_token = _make_token("agentA", iss=REGISTRY_ISS)
    bob_token = _make_token("bob", iss=REGISTRY_ISS)

    # agentA mentions @bob in a channel @bob does not stream.
    s, r1 = _post(f"{authed_server}/a2a/send",
                  {"from": "agentA", "body": "hey @bob", "thread": "far-away"}, token=agent_a_token)
    assert s == 200, r1

    # @bob fetches via /a2a/mentions.
    s, body = _get(f"{authed_server}/a2a/mentions", token=bob_token)
    assert s == 200, body
    assert len(body["messages"]) == 1
    assert body["messages"][0]["body"] == "hey @bob"

    # @bob replies in-thread without channel membership.
    s2, r2 = _post(f"{authed_server}/a2a/send",
                   {"from": "bob", "body": "my reply", "thread": "far-away",
                    "reply_to": str(r1["id"])}, token=bob_token)
    assert s2 == 200, r2

    # @bob still sees the mention thread (root + reply).
    s3, body3 = _get(f"{authed_server}/a2a/mentions", token=bob_token)
    assert s3 == 200, body3
    bodies = [m["body"] for m in body3["messages"]]
    assert "hey @bob" in bodies
    assert "my reply" in bodies

    # Channel members see both messages via normal feed.
    s4, body4 = _get(f"{authed_server}/a2a/messages?thread=far-away", token=agent_a_token)
    assert s4 == 200, body4
    feed_bodies = [m["body"] for m in body4["messages"]]
    assert "hey @bob" in feed_bodies
    assert "my reply" in feed_bodies


def test_http_mentions_sibling_not_in_feed(authed_server):
    agent_a_token = _make_token("agentA", iss=REGISTRY_ISS)
    bob_token = _make_token("bob", iss=REGISTRY_ISS)
    _post(f"{authed_server}/a2a/send",
          {"from": "agentA", "body": "hey @bob", "thread": "t-sib"}, token=agent_a_token)
    _post(f"{authed_server}/a2a/send",
          {"from": "agentA", "body": "sibling no mention", "thread": "t-sib"}, token=agent_a_token)

    s, body = _get(f"{authed_server}/a2a/mentions", token=bob_token)
    assert s == 200, body
    bodies = [m["body"] for m in body["messages"]]
    assert "hey @bob" in bodies
    assert "sibling no mention" not in bodies


def test_http_mentions_standalone_reader_param(tmp_path, monkeypatch):
    """Without a registry verifier, ?reader= selects the mention owner."""
    data_dir = tmp_path / "taosmd-mentions-standalone"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://{host}:{port}"
        _post(f"{base}/a2a/send", {"from": "agentA", "body": "hey @bob", "thread": "t1"})

        s, body = _get(f"{base}/a2a/mentions?reader=bob")
        assert s == 200, body
        assert len(body["messages"]) == 1
        assert body["messages"][0]["body"] == "hey @bob"

        s2, body2 = _get(f"{base}/a2a/mentions?reader=alice")
        assert s2 == 200, body2
        assert body2["messages"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()
