"""Tests for mention grant authorization on a2a_mentions_feed and a2a_inbox.

RED-FIRST: these tests FAIL on master because can_read() returns True always
and is not called by the feed/inbox paths. The fix implements mentionGrant
in can_read and wires it into both paths.
"""

from __future__ import annotations

import asyncio

import pytest

from taosmd import api as taosmd_api
from taosmd import service


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def grant_data_dir(tmp_path, monkeypatch):
    """Isolated data dir with a clean stores cache for each test."""
    data_dir = tmp_path / "taosmd-grant"
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


# ---------------------------------------------------------------------------
# a2a_mentions_feed: mention grant on thread root
# ---------------------------------------------------------------------------


def test_mentions_feed_denies_when_thread_root_has_no_grant(grant_data_dir):
    """
    Thread root (msg 1) has no mention of @bob.
    Msg 2 replies to 1 and mentions @bob.
    Msg 3 replies to 2.
    bob's feed must be EMPTY because the thread root (msg 1) grants no mention to bob.
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    # Msg 1: thread root, no mention
    r1 = asyncio.run(service.a2a_send("alice", "hello", thread="t1", data_dir=dd))
    # Msg 2: replies to 1, mentions @bob
    r2 = asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", reply_to=str(r1["id"]), data_dir=dd))
    # Msg 3: replies to 2
    r3 = asyncio.run(service.a2a_send("bob", "sure thing", thread="t1", reply_to=str(r2["id"]), data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    # RED: on master this returns 2 messages (r2, r3) because mention store finds r2
    # FIX: can_read checks mention grant on thread_root (r1), which has no @bob -> deny
    assert msgs == [], f"Expected empty feed, got {[m['body'] for m in msgs]}"


def test_mentions_feed_allows_when_thread_root_has_grant(grant_data_dir):
    """
    Thread root (msg 1) mentions @bob.
    Msg 2 replies to 1.
    bob's feed must include both messages.
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    r1 = asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", data_dir=dd))
    r2 = asyncio.run(service.a2a_send("bob", "sure thing", thread="t1", reply_to=str(r1["id"]), data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    bodies = [m["body"] for m in msgs]
    assert "hey @bob" in bodies
    assert "sure thing" in bodies
    assert len(msgs) == 2


def test_mentions_feed_carol_excluded_when_bob_mentioned(grant_data_dir):
    """
    Thread t1: alice mentions @bob.
    Thread t2: unrelated message.
    carol's feed must be empty (no mention of @carol anywhere).
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", data_dir=dd))
    asyncio.run(service.a2a_send("alice", "unrelated message", thread="t2", data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("carol", data_dir=dd))
    assert msgs == []


def test_mentions_feed_positive_control_bob_gets_t1_not_t2(grant_data_dir):
    """
    Positive control: bob receives t1 (mention + reply chain) but never t2.
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    r1 = asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", data_dir=dd))
    r2 = asyncio.run(service.a2a_send("bob", "reply in t1", thread="t1", reply_to=str(r1["id"]), data_dir=dd))
    asyncio.run(service.a2a_send("alice", "unrelated in t2", thread="t2", data_dir=dd))

    msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
    bodies = [m["body"] for m in msgs]
    threads = [m["thread"] for m in msgs]

    assert "hey @bob" in bodies
    assert "reply in t1" in bodies
    assert "unrelated in t2" not in bodies
    assert all(t == "t1" for t in threads)
    assert len(msgs) == 2


# ---------------------------------------------------------------------------
# a2a_inbox: mention grant on thread root
# ---------------------------------------------------------------------------


def test_inbox_denies_when_thread_root_has_no_grant(grant_data_dir):
    """
    Thread root (msg 1) has no mention of @bob.
    Msg 2 replies to 1 and mentions @bob.
    bob's inbox must be EMPTY because the thread root (msg 1) grants no mention to bob.
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    r1 = asyncio.run(service.a2a_send("alice", "hello", thread="t1", data_dir=dd))
    r2 = asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", reply_to=str(r1["id"]), data_dir=dd))

    msgs = asyncio.run(service.a2a_inbox("bob", limit=50, data_dir=dd))
    # RED: on master this returns 1 message (r2) because inbox finds the mention
    # FIX: can_read checks mention grant on thread_root (r1), which has no @bob -> deny
    assert msgs == [], f"Expected empty inbox, got {[m['body'] for m in msgs]}"


def test_inbox_allows_when_thread_root_has_grant(grant_data_dir):
    """
    Thread root (msg 1) mentions @bob.
    Msg 2 replies to 1 (from bob, so self-post excluded).
    bob's inbox must include the root message (addressed to bob, thread root has grant).
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    r1 = asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", data_dir=dd))
    r2 = asyncio.run(service.a2a_send("bob", "sure thing", thread="t1", reply_to=str(r1["id"]), data_dir=dd))

    msgs = asyncio.run(service.a2a_inbox("bob", limit=50, data_dir=dd))
    bodies = [m["body"] for m in msgs]
    # r2 is from bob (self-post) so excluded; only r1 should be in inbox
    assert "hey @bob" in bodies
    assert "sure thing" not in bodies  # self-post excluded
    assert len(msgs) == 1


def test_inbox_carol_excluded_when_bob_mentioned(grant_data_dir):
    """
    Thread t1: alice mentions @bob.
    Thread t2: unrelated message.
    carol's inbox must be empty (no mention of @carol, not in her thread, not recipient).
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", data_dir=dd))
    asyncio.run(service.a2a_send("alice", "unrelated message", thread="t2", data_dir=dd))

    msgs = asyncio.run(service.a2a_inbox("carol", limit=50, data_dir=dd))
    assert msgs == []


def test_inbox_positive_control_bob_gets_t1_not_t2(grant_data_dir):
    """
    Positive control: bob receives t1 (addressed message) but never t2.
    Reply from bob in t1 is self-post, excluded from inbox.
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    r1 = asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", data_dir=dd))
    r2 = asyncio.run(service.a2a_send("bob", "reply in t1", thread="t1", reply_to=str(r1["id"]), data_dir=dd))
    asyncio.run(service.a2a_send("alice", "unrelated in t2", thread="t2", data_dir=dd))

    msgs = asyncio.run(service.a2a_inbox("bob", limit=50, data_dir=dd))
    bodies = [m["body"] for m in msgs]
    threads = [m["thread"] for m in msgs]

    assert "hey @bob" in bodies
    assert "reply in t1" not in bodies  # self-post excluded
    assert "unrelated in t2" not in bodies
    assert all(t == "t1" for t in threads)
    assert len(msgs) == 1


# ---------------------------------------------------------------------------
# Mutation proof: if can_read returns True, tests fail
# ---------------------------------------------------------------------------


def test_mutation_proof_can_read_true_breaks_denial(grant_data_dir, monkeypatch):
    """
    Mutation proof: temporarily make can_read return True and verify
    the denial tests would fail (i.e. the filter is actually enforced by can_read).
    """
    _setup_stores(grant_data_dir)
    dd = str(grant_data_dir)

    # Msg 1: thread root, no mention
    r1 = asyncio.run(service.a2a_send("alice", "hello", thread="t1", data_dir=dd))
    # Msg 2: replies to 1, mentions @bob
    r2 = asyncio.run(service.a2a_send("alice", "hey @bob", thread="t1", reply_to=str(r1["id"]), data_dir=dd))

    # Monkeypatch can_read to always return True (simulating the master bug)
    original_can_read = service.can_read

    async def always_true(reader, msg, data_dir=None):
        return True

    monkeypatch.setattr(service, "can_read", always_true)

    try:
        msgs = asyncio.run(service.a2a_mentions_feed("bob", data_dir=dd))
        # With can_read=True, the mention store finds r2 and returns it
        # This proves the filter is enforced by can_read
        assert len(msgs) >= 1, "Mutation proof: can_read=True allows messages through"
        bodies = [m["body"] for m in msgs]
        assert "hey @bob" in bodies
    finally:
        monkeypatch.setattr(service, "can_read", original_can_read)