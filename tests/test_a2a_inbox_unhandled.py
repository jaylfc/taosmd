"""Tests for the A2A inbox 'unhandled' composed query.

The unhandled query returns messages past the consumer's cursor that are
addressed to the consumer AND have not been acknowledged by the consumer.

Covers:
  (a) mention before the cursor -> excluded
  (b) mention after the cursor, not acked -> included
  (c) mention after the cursor that has been acked -> excluded
  (d) positive control: the underlying inbox query returns the expected
      addressed messages before ack filtering
"""
from __future__ import annotations

import asyncio

import pytest

from taosmd import api as taosmd_api
from taosmd import service


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
def unhandled_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-unhandled"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _setup_stores(data_dir):
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


def _send(data_dir, sender, body, thread="general", recipient=None, kind="chat"):
    return asyncio.run(service.a2a_send(
        sender, body, thread=thread, recipient=recipient, kind=kind, data_dir=str(data_dir),
    ))


# ---------------------------------------------------------------------------
# Gate (a): mention before cursor -> excluded
# Gate (b): mention after cursor, not acked -> included
# Gate (c): mention after cursor, acked -> excluded
# ---------------------------------------------------------------------------

def test_a2a_inbox_unhandled_excludes_before_cursor_and_acked(unhandled_data_dir):
    _setup_stores(unhandled_data_dir)
    dd = str(unhandled_data_dir)
    consumer = "alice"

    # msg1: before cursor (will be excluded once cursor advances)
    msg1 = _send(dd, "bob", "early @alice", thread="general", recipient=None, kind="chat")
    # msg2: after cursor, not acked -> should appear in unhandled
    msg2 = _send(dd, "bob", "later @alice", thread="general", recipient=None, kind="chat")
    # msg3: after cursor, acked by alice -> excluded from unhandled
    msg3 = _send(dd, "bob", "acked @alice", thread="general", recipient=None, kind="chat")

    # Advance cursor past msg1 (to msg1's id) so msg2 and msg3 are past cursor
    asyncio.run(service.a2a_inbox_advance(consumer, msg1["id"], data_dir=dd))

    # Ack msg3 by consumer
    asyncio.run(service.a2a_ack(msg3["id"], consumer, data_dir=dd))

    # Unhandled query should return only msg2
    unhandled = asyncio.run(service.a2a_inbox_unhandled(consumer, data_dir=dd))
    assert len(unhandled) == 1
    assert unhandled[0]["id"] == msg2["id"]
    assert unhandled[0]["body"] == "later @alice"


# ---------------------------------------------------------------------------
# Positive control: inbox returns addressed messages before ack filter
# ---------------------------------------------------------------------------

def test_a2a_inbox_positive_control_before_ack_filter(unhandled_data_dir):
    _setup_stores(unhandled_data_dir)
    dd = str(unhandled_data_dir)
    consumer = "alice"

    msg1 = _send(dd, "bob", "early @alice", thread="general", kind="chat")
    msg2 = _send(dd, "bob", "later @alice", thread="general", kind="chat")
    msg3 = _send(dd, "bob", "acked @alice", thread="general", kind="chat")

    # No cursor advance yet: inbox should return all 3
    inbox = asyncio.run(service.a2a_inbox(consumer, data_dir=dd))
    ids = {m["id"] for m in inbox}
    assert msg1["id"] in ids
    assert msg2["id"] in ids
    assert msg3["id"] in ids
