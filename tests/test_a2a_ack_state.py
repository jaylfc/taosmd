"""Tests for A2A v2 slice 2b: acks as server state, not traffic.

Service-layer only -- no HTTP endpoints (those are slice 2c). Covers:

  (a) a2a_ack generates NO new bus message: the total A2A event count is
      unchanged after an ack;
  (b) acked_by is visible on the envelope in the existing read paths
      (a2a_feed and a2a_thread_messages);
  (c) double-ack by the same principal leaves exactly one entry in
      acked_by (idempotent).

The composed "unhandled for X" query (mentions past X's cursor minus acks)
is deferred to slice 2c -- it needs 2a's server-side cursor.
"""
from __future__ import annotations

import asyncio

import pytest

from taosmd import api as taosmd_api
from taosmd import service
from taosmd.archive import EVENT_A2A


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder -- no ONNX/QMD model required."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Isolated data dir with a clean stores cache for each test."""
    data_dir = tmp_path / "taosmd-a2a-ack"
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


def _send_message(data_dir, body: str = "ack me", thread: str = "acks") -> int:
    """Send a message and return its id."""
    receipt = asyncio.run(service.a2a_send(
        "agentA", body, thread=thread, data_dir=str(data_dir),
    ))
    return receipt["id"]


# ---------------------------------------------------------------------------
# Gate (a): ack generates NO new bus message
# ---------------------------------------------------------------------------

def test_a2a_ack_creates_no_new_message(isolated_data_dir):
    """Acknowledging a message must never create a new bus event."""
    stores = _setup_stores(isolated_data_dir)
    archive = stores["archive"]
    dd = str(isolated_data_dir)

    msg_id = _send_message(isolated_data_dir)

    before = asyncio.run(archive.count(event_type=EVENT_A2A))
    assert before == 1

    asyncio.run(service.a2a_ack(msg_id, "agentB", data_dir=dd))

    after = asyncio.run(archive.count(event_type=EVENT_A2A))
    assert after == before, "ack must not create a new bus message"


# ---------------------------------------------------------------------------
# Gate (b): acked_by visible on the envelope in the read path
# ---------------------------------------------------------------------------

def test_a2a_ack_visible_in_feed(isolated_data_dir):
    """acked_by appears on the envelope returned by a2a_feed."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    msg_id = _send_message(isolated_data_dir, thread="acks")

    # Before any ack: acked_by is omitted from the envelope (no null noise).
    msgs = asyncio.run(service.a2a_feed(thread="acks", data_dir=dd))
    assert len(msgs) == 1
    assert "acked_by" not in msgs[0]

    asyncio.run(service.a2a_ack(msg_id, "agentB", data_dir=dd))

    msgs = asyncio.run(service.a2a_feed(thread="acks", data_dir=dd))
    assert msgs[0]["acked_by"] == ["agentB"]


def test_a2a_ack_visible_in_thread_messages(isolated_data_dir):
    """acked_by appears on the envelope returned by a2a_thread_messages."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    msg_id = _send_message(isolated_data_dir, thread="acks")

    asyncio.run(service.a2a_ack(msg_id, "agentB", data_dir=dd))

    result = asyncio.run(service.a2a_thread_messages(thread="acks", data_dir=dd))
    msgs = result["messages"]
    assert len(msgs) == 1
    assert msgs[0]["acked_by"] == ["agentB"]


# ---------------------------------------------------------------------------
# Gate (c): idempotent double-ack by the same principal
# ---------------------------------------------------------------------------

def test_a2a_ack_double_ack_is_idempotent(isolated_data_dir):
    """A second ack by the same principal does not duplicate the entry."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    msg_id = _send_message(isolated_data_dir)

    asyncio.run(service.a2a_ack(msg_id, "agentB", data_dir=dd))
    asyncio.run(service.a2a_ack(msg_id, "agentB", data_dir=dd))

    msgs = asyncio.run(service.a2a_feed(thread="acks", data_dir=dd))
    assert len(msgs) == 1
    assert msgs[0]["acked_by"] == ["agentB"]
    assert len(msgs[0]["acked_by"]) == 1


def test_a2a_ack_distinct_principals_both_recorded(isolated_data_dir):
    """Two distinct principals each get their own acked_by entry."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    msg_id = _send_message(isolated_data_dir)

    asyncio.run(service.a2a_ack(msg_id, "agentB", data_dir=dd))
    asyncio.run(service.a2a_ack(msg_id, "agentC", data_dir=dd))

    msgs = asyncio.run(service.a2a_feed(thread="acks", data_dir=dd))
    assert len(msgs) == 1
    assert set(msgs[0]["acked_by"]) == {"agentB", "agentC"}
    assert len(msgs[0]["acked_by"]) == 2
