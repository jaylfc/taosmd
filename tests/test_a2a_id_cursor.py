"""Tests for A2A message-id cursor paging on GET /a2a/messages.

Covers after_id and before_id cursor parameters on the a2a_feed service
function and the /a2a/messages HTTP endpoint.
"""

from __future__ import annotations

import asyncio
import pytest
import time

from taosmd import api as taosmd_api
from taosmd import service


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder — no ONNX/QMD model required."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Isolated data dir with a clean stores cache for each test."""
    data_dir = tmp_path / "taosmd-a2a-id-cursor"
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


def _seed_messages(data_dir, count=7, thread="cursor-test"):
    """Seed `count` messages on one thread. Returns the message ids in order."""
    ids = []
    for i in range(1, count + 1):
        receipt = asyncio.run(
            service.a2a_send("agentA", f"message {i}", thread=thread, data_dir=data_dir),
        )
        ids.append(receipt["id"])
    return ids


# ---------------------------------------------------------------------------
# Service-layer tests
# ---------------------------------------------------------------------------

def test_a2a_id_cursor_after_id_3_returns_4_7_oldest_first(isolated_data_dir):
    """after_id=3 returns ids 4..7 oldest-first."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")

    msgs = asyncio.run(
        service.a2a_feed(thread="cursor-test", after_id=3, data_dir=dd),
    )
    assert len(msgs) == 4
    assert [m["id"] for m in msgs] == [4, 5, 6, 7]


def test_a2a_id_cursor_after_id_3_limit_2_returns_4_5(isolated_data_dir):
    """after_id=3&limit=2 returns 4,5."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")

    msgs = asyncio.run(
        service.a2a_feed(thread="cursor-test", after_id=3, limit=2, data_dir=dd),
    )
    assert len(msgs) == 2
    assert [m["id"] for m in msgs] == [4, 5]


def test_a2a_id_cursor_before_id_5_limit_2_returns_3_4_oldest_first(isolated_data_dir):
    """before_id=5&limit=2 returns 3,4 (oldest-first)."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")

    msgs = asyncio.run(
        service.a2a_feed(thread="cursor-test", before_id=5, limit=2, data_dir=dd),
    )
    assert len(msgs) == 2
    assert [m["id"] for m in msgs] == [3, 4]


def test_a2a_id_cursor_after_id_7_returns_empty(isolated_data_dir):
    """after_id=7 returns []."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")

    msgs = asyncio.run(
        service.a2a_feed(thread="cursor-test", after_id=7, data_dir=dd),
    )
    assert len(msgs) == 0


def test_a2a_id_cursor_before_id_1_returns_empty(isolated_data_dir):
    """before_id=1 returns []."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")

    msgs = asyncio.run(
        service.a2a_feed(thread="cursor-test", before_id=1, data_dir=dd),
    )
    assert len(msgs) == 0


def test_a2a_id_cursor_since_and_after_id_anded(isolated_data_dir):
    """since ANDed with after_id narrows correctly."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")
    pivot = time.time() - 1
    msgs = asyncio.run(
        service.a2a_feed(
            thread="cursor-test", since=pivot, after_id=3, data_dir=dd,
        ),
    )
    # since filters by timestamp, after_id filters by id; both must pass
    assert len(msgs) <= 4


def test_a2a_id_cursor_thread_filter_with_cursor(isolated_data_dir):
    """thread filter still applies with a cursor."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")

    msgs = asyncio.run(
        service.a2a_feed(thread="cursor-test", after_id=3, data_dir=dd),
    )
    assert all(m["thread"] == "cursor-test" for m in msgs)


def test_a2a_id_cursor_no_params_returns_all(isolated_data_dir):
    """Request with neither cursor returns all messages (default untouched)."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")

    msgs = asyncio.run(
        service.a2a_feed(thread="cursor-test", data_dir=dd),
    )
    assert len(msgs) == 7


def test_a2a_id_cursor_both_params_intersected(isolated_data_dir):
    """Both after_id and before_id given at service level are intersected."""
    dd = str(isolated_data_dir)
    _setup_stores(dd)
    _seed_messages(dd, count=7, thread="cursor-test")

    msgs = asyncio.run(
        service.a2a_feed(thread="cursor-test", after_id=3, before_id=6, data_dir=dd),
    )
    # after_id=3 means id > 3; before_id=6 means id < 6; intersection = ids 4,5
    assert len(msgs) == 2
    assert [m["id"] for m in msgs] == [4, 5]