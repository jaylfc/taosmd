"""Tests for a2a_inbox and a2a_inbox_advance (service layer).

Gates:
  (a) inbox excludes self-posts and alarm-kind by default
  (b) fetch does not advance the cursor, advance does
  (c) the cursor survives a fresh stores cache (persistence)
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
def inbox_data_dir(tmp_path, monkeypatch):
    """Isolated data dir with a clean stores cache for each test."""
    data_dir = tmp_path / "taosmd-inbox"
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


def _seed_inbox_fixture(data_dir):
    """Create 20 messages: 4 addressed to alice, rest not addressed or excluded.

    Messages 5 to 8 mention ``@alice`` on purpose.  They are addressed by every
    other rule the query applies, so the ONLY thing that can keep them out of
    the result is the default kind exclusion.  If that filter is removed the
    fixture yields 8 instead of 4, which is what makes gate (a) discriminate
    the kind half rather than assert it vacuously.
    """
    dd = str(data_dir)
    consumer = "alice"

    messages = [
        # (sender, body, thread, recipient, kind)
        ("bob", "hey @alice", "general", None, "chat"),           # 1: mention -> ADDRESSED
        ("alice", "self post", "general", None, "chat"),          # 2: self-post -> EXCLUDED
        ("bob", "in your thread", "alice", None, "chat"),         # 3: owned thread -> ADDRESSED
        ("bob", "for alice", "general", "alice", "chat"),         # 4: direct recipient -> ADDRESSED
        ("bob", "@alice alarm!", "general", None, "alarm"),       # 5: addressed, alarm-kind -> EXCLUDED
        ("bob", "@alice ack this", "general", None, "ack"),       # 6: addressed, ack-kind -> EXCLUDED
        ("bob", "@alice receipt", "general", None, "receipt"),    # 7: addressed, receipt-kind -> EXCLUDED
        ("bob", "@alice digest", "general", None, "digest"),      # 8: addressed, digest-kind -> EXCLUDED
        ("bob", "random chat", "general", None, "chat"),          # 9: not addressed
        ("bob", "another chat", "general", None, "chat"),         # 10: not addressed
        ("charlie", "hey @bob", "general", None, "chat"),         # 11: mentions bob, not alice
        ("bob", "bob thread", "bob", None, "chat"),               # 12: bob's thread, not alice
        ("bob", "to charlie", "general", "charlie", "chat"),      # 13: recipient is charlie
        ("alice", "my thread", "alice", None, "chat"),            # 14: self-post on owned thread
        ("bob", "system msg", "general", None, "system"),         # 15: not addressed
        ("bob", "review", "general", None, "review"),             # 16: not addressed
        ("charlie", "@alice hi", "general", None, "chat"),        # 17: mention -> ADDRESSED
        ("bob", "plain", "general", None, "chat"),                # 18: not addressed
        ("bob", "plain2", "general", None, "chat"),               # 19: not addressed
        ("bob", "plain3", "general", None, "chat"),               # 20: not addressed
    ]

    ids = []
    for sender, body, thread, recipient, kind in messages:
        receipt = asyncio.run(service.a2a_send(
            sender, body, thread=thread, recipient=recipient, kind=kind, data_dir=dd,
        ))
        ids.append(receipt["id"])
    return ids, consumer


# ---------------------------------------------------------------------------
# Gate (a): inbox excludes self-posts and alarm-kind
# ---------------------------------------------------------------------------

def test_a2a_inbox_excludes_self_and_alarm_kind(inbox_data_dir):
    """Exactly 4 of 20 messages are addressed to alice; self-posts and alarm
    kinds are excluded by default."""
    _setup_stores(inbox_data_dir)
    dd = str(inbox_data_dir)
    _seed_inbox_fixture(inbox_data_dir)

    msgs = asyncio.run(service.a2a_inbox("alice", limit=50, data_dir=dd))

    assert len(msgs) == 4
    bodies = {m["body"] for m in msgs}
    assert "hey @alice" in bodies
    assert "in your thread" in bodies
    assert "for alice" in bodies
    assert "@alice hi" in bodies

    senders = {m["from"] for m in msgs}
    assert "alice" not in senders

    kinds = {m["kind"] for m in msgs}
    assert "alarm" not in kinds
    assert "ack" not in kinds
    assert "receipt" not in kinds
    assert "digest" not in kinds

    # The four excluded-kind messages are addressed to alice by mention, so
    # their absence is attributable to the kind filter and nothing else.
    assert "@alice alarm!" not in bodies
    assert "@alice ack this" not in bodies
    assert "@alice receipt" not in bodies
    assert "@alice digest" not in bodies


def test_a2a_inbox_include_kinds_widens_the_default_exclusion(inbox_data_dir):
    """``include_kinds`` re-admits a kind the default excludes, and only that
    kind."""
    _setup_stores(inbox_data_dir)
    dd = str(inbox_data_dir)
    _seed_inbox_fixture(inbox_data_dir)

    widened = asyncio.run(
        service.a2a_inbox("alice", limit=50, include_kinds=["alarm"], data_dir=dd)
    )

    assert len(widened) == 5
    bodies = {m["body"] for m in widened}
    assert "@alice alarm!" in bodies
    assert "@alice ack this" not in bodies
    assert "@alice receipt" not in bodies
    assert "@alice digest" not in bodies


# ---------------------------------------------------------------------------
# Gate (b): fetch does not advance the cursor, advance does
# ---------------------------------------------------------------------------

def test_a2a_inbox_fetch_does_not_advance_cursor(inbox_data_dir):
    """Reading the inbox does not advance the cursor; a2a_inbox_advance does."""
    _setup_stores(inbox_data_dir)
    dd = str(inbox_data_dir)
    _seed_inbox_fixture(inbox_data_dir)

    first_read = asyncio.run(service.a2a_inbox("alice", limit=50, data_dir=dd))
    assert len(first_read) == 4

    # Second read without advance must return the same messages
    second_read = asyncio.run(service.a2a_inbox("alice", limit=50, data_dir=dd))
    assert len(second_read) == 4
    assert [m["id"] for m in second_read] == [m["id"] for m in first_read]

    # Advance cursor to the last seen id
    last_id = first_read[-1]["id"]
    asyncio.run(service.a2a_inbox_advance("alice", last_id, data_dir=dd))

    # Subsequent read must be empty (cursor moved past all messages)
    third_read = asyncio.run(service.a2a_inbox("alice", limit=50, data_dir=dd))
    assert len(third_read) == 0


# ---------------------------------------------------------------------------
# Gate (c): cursor survives a fresh stores cache (persistence)
# ---------------------------------------------------------------------------

def test_a2a_inbox_cursor_survives_stores_cache_reset(inbox_data_dir):
    """Cursor is persisted in the store, not module-global memory."""
    _setup_stores(inbox_data_dir)
    dd = str(inbox_data_dir)
    _seed_inbox_fixture(inbox_data_dir)

    first_read = asyncio.run(service.a2a_inbox("alice", limit=50, data_dir=dd))
    assert len(first_read) == 4

    last_id = first_read[-1]["id"]
    asyncio.run(service.a2a_inbox_advance("alice", last_id, data_dir=dd))

    # Simulate a fresh process: clear the stores cache and re-initialise
    taosmd_api._stores_cache.clear()
    _setup_stores(inbox_data_dir)

    # Cursor must still be at last_id, so inbox returns empty
    fresh_read = asyncio.run(service.a2a_inbox("alice", limit=50, data_dir=dd))
    assert len(fresh_read) == 0
