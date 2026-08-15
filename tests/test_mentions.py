"""Tests for taosmd.mentions.MentionStore.

Verifies that the store uses the shared _normalise_handle helper on both
the write path (record_mentions) and the read path (get_mentioned_message_ids)
so the index and query never disagree on handle spelling.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from taosmd.mentions import MentionStore


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "mentions.db")


@pytest.fixture()
def store(db_path: str) -> MentionStore:
    s = MentionStore(db_path)
    asyncio.run(s.init())
    yield s
    asyncio.run(s.close())


class TestMentionStore:
    def test_record_and_retrieve_bare_slug(self, store: MentionStore):
        asyncio.run(store.record_mentions(1, "hey @bob", "general", 1000.0))
        rows = asyncio.run(store.get_mentioned_message_ids("bob"))
        assert [r["message_id"] for r in rows] == [1]

    def test_record_and_retrieve_at_prefixed_reader(self, store: MentionStore):
        asyncio.run(store.record_mentions(1, "hey @bob", "general", 1000.0))
        rows = asyncio.run(store.get_mentioned_message_ids("@bob"))
        assert [r["message_id"] for r in rows] == [1]

    def test_record_and_retrieve_case_insensitive(self, store: MentionStore):
        asyncio.run(store.record_mentions(1, "hey @BOB", "general", 1000.0))
        rows = asyncio.run(store.get_mentioned_message_ids("bob"))
        assert [r["message_id"] for r in rows] == [1]

    def test_record_with_recipient(self, store: MentionStore):
        asyncio.run(store.record_mentions(1, "hello", "general", 1000.0, recipient="@alice"))
        rows = asyncio.run(store.get_mentioned_message_ids("alice"))
        assert [r["message_id"] for r in rows] == [1]

    def test_mint_strip_matches_canonical_reader(self, store: MentionStore):
        asyncio.run(store.record_mentions(1, "hey @hermes", "general", 1000.0))
        rows = asyncio.run(store.get_mentioned_message_ids("hermes-20260727-001415"))
        assert [r["message_id"] for r in rows] == [1]

    def test_install_discriminator_survives(self, store: MentionStore):
        asyncio.run(store.record_mentions(1, "hey @taOS-agent-1a2b3c4d", "general", 1000.0))
        rows = asyncio.run(store.get_mentioned_message_ids("taOS-agent-1a2b3c4d"))
        assert [r["message_id"] for r in rows] == [1]

    def test_since_filter(self, store: MentionStore):
        asyncio.run(store.record_mentions(1, "hey @bob", "general", 1000.0))
        asyncio.run(store.record_mentions(2, "hey @bob", "general", 2000.0))
        rows = asyncio.run(store.get_mentioned_message_ids("bob", since=1500.0))
        assert [r["message_id"] for r in rows] == [2]

    def test_limit(self, store: MentionStore):
        for i in range(5):
            asyncio.run(store.record_mentions(i, f"hey @bob {i}", "general", float(i * 1000)))
        rows = asyncio.run(store.get_mentioned_message_ids("bob", limit=2))
        assert len(rows) == 2

    def test_no_mentions_returns_empty(self, store: MentionStore):
        rows = asyncio.run(store.get_mentioned_message_ids("nobody"))
        assert rows == []

    def test_get_mention_recipients(self, store: MentionStore):
        asyncio.run(store.record_mentions(1, "hey @bob", "general", 1000.0))
        recipients = asyncio.run(store.get_mention_recipients(1))
        assert recipients == ["bob"]
