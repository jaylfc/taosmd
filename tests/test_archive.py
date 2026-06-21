"""Tests for the archive aggregation helpers used by the dashboard /stats."""
from __future__ import annotations

import pytest

from taosmd.archive import ArchiveStore


@pytest.mark.asyncio
async def test_daily_counts_and_recent(tmp_path):
    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()
    rid = await arc.record(event_type="conversation", data={"text": "hello"}, agent_name="a")
    assert rid >= 0, "conversation events should be recorded"

    counts = await arc.daily_counts(days=30)
    assert isinstance(counts, list)
    assert sum(d["count"] for d in counts) >= 1
    assert all(set(d) == {"date", "count"} for d in counts)

    recent = await arc.recent(limit=5)
    assert recent, "expected at least one recent event"
    assert set(recent[0]) == {"kind", "label", "ts"}
    assert recent[0]["kind"] == "conversation"
    assert "a" in recent[0]["label"]

    assert await arc.distinct_agents() == 1
    top = await arc.top_by("agent_name", limit=5)
    assert top and top[0]["name"] == "a" and top[0]["count"] >= 1
    with pytest.raises(ValueError):
        await arc.top_by("evil; DROP TABLE", limit=5)


@pytest.mark.asyncio
async def test_scope_filtering_and_list_memories(tmp_path):
    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()
    await arc.record(event_type="conversation", data={"text": "user note one"}, agent_name="user")
    await arc.record(event_type="conversation", data={"text": "agent note"}, agent_name="bot")

    assert await arc.scoped_total() == 2
    assert await arc.scoped_total("user") == 1
    g_all = await arc.daily_counts(30)
    g_user = await arc.daily_counts(30, agent="user")
    assert sum(d["count"] for d in g_all) == 2
    assert sum(d["count"] for d in g_user) == 1

    mems = await arc.list_memories(agent="user", limit=10)
    assert len(mems) == 1
    assert mems[0]["agent"] == "user"
    assert "user note" in mems[0]["text"]
    assert set(mems[0]) == {"text", "agent", "kind", "ts"}
    assert len(await arc.list_memories(limit=10)) == 2
