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
