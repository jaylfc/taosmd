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


@pytest.mark.asyncio
async def test_compress_keeps_source_when_gz_is_corrupt(tmp_path, monkeypatch):
    """A truncated/corrupt .gz must NOT cause the raw .jsonl to be deleted.

    compress_old_files is the only code path that deletes raw archive data,
    so a short write or power cut after unlink would lose a day of transcripts.
    """
    import gzip
    import os
    import time

    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()
    await arc.record(event_type="conversation", data={"text": "keep me"}, agent_name="a")

    jsonl = arc._archive_dir / "2026" / "11" / "08.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"text": "keep me"}\n', encoding="utf-8")
    original_bytes = jsonl.read_bytes()
    os.utime(jsonl, (time.time() - 3 * 86400, time.time() - 3 * 86400))

    real_gzip_open = gzip.open

    class _CorruptGzipFile:
        """Wraps a real gzip file but drops the second half of writes."""

        def __init__(self, *args, **kwargs):
            self._fh = real_gzip_open(*args, **kwargs)
            self._dropped = False

        def __enter__(self):
            self._fh.__enter__()
            return self

        def __exit__(self, *exc):
            return self._fh.__exit__(*exc)

        def write(self, data):
            if not self._dropped:
                self._dropped = True
                self._fh.write(data[: len(data) // 2])
            # silently drop the rest, simulating a truncated write

        def __getattr__(self, name):
            return getattr(self._fh, name)

    monkeypatch.setattr("taosmd.archive.gzip.open", _CorruptGzipFile)

    result = await arc.compress_old_files(days_old=1)
    assert result == 0, "corrupt compression must not be counted"
    assert jsonl.exists(), "raw .jsonl must survive a corrupt .gz"
    assert jsonl.read_bytes() == original_bytes, "raw .jsonl bytes must be unchanged"
    assert not jsonl.with_suffix(".jsonl.gz").exists(), "no .gz must be left behind"


@pytest.mark.asyncio
async def test_compress_round_trip_ok(tmp_path):
    """A clean .gz round-trips byte-equal and the source .jsonl is removed."""
    import gzip
    import os
    import time

    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()

    jsonl = arc._archive_dir / "2026" / "11" / "08.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"text": "round trip"}\n', encoding="utf-8")
    original_bytes = jsonl.read_bytes()
    os.utime(jsonl, (time.time() - 3 * 86400, time.time() - 3 * 86400))

    result = await arc.compress_old_files(days_old=1)
    assert result == 1, "one file compressed"
    assert not jsonl.exists(), "raw .jsonl is removed after successful compression"
    gz_path = jsonl.with_suffix(".jsonl.gz")
    assert gz_path.exists(), "the .gz must exist"
    with gzip.open(gz_path, "rb") as fh:
        assert fh.read() == original_bytes, ".gz must decompress to the original bytes"
