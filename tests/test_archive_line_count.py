"""Tests for the in-memory archive line-count optimisation (issue #95).

record() used to re-read the entire daily JSONL on every write to compute
the line_number. It now keeps an in-memory counter (self._line_counts) seeded
from disk once per path. These tests assert the counter stays consistent with
the file and survives a daily-file rollover (a new path is seeded fresh).
"""

from __future__ import annotations

import asyncio

from taosmd.archive import ArchiveStore


# Two timestamps in different UTC months -> different daily paths. The
# archive rolls its open file over by parent directory (year/month), so the
# rollover case needs a month boundary, not just a day boundary.
DAY_ONE = 1_700_000_000.0   # 2023-11-14
DAY_TWO = 1_702_600_000.0   # 2023-12-15


def _line_count_on_disk(path: str) -> int:
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def test_line_count_matches_file_and_survives_rollover(tmp_path, monkeypatch):
    async def go():
        store = ArchiveStore(
            archive_dir=tmp_path / "archive",
            index_path=tmp_path / "archive-index.db",
        )
        await store.init()

        import taosmd.archive as archive_mod

        # Day one: two records into the same daily file.
        monkeypatch.setattr(archive_mod.time, "time", lambda: DAY_ONE)
        await store.record("test_event", {"content": "first"})
        _, day_one_path = store._ensure_file(DAY_ONE)
        await store.record("test_event", {"content": "second"})

        assert store._line_counts[day_one_path] == 2
        assert store._line_counts[day_one_path] == _line_count_on_disk(day_one_path)

        # Day two: rollover to a new daily path, counter seeds fresh.
        monkeypatch.setattr(archive_mod.time, "time", lambda: DAY_TWO)
        await store.record("test_event", {"content": "third"})
        _, day_two_path = store._ensure_file(DAY_TWO)

        assert day_two_path != day_one_path
        assert store._line_counts[day_two_path] == 1
        assert store._line_counts[day_two_path] == _line_count_on_disk(day_two_path)

        return store

    store = asyncio.run(go())
    asyncio.run(store.close())
