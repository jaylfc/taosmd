"""Tests for the in-memory archive line-count optimisation (issue #95) and
day-boundary file rollover.

record() used to re-read the entire daily JSONL on every write to compute
the line_number. It now keeps an in-memory counter (self._line_counts) seeded
from disk once per path. These tests assert the counter stays consistent with
the file and that the open file handle rolls over at DAY boundaries: files
are one-per-day (YYYY/MM/DD.jsonl), so a handle kept open across midnight
UTC would keep appending to yesterday's file while the index points at
today's, corrupting file_path/line_number provenance.
"""

from __future__ import annotations

import asyncio

from taosmd.archive import ArchiveStore


# Two timestamps on consecutive UTC days in the SAME month. The archive must
# roll its open file at the day boundary, not just when the year/month parent
# directory changes.
DAY_ONE = 1_700_000_000.0            # 2023-11-14 UTC
DAY_TWO = DAY_ONE + 86_400.0         # 2023-11-15 UTC (same month)


def _line_count_on_disk(path: str) -> int:
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def test_line_count_matches_file_and_survives_day_rollover(tmp_path, monkeypatch):
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

        assert day_one_path.endswith("14.jsonl")
        assert store._line_counts[day_one_path] == 2
        assert store._line_counts[day_one_path] == _line_count_on_disk(day_one_path)

        # Midnight UTC passes (same month): rollover to a new daily path.
        monkeypatch.setattr(archive_mod.time, "time", lambda: DAY_TWO)
        row_id = await store.record("test_event", {"content": "third"})
        _, day_two_path = store._ensure_file(DAY_TWO)

        assert day_two_path != day_one_path
        assert day_two_path.endswith("15.jsonl")

        # The new day's FILE received the new event; yesterday's file did not.
        assert _line_count_on_disk(day_two_path) == 1
        assert _line_count_on_disk(day_one_path) == 2
        assert store._line_counts[day_two_path] == 1

        # The index row points at the right file and line.
        idx = store._conn.execute(
            "SELECT file_path, line_number FROM archive_index WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert idx["file_path"] == day_two_path
        assert idx["line_number"] == 1

        # Both days pass checksum verification (no torn/misplaced writes).
        for date in ("2023-11-14", "2023-11-15"):
            report = await store.verify_day(date)
            assert report["bad"] == 0
            assert report["ok"] == report["total"] > 0

        return store

    store = asyncio.run(go())
    asyncio.run(store.close())
