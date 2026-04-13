"""Tests for taosmd.session_catalog — splitter and catalog core."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from taosmd.session_catalog import SessionCatalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = 1744531200  # 2025-04-13 00:00:00 UTC


def _make_event(ts: float, summary: str = "test event") -> dict:
    return {
        "timestamp": ts,
        "event_type": "conversation",
        "summary": summary,
        "data": {},
    }


def _write_archive(archive_dir: Path, events: list[dict]) -> Path:
    """Write events to data/archive/YYYY/MM/DD.jsonl and return the file path."""
    first_ts = events[0]["timestamp"]
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)
    day_dir = archive_dir / dt.strftime("%Y") / dt.strftime("%m")
    day_dir.mkdir(parents=True, exist_ok=True)
    archive_file = day_dir / f"{dt.strftime('%d')}.jsonl"
    with open(archive_file, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return archive_file


def _make_catalog(tmp_path: Path) -> SessionCatalog:
    archive_dir = tmp_path / "archive"
    sessions_dir = tmp_path / "sessions"
    db_path = tmp_path / "catalog.db"
    return SessionCatalog(
        db_path=db_path,
        archive_dir=archive_dir,
        sessions_dir=sessions_dir,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixture: 5 events with a 45-min gap after the third
# ---------------------------------------------------------------------------
#
#  session 1:  t+0, t+5min, t+10min   (3 events)
#  [gap of 45 min]
#  session 2:  t+55min, t+60min       (2 events)

def _make_five_events():
    t = BASE_TS
    return [
        _make_event(t + 0,          "started coding the feature"),
        _make_event(t + 5 * 60,     "made progress on feature"),
        _make_event(t + 10 * 60,    "finished feature draft"),
        _make_event(t + 55 * 60,    "reviewing pull request"),
        _make_event(t + 60 * 60,    "approved and merged"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitter:

    def setup_method(self, method):
        """Each test creates its own tmp_path manually to stay independent."""
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def teardown_method(self, method):
        self._tmpdir.cleanup()

    def _catalog_with_events(self):
        cat = _make_catalog(self.tmp)
        _run(cat.init())
        events = _make_five_events()
        _write_archive(self.tmp / "archive", events)
        return cat

    def test_splitter_detects_two_sessions(self):
        """5 events with a 45-min gap should produce exactly 2 sessions."""
        cat = self._catalog_with_events()
        result = cat.split_day("2025-04-13")
        _run(cat.close())

        assert result["sessions_created"] == 2, (
            f"Expected 2 sessions, got {result['sessions_created']}"
        )

    def test_splitter_creates_split_files(self):
        """session-001 should have 3 events, session-002 should have 2 events."""
        cat = self._catalog_with_events()
        result = cat.split_day("2025-04-13")
        _run(cat.close())

        split_files = result["split_files"]
        assert len(split_files) == 2

        def count_lines(path):
            return sum(1 for line in open(path) if line.strip())

        assert count_lines(split_files[0]) == 3, (
            f"session-001 expected 3 events, got {count_lines(split_files[0])}"
        )
        assert count_lines(split_files[1]) == 2, (
            f"session-002 expected 2 events, got {count_lines(split_files[1])}"
        )

    def test_splitter_creates_catalog_entries(self):
        """Catalog entries should have correct line_start/end, turn_count, tier=1,
        and the topic should come from the first event summary."""
        cat = self._catalog_with_events()
        cat.split_day("2025-04-13")
        sessions = _run(cat.lookup_date("2025-04-13"))
        _run(cat.close())

        assert len(sessions) == 2

        s1 = sessions[0]
        assert s1["line_start"] == 1
        assert s1["line_end"] == 3
        assert s1["turn_count"] == 3
        assert s1["tier"] == 1
        assert "started coding the feature" in s1["topic"]

        s2 = sessions[1]
        assert s2["line_start"] == 4
        assert s2["line_end"] == 5
        assert s2["turn_count"] == 2
        assert s2["tier"] == 1
        assert "reviewing pull request" in s2["topic"]

    def test_splitter_is_idempotent(self):
        """Running split_day twice should produce 2 sessions not 4."""
        cat = self._catalog_with_events()
        cat.split_day("2025-04-13", force=True)
        cat.split_day("2025-04-13", force=True)
        sessions = _run(cat.lookup_date("2025-04-13"))
        _run(cat.close())

        assert len(sessions) == 2, (
            f"Expected 2 sessions after two runs, got {len(sessions)}"
        )
