"""Tests for taosmd.catalog_pipeline — orchestrator."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from taosmd.catalog_pipeline import CatalogPipeline


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
    """Write events to archive/YYYY/MM/DD.jsonl and return the file path."""
    first_ts = events[0]["timestamp"]
    dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)
    day_dir = archive_dir / dt.strftime("%Y") / dt.strftime("%m")
    day_dir.mkdir(parents=True, exist_ok=True)
    archive_file = day_dir / f"{dt.strftime('%d')}.jsonl"
    with open(archive_file, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return archive_file


def _make_pipeline(tmp_path: Path) -> CatalogPipeline:
    return CatalogPipeline(
        archive_dir=tmp_path / "archive",
        sessions_dir=tmp_path / "sessions",
        catalog_db=tmp_path / "catalog.db",
        crystals_db=tmp_path / "crystals.db",
        kg_db=tmp_path / "kg.db",
        llm_url="http://localhost:11434",
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixture: 3 events — 2 + gap + 1 = 2 sessions
#
#   session 1:  t+0, t+5min      (2 events)
#   [45-min gap]
#   session 2:  t+50min          (1 event)
# ---------------------------------------------------------------------------

def _make_three_events():
    t = BASE_TS
    return [
        _make_event(t + 0,       "started working on the feature"),
        _make_event(t + 5 * 60,  "made progress on the feature"),
        _make_event(t + 50 * 60, "reviewed the pull request"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineIndexDay:

    def setup_method(self, method):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def teardown_method(self, method):
        self._tmpdir.cleanup()

    def test_pipeline_index_day(self):
        """index_day() should split 3 events (with 45-min gap) into 2 sessions
        and report enriched == 2 (tier 1 heuristic path)."""
        pipeline = _make_pipeline(self.tmp)

        # Write archive: 3 events, gap after 2nd → 2 sessions
        _write_archive(self.tmp / "archive", _make_three_events())

        async def run():
            await pipeline.init()
            result = await pipeline.index_day("2025-04-13", skip_crystallize=True)
            await pipeline.close()
            return result

        result = _run(run())

        assert result["date"] == "2025-04-13"
        assert result["split"]["sessions_created"] == 2, (
            f"Expected 2 sessions, got {result['split']['sessions_created']}"
        )
        assert result["enrich"]["enriched"] == 2, (
            f"Expected 2 enriched sessions, got {result['enrich']['enriched']}"
        )
        assert "total_time" in result


class TestPipelineDetectBestTier:

    def test_pipeline_detect_best_tier_no_ollama(self, tmp_path):
        """detect_best_tier() with no Ollama running should return tier=1, model=None."""
        pipeline = CatalogPipeline(
            archive_dir=tmp_path / "archive",
            sessions_dir=tmp_path / "sessions",
            catalog_db=tmp_path / "catalog.db",
            crystals_db=tmp_path / "crystals.db",
            kg_db=tmp_path / "kg.db",
            # Use an address that will refuse connections immediately
            llm_url="http://127.0.0.1:19999",
        )

        async def run():
            return await pipeline.detect_best_tier()

        tier, model = _run(run())

        assert tier == 1, f"Expected tier=1 when Ollama is absent, got {tier}"
        assert model is None, f"Expected model=None when Ollama is absent, got {model!r}"
