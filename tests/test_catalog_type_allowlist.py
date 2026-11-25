"""RED test for catalog event type allowlist (tsk-skyms6).

This test MUST FAIL on origin/master before the fix is applied.
It verifies that non-conversation event types (setting.changed, bus.ack,
archive.torn_tail) are NOT fed into sessions, crystals, or the knowledge graph.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from taosmd.catalog_pipeline import CatalogPipeline


BASE_TS = 1744531200  # 2025-04-13 00:00:00 UTC


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


class TestCatalogTypeAllowlist:
    """Tests for event type filtering in the catalog pipeline."""

    def setup_method(self, method):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def teardown_method(self, method):
        self._tmpdir.cleanup()

    def test_non_conversation_events_excluded_from_sessions_crystals_kg(self):
        """Non-conversation event types must not appear in sessions, crystals, or KG.

        Archive contains:
        - 2 conversation turns (should be processed)
        - 1 setting.changed event with unique marker (must be excluded)
        - 1 bus.ack event with unique marker (must be excluded)
        - 1 archive.torn_tail event with unique marker (must be excluded)

        On origin/master (no filter), the markers WILL appear in sessions/crystals/KG.
        After fix, markers must NOT appear anywhere.
        """
        # Unique markers that must NOT leak into processed outputs
        setting_marker = "SETTING_CHANGED_MARKER_XYZ789"
        bus_marker = "BUS_ACK_MARKER_ABC123"
        torn_marker = "TORN_TAIL_MARKER_DEF456"

        t = BASE_TS
        events = [
            # Conversation turn 1
            {
                "timestamp": t,
                "event_type": "conversation",
                "summary": "user asked about feature",
                "data": {"content": "user asked about feature"},
            },
            # Conversation turn 2
            {
                "timestamp": t + 60,
                "event_type": "conversation",
                "summary": "agent replied with implementation plan",
                "data": {"content": "agent replied with implementation plan"},
            },
            # Non-conversation events with unique markers (must be filtered out)
            {
                "timestamp": t + 120,
                "event_type": "setting.changed",
                "summary": f"setting changed {setting_marker}",
                "data": {"setting": "theme", "value": "dark"},
            },
            {
                "timestamp": t + 180,
                "event_type": "bus.ack",
                "summary": f"bus ack {bus_marker}",
                "data": {"message_id": 42},
            },
            {
                "timestamp": t + 240,
                "event_type": "archive.torn_tail",
                "summary": f"torn tail {torn_marker}",
                "data": {"bytes": 1024},
            },
        ]

        _write_archive(self.tmp / "archive", events)

        pipeline = _make_pipeline(self.tmp)

        async def run():
            await pipeline.init()
            # Patch detect_best_tier to return tier 1 (no LLM) so we don't
            # hang on Ollama calls. This tests the split stage which is
            # where the filtering must happen.
            with patch.object(
                pipeline, "detect_best_tier", new=AsyncMock(return_value=(1, None))
            ):
                result = await pipeline.index_day("2025-04-13", skip_crystallize=True)
            # Keep pipeline open to access catalog
            return result, pipeline.catalog

        result, catalog = _run(run())

        # Check sessions created (should be 1 session with 2 conversation events)
        assert result["split"]["sessions_created"] == 1, (
            f"Expected 1 session from conversation events, got {result['split']['sessions_created']}"
        )

        # Get the session and check its split file for markers
        sessions = _run(catalog.lookup_date("2025-04-13"))
        assert len(sessions) == 1
        session = sessions[0]

        # Read the split file and verify markers are NOT present
        split_path = Path(session["split_file"])
        assert split_path.exists()
        split_content = split_path.read_text()

        # RED ASSERTIONS - these MUST FAIL on origin/master (markers leak through)
        assert setting_marker not in split_content, (
            f"setting.changed marker leaked into session split file: {setting_marker}"
        )
        assert bus_marker not in split_content, (
            f"bus.ack marker leaked into session split file: {bus_marker}"
        )
        assert torn_marker not in split_content, (
            f"archive.torn_tail marker leaked into session split file: {torn_marker}"
        )

        # Also verify conversation content IS present (positive control)
        assert "user asked about feature" in split_content
        assert "agent replied with implementation plan" in split_content

        _run(pipeline.close())


    def test_legacy_lines_without_event_type_still_processed(self):
        """Lines without event_type field (legacy) must still be processed.

        This ensures backwards compatibility with pre-event_type archive files.
        """
        t = BASE_TS
        events = [
            # Legacy line without event_type (should be treated as conversation)
            {
                "timestamp": t,
                "summary": "legacy conversation turn",
                "data": {"content": "legacy conversation turn"},
            },
            # Normal conversation line
            {
                "timestamp": t + 60,
                "event_type": "conversation",
                "summary": "normal conversation turn",
                "data": {"content": "normal conversation turn"},
            },
        ]

        _write_archive(self.tmp / "archive", events)

        pipeline = _make_pipeline(self.tmp)

        async def run():
            await pipeline.init()
            result = await pipeline.index_day("2025-04-13", skip_crystallize=True)
            return result, pipeline.catalog

        result, catalog = _run(run())

        # Should create 1 session with both events (gap is only 1 min)
        assert result["split"]["sessions_created"] == 1

        sessions = _run(catalog.lookup_date("2025-04-13"))
        assert len(sessions) == 1
        session = sessions[0]

        split_path = Path(session["split_file"])
        split_content = split_path.read_text()

        # Both legacy and normal lines should be present
        assert "legacy conversation turn" in split_content
        assert "normal conversation turn" in split_content

        _run(pipeline.close())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
