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


class TestEnricher:

    def setup_method(self, method):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def teardown_method(self, method):
        self._tmpdir.cleanup()

    def _catalog_with_split(self):
        """Return an initialised catalog that has already been split for 2025-04-13."""
        cat = _make_catalog(self.tmp)
        _run(cat.init())
        events = _make_five_events()
        _write_archive(self.tmp / "archive", events)
        cat.split_day("2025-04-13")
        return cat

    def test_enrich_session_heuristic_fallback(self):
        """enrich_session() with an unreachable LLM should keep tier=1 and the
        original heuristic topic derived from the first event summary."""
        cat = self._catalog_with_split()
        sessions = _run(cat.lookup_date("2025-04-13"))
        session_id = sessions[0]["id"]
        original_topic = sessions[0]["topic"]

        # Use an invalid port so httpx fails immediately
        result = _run(
            cat.enrich_session(
                session_id,
                llm_url="http://localhost:99999",
                model="test-model",
                tier=2,
            )
        )
        _run(cat.close())

        assert result is not None
        assert result["tier"] == 1, (
            f"Expected tier=1 after fallback, got {result['tier']}"
        )
        assert result["topic"] == original_topic, (
            f"Expected original topic to be preserved, got {result['topic']!r}"
        )

    def test_enrich_updates_fts(self):
        """_update_enrichment() should update the FTS index so that
        search_topic() can find the new topic text."""
        cat = self._catalog_with_split()
        sessions = _run(cat.lookup_date("2025-04-13"))
        session_id = sessions[0]["id"]

        _run(
            cat._update_enrichment(
                session_id,
                topic="ONNX embedding fixes",
                description="Fixed ONNX embedding pipeline for batch inference.",
                category="debugging",
                tier=2,
            )
        )

        results = _run(cat.search_topic("ONNX embedding"))
        _run(cat.close())

        assert len(results) >= 1, "Expected at least one FTS hit for 'ONNX embedding'"
        ids = [r["id"] for r in results]
        assert session_id in ids, (
            f"session_id {session_id} not found in FTS results: {ids}"
        )

    def test_enrich_session_respects_agent_name(self):
        """enrich_session with agent_name='alice' must not touch bob's session.

        Two sessions are created: one whose split file contains only alice's
        events, one whose split file contains only bob's events.  Enrichment
        called with agent_name='alice' should leave bob's session at tier=1
        (LLM unreachable, so even alice's tier stays at 1 — but the key
        assertion is that bob's context comes back empty, meaning the enrichment
        path received no content for that session and left the tier unchanged).
        """
        import tempfile
        tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(tmpdir.name)

        cat = _make_catalog(tmp)
        _run(cat.init())

        # Write a single archive file with events tagged for two agents.
        archive_dir = tmp / "archive"
        t = BASE_TS

        alice_events = [
            {
                "timestamp": t,
                "event_type": "conversation",
                "agent_name": "alice",
                "summary": "alice started coding",
                "data": {},
            },
            {
                "timestamp": t + 5 * 60,
                "event_type": "conversation",
                "agent_name": "alice",
                "summary": "alice finished coding",
                "data": {},
            },
        ]
        bob_events = [
            {
                "timestamp": t + 55 * 60,
                "event_type": "conversation",
                "agent_name": "bob",
                "summary": "bob reviewing PR",
                "data": {},
            },
            {
                "timestamp": t + 60 * 60,
                "event_type": "conversation",
                "agent_name": "bob",
                "summary": "bob merged PR",
                "data": {},
            },
        ]

        _write_archive(archive_dir, alice_events + bob_events)
        cat.split_day("2025-04-13")

        sessions = _run(cat.lookup_date("2025-04-13"))
        assert len(sessions) == 2, f"Expected 2 sessions, got {len(sessions)}"

        # Both sessions start at tier=1.
        assert sessions[0]["tier"] == 1
        assert sessions[1]["tier"] == 1

        alice_session_id = sessions[0]["id"]
        bob_session_id = sessions[1]["id"]

        # Call enrich with agent_name="alice". LLM is unreachable so tier stays
        # at 1 for alice's session, but the important invariant is that bob's
        # get_session_context returns no matching lines — the agent filter works.
        _run(
            cat.enrich_session(
                alice_session_id,
                llm_url="http://localhost:99999",
                model="test-model",
                tier=2,
                agent_name="alice",
            )
        )

        # Fetch context for bob's session scoped to alice — should be empty.
        bob_ctx = _run(
            cat.get_session_context(bob_session_id, agent_name="alice")
        )
        _run(cat.close())
        tmpdir.cleanup()

        assert bob_ctx is not None, "get_session_context returned None for bob's session"
        assert bob_ctx["archive_lines"] == [], (
            f"Expected no alice lines in bob's session, got: {bob_ctx['archive_lines']}"
        )

        # Bob's tier must remain 1 — enrichment was never called for it.
        assert bob_ctx["tier"] == 1, (
            f"Expected bob's tier=1, got {bob_ctx['tier']}"
        )

    def test_llm_enrich_json_path(self):
        """_llm_enrich parses a clean JSON response from the module prompt."""
        from unittest.mock import AsyncMock, MagicMock, patch

        cat = _make_catalog(self.tmp)
        _run(cat.init())

        json_response = '{"topic": "ONNX batch inference", "description": "Debugged batch inference pipeline.", "category": "debugging"}'

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": json_response}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("taosmd.session_catalog.httpx.AsyncClient", return_value=mock_client):
            topic, description, category = _run(
                cat._llm_enrich("some session content", "http://localhost:11434", "test-model")
            )

        _run(cat.close())

        assert topic == "ONNX batch inference"
        assert description == "Debugged batch inference pipeline."
        assert category == "debugging"

    def test_llm_enrich_json_fallback_to_legacy_parser(self):
        """_llm_enrich falls back to _parse_enrichment when JSON parse fails."""
        from unittest.mock import AsyncMock, MagicMock, patch

        cat = _make_catalog(self.tmp)
        _run(cat.init())

        legacy_response = (
            "TOPIC: legacy topic phrase\n"
            "DESCRIPTION: legacy description sentence.\n"
            "CATEGORY: coding\n"
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": legacy_response}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("taosmd.session_catalog.httpx.AsyncClient", return_value=mock_client):
            topic, description, category = _run(
                cat._llm_enrich("some session content", "http://localhost:11434", "test-model")
            )

        _run(cat.close())

        assert topic == "legacy topic phrase"
        assert description == "legacy description sentence."
        assert category == "coding"
