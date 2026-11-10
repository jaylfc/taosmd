"""Tests for taosmd.catalog_pipeline — orchestrator."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path


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
        """detect_best_tier() with no Ollama returns tier 1 or 2 depending on NPU."""
        pipeline = CatalogPipeline(
            archive_dir=tmp_path / "archive",
            sessions_dir=tmp_path / "sessions",
            catalog_db=tmp_path / "catalog.db",
            crystals_db=tmp_path / "crystals.db",
            kg_db=tmp_path / "kg.db",
            llm_url="http://127.0.0.1:19999",  # No Ollama
        )

        async def run():
            return await pipeline.detect_best_tier()

        tier, model = _run(run())

        # On Pi with qmd/rkllama running → tier 2 (NPU detected)
        # On machines without NPU → tier 1
        assert tier in (1, 2), f"Expected tier 1 or 2, got {tier}"
        if tier == 2:
            assert model is not None, "Tier 2 should have a model name"


class TestCrystallizeAgentScoping:

    def setup_method(self, method):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def teardown_method(self, method):
        self._tmpdir.cleanup()

    def test_crystallize_stage_passes_agent_name_to_lookup_date(self, live_embed_backend):
        """index_day(agent_name='alice') must call lookup_date with agent_name='alice'
        in the crystallize stage, so bob's sessions are never touched."""
        from unittest.mock import AsyncMock, patch

        t = BASE_TS

        alice_events = [
            {
                "timestamp": t,
                "event_type": "conversation",
                "agent_name": "alice",
                "summary": "alice started coding",
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
        ]

        _write_archive(self.tmp / "archive", alice_events + bob_events)

        pipeline = _make_pipeline(self.tmp)

        lookup_calls: list[dict] = []

        async def run():
            await pipeline.init()

            # Record calls to lookup_date so we can assert on agent_name.
            original_lookup = pipeline.catalog.lookup_date

            async def recording_lookup(date, *, agent_name=None):
                lookup_calls.append({"date": date, "agent_name": agent_name})
                return await original_lookup(date, agent_name=agent_name)

            pipeline.catalog.lookup_date = recording_lookup

            # Patch detect_best_tier → tier 2, and run_if_enabled → True so
            # the crystallize block is entered (alice is not a registered agent).
            with patch.object(
                pipeline, "detect_best_tier", new=AsyncMock(return_value=(2, "test-model"))
            ), patch("taosmd.catalog_pipeline.run_if_enabled", return_value=True):
                await pipeline.index_day("2025-04-13", agent_name="alice")

            await pipeline.close()

        _run(run())

        # The crystallize stage must have called lookup_date with agent_name="alice".
        # (Other stages also call lookup_date unscoped — only the crystallize fix
        # is verified here.)
        scoped_calls = [c for c in lookup_calls if c["agent_name"] == "alice"]
        assert scoped_calls, (
            f"Expected at least one lookup_date call with agent_name='alice' "
            f"(crystallize stage); got calls: {lookup_calls}"
        )


# ---------------------------------------------------------------------------
# Provider prefix must not reach the /api/generate payload
# ---------------------------------------------------------------------------

class TestEnrichModelResolution:

    def test_index_day_sends_bare_model_to_api_generate(self, tmp_path, monkeypatch):
        """A provider-prefixed resolved model ("ollama:qwen3.5:9b") reaches
        session enrichment's /api/generate call as the bare model name.

        This is the nightly auto_setup cron path: catalog_pipeline resolves
        the memory model (which can be a provider-prefixed profile value) and
        forwards it to session_catalog.enrich_session / crystallize.
        """
        pipeline = _make_pipeline(tmp_path)
        _write_archive(tmp_path / "archive", _make_three_events())

        async def _fake_detect(self):
            return (3, "qwen3.5:9b")

        monkeypatch.setattr(CatalogPipeline, "detect_best_tier", _fake_detect)
        monkeypatch.setattr(
            "taosmd.catalog_pipeline.resolve_memory_model",
            lambda fallback=None, **kw: "ollama:qwen3.5:9b",
        )
        # Bypass the per-agent librarian task gate: this test is about the
        # model string at the HTTP boundary, not task enablement.
        monkeypatch.setattr(
            "taosmd.catalog_pipeline.run_if_enabled",
            lambda agent, task, fn, *a, fallback=None, **kw: fn(*a, **kw),
        )

        payloads: list[dict] = []
        canned = json.dumps({
            "topic": "feature work",
            "description": "worked on the feature",
            "category": "other",
        })

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"response": canned}

        class _FakeAsyncClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, **kw):
                payloads.append(json)
                return _FakeResponse()

        monkeypatch.setattr(
            "taosmd.session_catalog.httpx.AsyncClient", _FakeAsyncClient
        )

        async def run():
            await pipeline.init()
            result = await pipeline.index_day("2025-04-13", skip_crystallize=True)
            await pipeline.close()
            return result

        result = asyncio.run(run())

        assert result["enrich"]["enriched"] == 2
        assert len(payloads) == 2, "expected one /api/generate call per session"
        for payload in payloads:
            assert payload["model"] == "qwen3.5:9b", (
                f"provider prefix leaked into the /api/generate payload: "
                f"{payload['model']!r}"
            )


# ---------------------------------------------------------------------------
# Gate wiring: CatalogPipeline.index_day (intake_classification, catalog_enrichment)
# ---------------------------------------------------------------------------


class TestCatalogPipelineAgentGates:
    """Test that anonymous/empty agent_name always runs enrichment + intake classification,
    while registered agents respect their librarian config."""

    def setup_method(self, method):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def teardown_method(self, method):
        self._tmpdir.cleanup()

    def _write_single_session(self, date: str = "2025-04-13", agent_name: str = "") -> None:
        """Write a single-session archive file for the given date."""
        t = BASE_TS
        events = [
            {
                "timestamp": t,
                "event_type": "conversation",
                "agent_name": agent_name,
                "summary": "test session content",
                "data": {},
            },
        ]
        _write_archive(self.tmp / "archive", events)

    def test_anonymous_agent_always_runs_enrichment_and_classification(self, tmp_path, monkeypatch):
        """agent_name='' (anonymous/nightly cron) -> enrich_session awaited and classify status 'queued'."""
        from unittest.mock import AsyncMock, patch

        self._write_single_session(agent_name="")

        pipeline = _make_pipeline(self.tmp)

        # Force tier >= 2 and a model so the enrichment/classification blocks are entered
        with patch.object(
            pipeline, "detect_best_tier", new=AsyncMock(return_value=(2, "test-model"))
        ):
            # Stub enrich_session to verify it's called
            enrich_spy = AsyncMock()
            with patch.object(pipeline.catalog, "enrich_session", enrich_spy):
                async def run():
                    await pipeline.init()
                    result = await pipeline.index_day("2025-04-13", agent_name="")
                    await pipeline.close()
                    return result

                result = asyncio.run(run())

        # Before fix: enrich_spy.assert_not_called() and classify_results[0]["status"] == "disabled"
        # After fix: enrich_spy.assert_awaited_once() and classify_results[0]["status"] == "queued"
        enrich_spy.assert_awaited_once()
        classify_results = result["intake_classify"]["results"]
        assert classify_results, "Expected at least one classify result"
        assert classify_results[0]["status"] == "queued", (
            f"Anonymous agent should get 'queued', got {classify_results[0]['status']!r}"
        )

    def test_registered_agent_with_enrichment_disabled_skips_enrichment(self, tmp_path, monkeypatch):
        """Registered agent with catalog_enrichment=False -> enrich_session NOT awaited."""
        from unittest.mock import AsyncMock, patch

        import taosmd.agents as agents_mod

        # Create a temp registry and register an agent with catalog_enrichment disabled
        registry = agents_mod.AgentRegistry(self.tmp)
        registry.register_agent("alice")
        registry.set_librarian("alice", tasks={"catalog_enrichment": False})

        self._write_single_session(agent_name="alice")

        pipeline = _make_pipeline(self.tmp)

        # Patch is_task_enabled to use our tmp registry
        with patch("taosmd.catalog_pipeline.run_if_enabled", side_effect=agents_mod.run_if_enabled):
            with patch.object(
                pipeline, "detect_best_tier", new=AsyncMock(return_value=(2, "test-model"))
            ):
                enrich_spy = AsyncMock()
                with patch.object(pipeline.catalog, "enrich_session", enrich_spy):
                    async def run():
                        await pipeline.init()
                        result = await pipeline.index_day("2025-04-13", agent_name="alice")
                        await pipeline.close()
                        return result

                    result = asyncio.run(run())

        # enrich_session should NOT be called because catalog_enrichment is disabled for alice
        enrich_spy.assert_not_awaited()
        # classify should also be disabled for alice
        classify_results = result["intake_classify"]["results"]
        assert classify_results, "Expected at least one classify result"
        assert classify_results[0]["status"] == "disabled", (
            f"Agent with intake_classification disabled should get 'disabled', got {classify_results[0]['status']!r}"
        )

    def test_unregistered_nonempty_agent_skips_enrichment(self, tmp_path, monkeypatch):
        """Unregistered non-empty agent_name -> enrich_session NOT awaited (unchanged behavior)."""
        from unittest.mock import AsyncMock, patch

        self._write_single_session(agent_name="unknown_agent")

        pipeline = _make_pipeline(self.tmp)

        with patch.object(
            pipeline, "detect_best_tier", new=AsyncMock(return_value=(2, "test-model"))
        ):
            enrich_spy = AsyncMock()
            with patch.object(pipeline.catalog, "enrich_session", enrich_spy):
                async def run():
                    await pipeline.init()
                    result = await pipeline.index_day("2025-04-13", agent_name="unknown_agent")
                    await pipeline.close()
                    return result

                result = asyncio.run(run())

        # Unknown agent should NOT run enrichment (is_task_enabled returns False for unknown)
        enrich_spy.assert_not_awaited()
        classify_results = result["intake_classify"]["results"]
        assert classify_results, "Expected at least one classify result"
        assert classify_results[0]["status"] == "disabled", (
            f"Unknown agent should get 'disabled', got {classify_results[0]['status']!r}"
        )
