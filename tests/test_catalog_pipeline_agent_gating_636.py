"""Regression tests for #636: index_day gated librarian stages on the
pipeline-level agent_name, so the documented nightly cron (which passes "")
resolved every tenant's tasks as disabled regardless of their config.

The fix resolves each gate from the session's owning agent
(session.agent_name), falling back to the pipeline-level name.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import taosmd.agents as A
from taosmd.catalog_pipeline import CatalogPipeline

BASE_TS = 1744531200  # 2025-04-13 00:00:00 UTC


def _write_archive(archive_dir: Path, events: list[dict]) -> None:
    dt = datetime.fromtimestamp(events[0]["timestamp"], tz=timezone.utc)
    day_dir = archive_dir / dt.strftime("%Y") / dt.strftime("%m")
    day_dir.mkdir(parents=True, exist_ok=True)
    with open(day_dir / f"{dt.strftime('%d')}.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _make_pipeline(tmp: Path) -> CatalogPipeline:
    return CatalogPipeline(
        archive_dir=tmp / "archive",
        sessions_dir=tmp / "sessions",
        catalog_db=tmp / "catalog.db",
        crystals_db=tmp / "crystals.db",
        kg_db=tmp / "kg.db",
        llm_url="http://localhost:11434",
    )


def _hermetic_model(monkeypatch) -> None:
    """Pin model resolution to the mocked detect_best_tier fallback.

    Without this, index_day's resolve_memory_model consults the real
    generator profile for the real host tier: on CPU-only runners the
    balanced profile maps 'cpu' to the empty (retrieval-only) value, the
    fallback is ignored, and both stages skip for model reasons rather
    than gate reasons. These tests are about the #636/#637 agent gate.
    """
    import taosmd.catalog_pipeline as CP
    monkeypatch.setattr(CP, "resolve_memory_model",
                        lambda fallback=None, **kw: fallback or "")


def test_nightly_cron_shape_queues_enabled_tenant(tmp_path, monkeypatch):
    """Empty pipeline agent_name + tenant with intake ON => queued, not disabled."""
    _write_archive(tmp_path / "archive", [{
        "timestamp": BASE_TS, "event_type": "conversation",
        "agent_name": "carol", "summary": "carol fixed the images", "data": {},
    }])
    # Point the canonical registry resolver at a temp data dir.
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(tmp_path / "reg"))
    reg = A._registry()
    reg.register_agent("carol")
    reg.set_librarian("carol", enabled=True, tasks={
        "intake_classification": True,
        "catalog_enrichment": False,  # keep the test hermetic (no LLM call)
        "crystallise": False,
    })

    pipeline = _make_pipeline(tmp_path)
    _hermetic_model(monkeypatch)

    async def run():
        await pipeline.init()
        with patch.object(pipeline, "detect_best_tier",
                          new=AsyncMock(return_value=(2, "test-model"))):
            return await pipeline.index_day("2025-04-13")
    result = asyncio.run(run())
    asyncio.run(pipeline.close())

    statuses = [r["status"] for r in result["intake_classify"]["results"]]
    assert statuses and "disabled" not in statuses, (
        f"#636 regression: nightly-cron shape disabled an enabled tenant: {statuses}")


def test_crystallize_skips_only_disabled_tenants(tmp_path, monkeypatch):
    """A tenant with crystallise OFF is skipped; the reason names the gate,
    not skip_crystallize=True (#636 + #637)."""
    _write_archive(tmp_path / "archive", [{
        "timestamp": BASE_TS, "event_type": "conversation",
        "agent_name": "carol", "summary": "carol fixed the images", "data": {},
    }])
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(tmp_path / "reg"))
    reg = A._registry()
    reg.register_agent("carol")
    reg.set_librarian("carol", enabled=True, tasks={
        "intake_classification": False,
        "catalog_enrichment": False,
        "crystallise": False,
    })

    pipeline = _make_pipeline(tmp_path)
    _hermetic_model(monkeypatch)

    async def run():
        await pipeline.init()
        with patch.object(pipeline, "detect_best_tier",
                          new=AsyncMock(return_value=(2, "test-model"))):
            return await pipeline.index_day("2025-04-13")
    result = asyncio.run(run())
    asyncio.run(pipeline.close())

    cry = result["crystallize"]
    assert cry.get("skipped") is True
    assert "gate" in cry["reason"], (
        f"#637 regression: reason should name the gate, got {cry['reason']!r}")


def test_enrichment_gated_per_owner(tmp_path, monkeypatch):
    """Two sessions same day, owners carol (catalog_enrichment ON) and dave
    (catalog_enrichment OFF). Patch enrich_session and assert it was called
    for carol only with agent_name='carol'."""
    _write_archive(tmp_path / "archive", [
        {
            "timestamp": BASE_TS,
            "event_type": "conversation",
            "agent_name": "carol",
            "summary": "carol fixed the images",
            "data": {},
        },
        {
            "timestamp": BASE_TS + 2000,  # >30 min gap (1800s) to force separate session
            "event_type": "conversation",
            "agent_name": "dave",
            "summary": "dave reviewed the PR",
            "data": {},
        },
    ])
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(tmp_path / "reg"))
    reg = A._registry()
    reg.register_agent("carol")
    reg.set_librarian("carol", enabled=True, tasks={
        "intake_classification": False,
        "catalog_enrichment": True,
        "crystallise": False,
    })
    reg.register_agent("dave")
    reg.set_librarian("dave", enabled=True, tasks={
        "intake_classification": False,
        "catalog_enrichment": False,
        "crystallise": False,
    })

    pipeline = _make_pipeline(tmp_path)
    _hermetic_model(monkeypatch)

    enrich_recorder = AsyncMock()
    async def run():
        await pipeline.init()
        with patch.object(pipeline, "detect_best_tier",
                          new=AsyncMock(return_value=(2, "test-model"))):
            with patch.object(pipeline.catalog, "enrich_session",
                              new=enrich_recorder):
                return await pipeline.index_day("2025-04-13")
    asyncio.run(run())

    # Assert split produced 2 sessions (before closing)
    sessions = asyncio.run(pipeline.catalog.lookup_date("2025-04-13"))
    assert len(sessions) == 2, f"expected 2 sessions, got {len(sessions)}"

    # Assert enrich_session called for carol only with agent_name='carol'
    assert enrich_recorder.call_count == 1, (
        f"expected 1 enrich call, got {enrich_recorder.call_count}")
    call_kwargs = enrich_recorder.call_args.kwargs
    assert call_kwargs.get("agent_name") == "carol", (
        f"expected agent_name='carol', got {call_kwargs.get('agent_name')!r}")
    # Verify it was carol's session
    carol_sessions = [s for s in sessions if s.get("agent_name") == "carol"]
    assert len(carol_sessions) == 1
    assert call_kwargs.get("session_id") == carol_sessions[0]["id"]

    asyncio.run(pipeline.close())


def test_crystallize_two_tenants_only_enabled_one_runs(tmp_path, monkeypatch):
    """Two sessions same day, owners carol (crystallise ON) and dave
    (crystallise OFF). Patch CrystalStore.crystallize and assert it ran for
    carol only, not dave, and result['crystallize'] is not skipped."""
    _write_archive(tmp_path / "archive", [
        {
            "timestamp": BASE_TS,
            "event_type": "conversation",
            "agent_name": "carol",
            "summary": "carol fixed the images",
            "data": {},
        },
        {
            "timestamp": BASE_TS + 2000,  # >30 min gap (1800s) to force separate session
            "event_type": "conversation",
            "agent_name": "dave",
            "summary": "dave reviewed the PR",
            "data": {},
        },
    ])
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(tmp_path / "reg"))
    reg = A._registry()
    reg.register_agent("carol")
    reg.set_librarian("carol", enabled=True, tasks={
        "intake_classification": False,
        "catalog_enrichment": False,
        "crystallise": True,
    })
    reg.register_agent("dave")
    reg.set_librarian("dave", enabled=True, tasks={
        "intake_classification": False,
        "catalog_enrichment": False,
        "crystallise": False,
    })

    pipeline = _make_pipeline(tmp_path)
    _hermetic_model(monkeypatch)

    crystallize_recorder = AsyncMock()
    async def run():
        await pipeline.init()
        with patch.object(pipeline, "detect_best_tier",
                          new=AsyncMock(return_value=(2, "test-model"))):
            # Patch CrystalStore.crystallize at the class level since a new
            # CrystalStore is instantiated inside index_day
            with patch("taosmd.crystallize.CrystalStore.crystallize",
                       new=crystallize_recorder):
                return await pipeline.index_day("2025-04-13")
    result = asyncio.run(run())

    # Assert split produced 2 sessions (before closing)
    sessions = asyncio.run(pipeline.catalog.lookup_date("2025-04-13"))
    assert len(sessions) == 2, f"expected 2 sessions, got {len(sessions)}"

    # Assert crystallize called for carol only
    assert crystallize_recorder.call_count == 1, (
        f"expected 1 crystallize call, got {crystallize_recorder.call_count}")
    call_kwargs = crystallize_recorder.call_args.kwargs
    assert call_kwargs.get("session_id") is not None
    carol_sessions = [s for s in sessions if s.get("agent_name") == "carol"]
    assert len(carol_sessions) == 1
    assert str(call_kwargs.get("session_id")) == str(carol_sessions[0]["id"])

    # Assert result['crystallize'] is not skipped
    cry = result["crystallize"]
    assert cry.get("skipped") is not True, (
        f"crystallize should not be skipped, got {cry}")
    assert cry.get("crystallized") == 1, (
        f"expected crystallized=1, got {cry.get('crystallized')}")

    asyncio.run(pipeline.close())
