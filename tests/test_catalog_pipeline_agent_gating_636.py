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
