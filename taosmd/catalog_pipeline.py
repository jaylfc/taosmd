"""Catalog Pipeline Orchestrator (taOSmd).

Orchestrates the three-stage session processing pipeline:
  1. Split  — split day's archive JSONL into per-session files
  2. Enrich — LLM-based topic/description/category enrichment
  3. Crystallize — compact digest + KG lesson extraction (tier >= 2 only)

The pipeline runs synchronously — each stage completes before the next begins.
For resource-constrained scheduling, taOS wraps this with its own job queue.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from .agents import run_if_enabled
from .prompts import intake_classification_prompt
from .session_catalog import SessionCatalog
from .crystallize import CrystalStore
from .knowledge_graph import TemporalKnowledgeGraph

logger = logging.getLogger(__name__)


class CatalogPipeline:
    """Three-stage pipeline: split → enrich → crystallize.

    When job_queue and resource_manager are provided, heavy tasks
    (enrichment, crystallization) are dispatched through the queue
    """

    def __init__(
        self,
        archive_dir: str | Path,
        sessions_dir: str | Path,
        catalog_db: str | Path,
        crystals_db: str | Path,
        kg_db: str | Path,
        llm_url: str = "http://localhost:11434",
    ):
        self._archive_dir = Path(archive_dir)
        self._sessions_dir = Path(sessions_dir)
        self._catalog_db = Path(catalog_db)
        self._crystals_db = Path(crystals_db)
        self._kg_db = Path(kg_db)
        self._llm_url = llm_url

        self.catalog = SessionCatalog(
            db_path=self._catalog_db,
            archive_dir=self._archive_dir,
            sessions_dir=self._sessions_dir,
        )

    async def init(self) -> None:
        """Initialise the SessionCatalog."""
        await self.catalog.init()

    async def close(self) -> None:
        """Close the SessionCatalog."""
        await self.catalog.close()

    # ------------------------------------------------------------------
    # Tier detection
    # ------------------------------------------------------------------

    async def detect_best_tier(self) -> tuple[int, str | None]:
        """Detect the best available processing tier and model.

        Checks Ollama first (GPU/CPU), then rkllama/qmd (NPU).

        Tier 3: qwen3.5:9b+ (GPU worker or local GPU)
        Tier 2: qwen3:4b / qwen3.5:4b (NPU or CPU)
        Tier 1: no LLM available (heuristic only)
        """
        # Check Ollama first (standard endpoint)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._llm_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]

                for name in models:
                    lower = name.lower()
                    if "qwen3.5:9b" in lower or "qwen3.5:27b" in lower:
                        return (3, name)

                for name in models:
                    lower = name.lower()
                    if lower.startswith("qwen3") or lower.startswith("qwen3.5"):
                        return (2, name)
        except Exception as exc:
            logger.debug("Ollama not reachable at %s: %s", self._llm_url, exc)

        # Check rkllama/qmd (NPU on RK3588) — typically on port 7832
        for npu_url in ["http://localhost:7832", "http://localhost:8080"]:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{npu_url}/health")
                    if resp.status_code == 200:
                        # rkllama is running — it serves qwen3:4b on NPU
                        logger.info("NPU backend detected at %s", npu_url)
                        return (2, "qwen3:4b")
            except Exception:
                continue

        # Check if Ollama OpenAI-compat endpoint works (some setups)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._llm_url}/v1/models")
                if resp.status_code == 200:
                    models = [m["id"] for m in resp.json().get("data", [])]
                    for name in models:
                        lower = name.lower()
                        if lower.startswith("qwen3") or lower.startswith("qwen3.5"):
                            return (2, name)
        except Exception:
            pass

        return (1, None)

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------

    async def index_day(
        self,
        date: str,
        force: bool = False,
        skip_crystallize: bool = False,
        agent_name: str = "",
    ) -> dict[str, Any]:
        """Run all 3 stages for one day.

        Args:
            date: YYYY-MM-DD
            force: force re-split even if entries already exist
            skip_crystallize: skip stage 3 even when tier >= 2

        Returns:
            dict with keys date, split, enrich, crystallize, total_time
        """
        t0 = time.time()
        result: dict[str, Any] = {
            "date": date,
            "split": {},
            "intake_classify": {},
            "enrich": {},
            "crystallize": {},
            "total_time": 0.0,
        }

        # --- Stage 1: Split ---
        split_result = self.catalog.split_day(date, force=force)
        result["split"] = split_result
        sessions_created = split_result.get("sessions_created", 0)

        tier, model = await self.detect_best_tier()

        # --- Stage 1b: Intake Classification (taxonomy filing) ---
        classify_results = []
        classify_errors = []
        if tier >= 2 and model:
            sessions_to_classify = await self.catalog.lookup_date(date)
            for session in sessions_to_classify:
                sid = session["id"]
                enabled = run_if_enabled(
                    agent_name, "intake_classification",
                    lambda: True, fallback=False
                )
                if enabled:
                    classify_results.append({"session_id": sid, "status": "queued"})
                else:
                    classify_results.append({"session_id": sid, "status": "disabled"})
        result["intake_classify"] = {
            "count": len(classify_results),
            "results": classify_results,
            "errors": classify_errors,
        }

        # --- Stage 2: Enrich ---

        sessions = await self.catalog.lookup_date(date)
        enriched_count = 0
        enrich_errors = []

        for session in sessions:
            sid = session["id"]
            if tier >= 2 and model:
                _enrich_enabled = run_if_enabled(
                    agent_name, "catalog_enrichment",
                    lambda: True, fallback=False
                )
                if _enrich_enabled:
                    try:
                        await self.catalog.enrich_session(
                            session_id=sid,
                            llm_url=self._llm_url,
                            model=model,
                            tier=tier,
                            agent_name=agent_name,
                        )
                        enriched_count += 1
                    except Exception as exc:
                        logger.warning("Enrich failed for session %s: %s", sid, exc)
                        enrich_errors.append({"session_id": sid, "error": str(exc)})
                else:
                    enriched_count += 1  # heuristic fallback, counted as done
            else:
                enriched_count += 1

        result["enrich"] = {
            "enriched": enriched_count,
            "tier": tier,
            "model": model,
            "errors": enrich_errors,
        }

        # --- Stage 3: Crystallize ---
        _crystallize_enabled = run_if_enabled(
            agent_name, "crystallise",
            lambda: True, fallback=False
        )
        if tier >= 2 and not skip_crystallize and model and _crystallize_enabled:
            crystallized_count = 0
            crystal_errors = []

            cs = CrystalStore(db_path=self._crystals_db)
            await cs.init()
            kg = TemporalKnowledgeGraph(db_path=self._kg_db)
            await kg.init()

            sessions_fresh = await self.catalog.lookup_date(date, agent_name=agent_name or None)
            for session in sessions_fresh:
                sid = session["id"]
                try:
                    ctx = await self.catalog.get_session_context(sid, agent_name=agent_name or None)
                    if ctx is None:
                        continue

                    content_lines = ctx.get("archive_lines") or []
                    turns = []
                    for line in content_lines:
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        summary = event.get("summary") or (
                            event.get("data", {}) or {}
                        ).get("content", "")
                        turns.append({
                            "role": "user",
                            "content": summary,
                            "timestamp": event.get("timestamp", time.time()),
                        })

                    if turns:
                        await cs.crystallize(
                            session_id=str(sid),
                            turns=turns,
                            llm_url=self._llm_url,
                            model=model,
                            kg=kg,
                        )
                        crystallized_count += 1

                except Exception as exc:
                    logger.warning("Crystallize failed for session %s: %s", sid, exc)
                    crystal_errors.append({"session_id": sid, "error": str(exc)})

            await cs.close()
            await kg.close()

            result["crystallize"] = {
                "crystallized": crystallized_count,
                "errors": crystal_errors,
            }
        else:
            result["crystallize"] = {
                "crystallized": 0,
                "skipped": True,
                "reason": "tier < 2" if tier < 2 else "skip_crystallize=True",
            }

        result["total_time"] = round(time.time() - t0, 3)
        return result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def index_yesterday(self) -> dict[str, Any]:
        """Convenience for cron: index yesterday's date."""
        yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        return await self.index_day(yesterday)

    async def index_range(
        self,
        start_date: str,
        end_date: str,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Index all dates in an inclusive range.

        Args:
            start_date: YYYY-MM-DD
            end_date:   YYYY-MM-DD
            force:      force re-split for each day

        Returns:
            List of per-day result dicts.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        results = []
        current = start
        while current <= end:
            day_str = current.strftime("%Y-%m-%d")
            result = await self.index_day(day_str, force=force)
            results.append(result)
            current += timedelta(days=1)

        return results

    async def rebuild(self) -> list[dict[str, Any]]:
        """Find all archive files by globbing and index each date with force=True.

        Returns:
            List of per-day result dicts.
        """
        archive_files = sorted(self._archive_dir.glob("**/*.jsonl"))
        archive_files += sorted(self._archive_dir.glob("**/*.jsonl.gz"))

        dates_seen: set[str] = set()
        results = []

        for path in archive_files:
            # Expect structure: archive_dir/YYYY/MM/DD.jsonl[.gz]
            parts = path.parts
            try:
                # Walk up from filename to get year/month/day
                day_part = path.stem  # "DD" (strip .jsonl or .jsonl from .gz)
                if day_part.endswith(".jsonl"):
                    day_part = day_part[:-6]
                month_part = path.parent.name
                year_part = path.parent.parent.name
                date_str = f"{year_part}-{month_part}-{day_part}"
                # Validate
                datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, IndexError):
                logger.warning("Skipping unrecognised archive path: %s", path)
                continue

            if date_str in dates_seen:
                continue
            dates_seen.add(date_str)

            result = await self.index_day(date_str, force=True)
            results.append(result)

        return results
