"""Tests for the job worker."""

import asyncio
import json
import os
import tempfile

from taosmd.job_queue import JobQueue, JOB_ENRICH, RESOURCE_CPU
from taosmd.job_worker import JobWorker


def _run(coro):
    return asyncio.run(coro)


def test_worker_runs_enrich_job_gracefully():
    """Worker processes an enrich job (will fail due to no Ollama, but doesn't crash)."""
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        # Create a fake catalog with a session
        from taosmd.session_catalog import SessionCatalog
        catalog = SessionCatalog(
            db_path=os.path.join(tmp, "catalog.db"),
            archive_dir=os.path.join(tmp, "archive"),
            sessions_dir=os.path.join(tmp, "sessions"),
        )
        await catalog.init()

        # Enqueue an enrich job
        await q.enqueue(
            JOB_ENRICH,
            payload={
                "session_id": 999,  # Doesn't exist — will fail gracefully
                "model": "qwen3:4b",
                "llm_url": "http://localhost:99999",
                "catalog_db": os.path.join(tmp, "catalog.db"),
            },
            resource_type=RESOURCE_CPU,
        )

        worker = JobWorker(q, resource_types=["cpu"])
        result = await worker.run_once()

        # Job should have been processed (and failed gracefully)
        assert result is not None
        # The job should be marked as failed or completed
        stats = await q.stats()
        assert stats["total_running"] == 0  # Not stuck

        await catalog.close()
        await q.close()
    _run(run())


def test_worker_no_jobs_returns_none():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        worker = JobWorker(q)
        result = await worker.run_once()
        assert result is None

        await q.close()
    _run(run())


def test_worker_respects_resource_filter():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        # Enqueue GPU job
        await q.enqueue("extract", resource_type="gpu")

        # Worker only accepts CPU jobs
        worker = JobWorker(q, resource_types=["cpu"])
        result = await worker.run_once()
        assert result is None  # Can't take GPU jobs

        # But pending count is still 1
        assert await q.pending_count() == 1

        await q.close()
    _run(run())


def test_worker_stop():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        worker = JobWorker(q)
        worker.stop()
        processed = await worker.run_loop(poll_interval=0.1, max_idle=0.5)
        assert processed == 0  # Stopped immediately

        await q.close()
    _run(run())
