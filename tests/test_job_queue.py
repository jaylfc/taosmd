"""Tests for the job queue."""

import asyncio
import os
import tempfile

from taosmd.job_queue import JobQueue, Priority, RESOURCE_CPU, RESOURCE_NPU


def _run(coro):
    return asyncio.run(coro)


def test_enqueue_and_dequeue():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        job_id = await q.enqueue("embed", payload={"text": "hello"}, resource_type=RESOURCE_CPU)
        assert job_id

        job = await q.dequeue()
        assert job is not None
        assert job["id"] == job_id
        assert job["status"] == "running"

        # No more jobs
        assert await q.dequeue() is None
        await q.close()
    _run(run())


def test_priority_ordering():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        low = await q.enqueue("split", priority=Priority.BACKGROUND)
        high = await q.enqueue("enrich", priority=Priority.URGENT)

        job = await q.dequeue()
        assert job["id"] == high  # Urgent first
        await q.complete(high)

        job = await q.dequeue()
        assert job["id"] == low
        await q.close()
    _run(run())


def test_concurrency_limits():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()
        await q.set_limit(RESOURCE_NPU, 2)  # Allow 2 concurrent NPU jobs

        j1 = await q.enqueue("extract", resource_type=RESOURCE_NPU)
        j2 = await q.enqueue("extract", resource_type=RESOURCE_NPU)
        j3 = await q.enqueue("extract", resource_type=RESOURCE_NPU)

        # Dequeue 2 — both should succeed
        assert (await q.dequeue()) is not None  # j1
        assert (await q.dequeue()) is not None  # j2

        # Third should be blocked (limit=2)
        assert (await q.dequeue(resource_types=[RESOURCE_NPU])) is None

        # Complete one, third should now be available
        await q.complete(j1)
        job = await q.dequeue()
        assert job is not None
        assert job["id"] == j3
        await q.close()
    _run(run())


def test_complete_and_fail():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        j1 = await q.enqueue("embed")
        await q.dequeue()  # Claim it
        assert await q.complete(j1, result={"status": "ok"})

        j2 = await q.enqueue("extract")
        await q.dequeue()
        assert await q.fail(j2, "model crashed")

        job = await q.get_job(j2)
        assert job["status"] == "failed"
        assert job["error"] == "model crashed"
        await q.close()
    _run(run())


def test_stale_job_recovery():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        # Simulate a job left running from a crash
        j1 = await q.enqueue("index")
        await q.dequeue()  # Now "running"

        # Re-init (simulates restart)
        await q.close()
        q2 = JobQueue(os.path.join(tmp, "q.db"))
        await q2.init()

        # Stale job should be marked failed
        job = await q2.get_job(j1)
        assert job["status"] == "failed"
        assert "stale" in job["error"]
        await q2.close()
    _run(run())


def test_stats():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        await q.enqueue("embed", resource_type=RESOURCE_CPU)
        await q.enqueue("extract", resource_type=RESOURCE_NPU)
        await q.dequeue()  # Claim first

        stats = await q.stats()
        assert stats["total_pending"] == 1
        assert stats["total_running"] == 1
        assert stats["limits"][RESOURCE_NPU] == 3
        await q.close()
    _run(run())
