"""Tests for the resource manager."""

import asyncio
import os
import tempfile

from taosmd.job_queue import JobQueue
from taosmd.resource_manager import (
    ResourceManager, ResourceSnapshot, _count_cpu_cores, _detect_npu,
)


def _run(coro):
    return asyncio.run(coro)


def test_cpu_detection():
    cores = _count_cpu_cores()
    assert cores >= 1


def test_npu_detection():
    # Just verify it doesn't crash — result depends on hardware
    npu = _detect_npu()
    assert isinstance(npu, int)
    assert npu >= 0


def test_snapshot_properties():
    snap = ResourceSnapshot()
    snap.cpu_cores = 8
    snap.npu_cores = 3
    snap.gpu = {"name": "RTX 3060", "vram_mb": 12288, "count": 1}
    snap.ollama_models = ["qwen3:4b"]

    assert snap.has_gpu
    assert snap.has_npu
    assert snap.has_ollama
    d = snap.to_dict()
    assert d["cpu_cores"] == 8
    assert d["npu_cores"] == 3


def test_snapshot_no_hardware():
    snap = ResourceSnapshot()
    assert not snap.has_gpu
    assert not snap.has_npu
    assert not snap.has_ollama


def test_refresh_updates_queue_limits():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        mgr = ResourceManager(job_queue=q, ollama_url="http://localhost:99999")
        snap = await mgr.refresh()

        # Should have set limits based on detected hardware
        limits = await q.get_limits()
        assert limits["cpu"] >= 1  # At least 1
        assert limits["embed"] == 1  # Always 1
        assert snap.cpu_cores >= 1

        await q.close()
    _run(run())


def test_best_model_no_ollama():
    async def run():
        mgr = ResourceManager(ollama_url="http://localhost:99999")
        result = await mgr.best_model_for_task("extract")
        assert result == {}  # No models available

        result = await mgr.best_model_for_task("embed")
        assert result["model"] == "all-MiniLM-L6-v2"  # Embedding always available
    _run(run())


def test_migration_worker_joins():
    async def run():
        mgr = ResourceManager(ollama_url="http://localhost:99999")
        # First refresh — establishes baseline (no prev)
        mgr._snapshot = ResourceSnapshot()
        mgr._prev_snapshot = None

        action = await mgr.evaluate_migration()
        assert action is None  # No prev snapshot yet

        # Simulate: GPU worker appears
        mgr._prev_snapshot = ResourceSnapshot()  # Empty
        snap = ResourceSnapshot()
        snap.cluster_workers = [{"name": "fedora", "gpu": True, "models": ["qwen3.5:9b"]}]
        mgr._snapshot = snap
        mgr._last_refresh = 0

        async def mock_snap(force_refresh=False):
            return snap
        mgr.get_snapshot = mock_snap

        action = await mgr.evaluate_migration()
        assert action is not None
        assert action["action"] == "upgrade"
        assert "fedora" in action["to_location"]
    _run(run())


def test_yield_and_reclaim():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        mgr = ResourceManager(job_queue=q, ollama_url="http://localhost:99999")
        await mgr.refresh()

        # Full power — CPU should use all cores
        limits = await q.get_limits()
        full_cpu = limits["cpu"]
        assert full_cpu >= 1

        # Yield — throttle everything
        result = await mgr.yield_resources()
        assert result["cpu"] == 1
        assert result["gpu"] == 0
        assert mgr.is_yielded

        limits = await q.get_limits()
        assert limits["cpu"] == 1
        assert limits["gpu"] == 0

        # Refresh should NOT override yielded limits
        await mgr.refresh()
        limits = await q.get_limits()
        assert limits["cpu"] == 1  # Still throttled

        # Reclaim — back to full power
        result = await mgr.reclaim_resources()
        assert not mgr.is_yielded
        limits = await q.get_limits()
        assert limits["cpu"] == full_cpu  # Back to full

        await q.close()
    _run(run())


def test_migration_worker_disconnects():
    async def run():
        mgr = ResourceManager(ollama_url="http://localhost:99999")

        prev = ResourceSnapshot()
        prev.cluster_workers = [{"name": "fedora", "gpu": True, "models": ["qwen3.5:9b"]}]
        mgr._prev_snapshot = prev

        curr = ResourceSnapshot()
        curr.cluster_workers = []
        mgr._snapshot = curr

        async def mock_snap(force_refresh=False):
            return curr
        mgr.get_snapshot = mock_snap

        action = await mgr.evaluate_migration()
        assert action is not None
        assert action["action"] == "downgrade"
        assert "disconnected" in action["reason"]
    _run(run())


def test_no_migration_when_stable():
    async def run():
        mgr = ResourceManager(ollama_url="http://localhost:99999")

        snap = ResourceSnapshot()
        snap.cluster_workers = [{"name": "fedora", "gpu": True, "models": ["qwen3.5:9b"], "gpu_utilisation": 50}]
        mgr._prev_snapshot = ResourceSnapshot()
        mgr._prev_snapshot.cluster_workers = list(snap.cluster_workers)
        mgr._snapshot = snap

        async def mock_snap(force_refresh=False):
            return snap
        mgr.get_snapshot = mock_snap

        action = await mgr.evaluate_migration()
        assert action is None
    _run(run())


def test_can_accept_job():
    async def run():
        tmp = tempfile.mkdtemp()
        q = JobQueue(os.path.join(tmp, "q.db"))
        await q.init()

        mgr = ResourceManager(job_queue=q)
        await mgr.refresh()

        # Force limit to 1 AFTER refresh (refresh may set it higher)
        await q.set_limit("cpu", 1)

        # No running jobs — should accept
        assert await mgr.can_accept_job("cpu")

        # Fill the slot
        await q.enqueue("test", resource_type="cpu")
        await q.dequeue()

        # Now full — should reject
        assert not await mgr.can_accept_job("cpu")

        await q.close()
    _run(run())
