"""Tests for worker heartbeat protocol."""

import asyncio
import os
import tempfile
import time

from taosmd.worker_heartbeat import WorkerRegistry, OFFLINE_THRESHOLD


def _run(coro):
    return asyncio.run(coro)


def test_register_and_query_worker():
    async def run():
        tmp = tempfile.mkdtemp()
        reg = WorkerRegistry(os.path.join(tmp, "workers.db"))
        await reg.init()

        await reg.receive_heartbeat({
            "name": "fedora-gpu",
            "url": "http://192.168.1.100:6970",
            "cpu_cores": 12,
            "gpu_name": "RTX 3060",
            "gpu_vram_mb": 12288,
            "gpu_utilisation": 45,
            "ram_total_mb": 32768,
            "ram_available_mb": 24000,
            "models": ["qwen3.5:9b", "qwen3:4b"],
            "worker_type": "gaming",
        })

        workers = await reg.online_workers()
        assert len(workers) == 1
        w = workers[0]
        assert w["name"] == "fedora-gpu"
        assert w["gpu"] is True
        assert "qwen3.5:9b" in w["models"]
        assert w["online"] is True

        await reg.close()
    _run(run())


def test_heartbeat_updates_existing():
    async def run():
        tmp = tempfile.mkdtemp()
        reg = WorkerRegistry(os.path.join(tmp, "workers.db"))
        await reg.init()

        await reg.receive_heartbeat({"name": "w1", "gpu_utilisation": 10, "models": ["qwen3:4b"]})
        await reg.receive_heartbeat({"name": "w1", "gpu_utilisation": 95, "models": ["qwen3:4b"]})

        w = await reg.get_worker("w1")
        assert w["gpu_utilisation"] == 95
        assert len(await reg.online_workers()) == 1  # Still 1, not 2

        await reg.close()
    _run(run())


def test_offline_detection():
    async def run():
        tmp = tempfile.mkdtemp()
        reg = WorkerRegistry(os.path.join(tmp, "workers.db"))
        await reg.init()

        # Register a worker with an old heartbeat
        await reg.receive_heartbeat({"name": "old-worker", "models": []})
        # Manually backdate it
        reg._conn.execute(
            "UPDATE workers SET last_heartbeat = ? WHERE name = ?",
            (time.time() - OFFLINE_THRESHOLD - 10, "old-worker"),
        )
        reg._conn.commit()

        online = await reg.online_workers()
        assert len(online) == 0

        all_w = await reg.all_workers()
        assert len(all_w) == 1
        assert all_w[0]["online"] is False

        await reg.close()
    _run(run())


def test_yielded_workers_excluded_from_resource_manager():
    async def run():
        tmp = tempfile.mkdtemp()
        reg = WorkerRegistry(os.path.join(tmp, "workers.db"))
        await reg.init()

        await reg.receive_heartbeat({
            "name": "gaming-pc",
            "gpu_name": "RTX 4090",
            "models": ["qwen3.5:27b"],
            "is_yielded": True,  # User is gaming
        })
        await reg.receive_heartbeat({
            "name": "server",
            "gpu_name": "RTX 3060",
            "models": ["qwen3.5:9b"],
            "is_yielded": False,
        })

        # For resource manager — yielded workers excluded
        rm_workers = await reg.for_resource_manager()
        assert len(rm_workers) == 1
        assert rm_workers[0]["name"] == "server"

        # But all_workers still shows both
        all_w = await reg.all_workers()
        assert len(all_w) == 2

        await reg.close()
    _run(run())


def test_multiple_workers():
    async def run():
        tmp = tempfile.mkdtemp()
        reg = WorkerRegistry(os.path.join(tmp, "workers.db"))
        await reg.init()

        await reg.receive_heartbeat({"name": "pi4", "cpu_cores": 4, "models": ["qwen3:4b"], "worker_type": "headless"})
        await reg.receive_heartbeat({"name": "fedora", "gpu_name": "RTX 3060", "models": ["qwen3.5:9b"], "worker_type": "gaming"})
        await reg.receive_heartbeat({"name": "macbook", "cpu_cores": 10, "models": ["qwen3.5:4b"], "worker_type": "general"})

        stats = await reg.stats()
        assert stats["online"] == 3
        assert stats["gpu_workers"] == 1

        rm_workers = await reg.for_resource_manager()
        assert len(rm_workers) == 3
        gpu_names = [w["name"] for w in rm_workers if w["gpu"]]
        assert "fedora" in gpu_names

        await reg.close()
    _run(run())
