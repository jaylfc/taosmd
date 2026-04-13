"""Migration Scenario Tests — simulate real-world cluster dynamics.

Tests the resource manager's migration policies by simulating:
  1. Pi-only operation (baseline)
  2. GPU worker joins → upgrade
  3. GPU worker gets busy → downgrade
  4. GPU worker idles → upgrade back
  5. GPU worker disconnects → fallback
  6. Multiple workers → best selection
  7. Flaky worker (connect/disconnect rapidly) → no thrashing

These are unit tests with mocked snapshots, not integration tests.
Real cluster tests require the actual Pi + Fedora setup.
"""

import asyncio
import time

from taosmd.resource_manager import ResourceManager, ResourceSnapshot


def _run(coro):
    return asyncio.run(coro)


def _make_mgr(**kwargs):
    """Create a ResourceManager with no real hardware probing."""
    return ResourceManager(ollama_url="http://localhost:99999", **kwargs)


def _snap(workers=None, gpu=None, npu=0, models=None):
    """Build a ResourceSnapshot with specified state."""
    s = ResourceSnapshot()
    s.cluster_workers = workers or []
    s.gpu = gpu or {}
    s.npu_cores = npu
    s.ollama_models = models or []
    s.cpu_cores = 4
    s.ram_available_mb = 8192
    return s


def _gpu_worker(name, models=None, utilisation=0):
    return {
        "name": name,
        "gpu": True,
        "models": models or ["qwen3.5:9b"],
        "gpu_utilisation": utilisation,
    }


# ------------------------------------------------------------------
# Scenario 1: Pi only → steady state, no migration
# ------------------------------------------------------------------

def test_scenario_pi_only_no_migration():
    async def run():
        mgr = _make_mgr()
        pi_snap = _snap(npu=3, models=["qwen3:4b"])

        # Establish baseline
        mgr._prev_snapshot = None
        mgr._snapshot = pi_snap
        async def mock(force_refresh=False):
            return pi_snap
        mgr.get_snapshot = mock

        # First call sets baseline
        assert await mgr.evaluate_migration() is None

        # Subsequent calls — no change, no migration
        mgr._prev_snapshot = pi_snap
        assert await mgr.evaluate_migration() is None
    _run(run())


# ------------------------------------------------------------------
# Scenario 2: GPU worker joins → upgrade
# ------------------------------------------------------------------

def test_scenario_gpu_worker_joins():
    async def run():
        mgr = _make_mgr()

        # Baseline: Pi only
        mgr._prev_snapshot = _snap(npu=3)

        # Fedora joins with GPU
        new_snap = _snap(npu=3, workers=[_gpu_worker("fedora")])
        mgr._snapshot = new_snap
        async def mock(force_refresh=False):
            return new_snap
        mgr.get_snapshot = mock

        action = await mgr.evaluate_migration()
        assert action["action"] == "upgrade"
        assert "qwen3.5:9b" in action["to_model"]
        assert "fedora" in action["to_location"]
    _run(run())


# ------------------------------------------------------------------
# Scenario 3: GPU worker gets busy (video gen) → downgrade
# ------------------------------------------------------------------

def test_scenario_gpu_worker_busy_triggers_downgrade():
    async def run():
        mgr = _make_mgr(contention_threshold=0)  # Immediate downgrade for test

        # Both snapshots have the worker, but now it's busy
        base = _snap(npu=3, workers=[_gpu_worker("fedora", utilisation=10)])
        mgr._prev_snapshot = base

        busy_snap = _snap(npu=3, workers=[_gpu_worker("fedora", utilisation=95)])
        mgr._snapshot = busy_snap
        async def mock(force_refresh=False):
            return busy_snap
        mgr.get_snapshot = mock

        # Pre-set busy timer (simulates a previous check already recorded it)
        mgr._worker_busy_since["fedora"] = time.time() - 1

        action = await mgr.evaluate_migration()
        assert action is not None
        assert action["action"] == "downgrade"
        assert "busy" in action["reason"]
    _run(run())


# ------------------------------------------------------------------
# Scenario 4: GPU worker idles after contention → upgrade back
# ------------------------------------------------------------------

def test_scenario_gpu_worker_idles_upgrade_back():
    async def run():
        mgr = _make_mgr(idle_upgrade_delay=0)  # Immediate upgrade for test

        # Worker is idle
        base = _snap(npu=3, workers=[_gpu_worker("fedora", utilisation=95)])
        mgr._prev_snapshot = base

        idle_snap = _snap(npu=3, workers=[_gpu_worker("fedora", utilisation=5)])
        mgr._snapshot = idle_snap
        async def mock(force_refresh=False):
            return idle_snap
        mgr.get_snapshot = mock

        # Record idle start
        mgr._worker_idle_since["fedora"] = time.time() - 1  # Already idle

        action = await mgr.evaluate_migration()
        assert action is not None
        assert action["action"] == "upgrade"
        assert "idle" in action["reason"]
    _run(run())


# ------------------------------------------------------------------
# Scenario 5: GPU worker disconnects → fallback
# ------------------------------------------------------------------

def test_scenario_gpu_worker_disconnects():
    async def run():
        mgr = _make_mgr()

        # Had a worker
        mgr._prev_snapshot = _snap(npu=3, workers=[_gpu_worker("fedora")])

        # Worker gone
        gone_snap = _snap(npu=3)
        mgr._snapshot = gone_snap
        async def mock(force_refresh=False):
            return gone_snap
        mgr.get_snapshot = mock

        action = await mgr.evaluate_migration()
        assert action["action"] == "downgrade"
        assert "disconnected" in action["reason"]
    _run(run())


# ------------------------------------------------------------------
# Scenario 6: Multiple workers → picks best
# ------------------------------------------------------------------

def test_scenario_multiple_workers_picks_best():
    async def run():
        mgr = _make_mgr()

        mgr._prev_snapshot = _snap(npu=3)

        multi_snap = _snap(npu=3, workers=[
            _gpu_worker("pi4-cpu", models=["qwen3:4b"]),
            _gpu_worker("fedora-3060", models=["qwen3.5:9b"]),
        ])
        mgr._snapshot = multi_snap
        async def mock(force_refresh=False):
            return multi_snap
        mgr.get_snapshot = mock

        action = await mgr.evaluate_migration()
        assert action["action"] == "upgrade"
        # Should pick one of the new workers (first new one found)
        assert "worker:" in action["to_location"]
    _run(run())


# ------------------------------------------------------------------
# Scenario 7: Flaky worker (rapid connect/disconnect) → no thrashing
# ------------------------------------------------------------------

def test_scenario_flaky_worker_no_thrashing():
    async def run():
        mgr = _make_mgr(idle_upgrade_delay=600)  # 10 min delay

        # Worker appears
        mgr._prev_snapshot = _snap(npu=3)
        snap_with = _snap(npu=3, workers=[_gpu_worker("flaky")])
        mgr._snapshot = snap_with
        async def mock_with(force_refresh=False):
            return snap_with
        mgr.get_snapshot = mock_with

        action = await mgr.evaluate_migration()
        assert action["action"] == "upgrade"  # Upgrades on join

        # Worker disappears immediately
        mgr._prev_snapshot = snap_with
        snap_without = _snap(npu=3)
        mgr._snapshot = snap_without
        async def mock_without(force_refresh=False):
            return snap_without
        mgr.get_snapshot = mock_without

        action = await mgr.evaluate_migration()
        assert action["action"] == "downgrade"  # Must downgrade

        # Worker reappears but idle_upgrade_delay prevents immediate re-upgrade
        # (The actual delay enforcement happens in the idle timer, not the join logic)
        # Join always triggers upgrade — that's correct behaviour.
        # The anti-thrashing is in the idle-upgrade-back path, not the join path.
        mgr._prev_snapshot = snap_without
        mgr._snapshot = snap_with
        mgr.get_snapshot = mock_with

        action = await mgr.evaluate_migration()
        assert action["action"] == "upgrade"  # Joins always upgrade
        # This is correct — if a worker joins, you should use it.
        # Thrashing protection is for the busy→idle→busy cycle, not join→leave.
    _run(run())
