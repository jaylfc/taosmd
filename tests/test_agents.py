"""Tests for the agent registry — multi-agent isolation primitives."""

from __future__ import annotations

import json
import time

import pytest

from taosmd.agents import (
    AgentExistsError,
    AgentNotFoundError,
    AgentRegistry,
    InvalidAgentNameError,
    LIBRARIAN_TASKS,
    FANOUT_LEVELS,
)


@pytest.fixture
def registry(tmp_path):
    return AgentRegistry(tmp_path)


# --- registration -----------------------------------------------------------


def test_register_creates_record_and_dir(registry, tmp_path):
    record = registry.register_agent("alice", display_name="Alice the Researcher")
    assert record["name"] == "alice"
    assert record["display_name"] == "Alice the Researcher"
    assert record["created_at"] > 0
    assert record["last_ingest_at"] == 0
    assert record["total_chunks"] == 0
    # Per-agent dir created eagerly
    assert (tmp_path / "agent-memory" / "alice").is_dir()
    # Registry persisted to disk
    on_disk = json.loads((tmp_path / "agents.json").read_text())
    assert on_disk["agents"][0]["name"] == "alice"


def test_register_defaults_display_name_to_name(registry):
    record = registry.register_agent("bob")
    assert record["display_name"] == "bob"


def test_register_duplicate_raises(registry):
    registry.register_agent("alice")
    with pytest.raises(AgentExistsError):
        registry.register_agent("alice")


def test_register_clobber_overwrites(registry):
    first = registry.register_agent("alice", display_name="First")
    time.sleep(0.01)
    second = registry.register_agent("alice", display_name="Second", clobber=True)
    assert second["display_name"] == "Second"
    assert second["created_at"] >= first["created_at"]
    # Still only one alice
    assert len(registry.list_agents()) == 1


# --- name validation --------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "Has Spaces",
        "UPPERCASE",
        "-starts-with-dash",
        "1starts-with-digit",
        "has.dot",
        "x" * 64,  # too long
        "has/slash",
        "has spaces",
    ],
)
def test_invalid_names_rejected(registry, bad_name):
    with pytest.raises(InvalidAgentNameError):
        registry.register_agent(bad_name)


@pytest.mark.parametrize(
    "good_name",
    [
        "alice",
        "openclaw-research",
        "agent_1",
        "x",
        "a" + "b" * 62,  # exactly 63 chars
    ],
)
def test_valid_names_accepted(registry, good_name):
    registry.register_agent(good_name)


# --- list / exists / get ----------------------------------------------------


def test_list_agents_empty(registry):
    assert registry.list_agents() == []


def test_list_agents_returns_all(registry):
    registry.register_agent("alice")
    registry.register_agent("bob")
    names = {a["name"] for a in registry.list_agents()}
    assert names == {"alice", "bob"}


def test_agent_exists(registry):
    assert registry.agent_exists("alice") is False
    registry.register_agent("alice")
    assert registry.agent_exists("alice") is True


def test_get_agent_returns_record(registry):
    registry.register_agent("alice")
    record = registry.get_agent("alice")
    assert record["name"] == "alice"


def test_get_agent_missing_raises(registry):
    with pytest.raises(AgentNotFoundError):
        registry.get_agent("nonexistent")


# --- delete -----------------------------------------------------------------


def test_delete_agent_keeps_data_by_default(registry, tmp_path):
    registry.register_agent("alice")
    agent_dir = tmp_path / "agent-memory" / "alice"
    (agent_dir / "marker.txt").write_text("hello")
    registry.delete_agent("alice")
    assert registry.agent_exists("alice") is False
    # Index files preserved unless drop_data=True
    assert (agent_dir / "marker.txt").exists()


def test_delete_agent_drop_data_removes_dir(registry, tmp_path):
    registry.register_agent("alice")
    agent_dir = tmp_path / "agent-memory" / "alice"
    (agent_dir / "marker.txt").write_text("hello")
    registry.delete_agent("alice", drop_data=True)
    assert not agent_dir.exists()


def test_delete_missing_raises(registry):
    with pytest.raises(AgentNotFoundError):
        registry.delete_agent("nonexistent")


# --- ensure_agent (back-compat lazy registration) --------------------------


def test_ensure_agent_creates_when_missing(registry):
    record = registry.ensure_agent("alice")
    assert record["name"] == "alice"
    assert registry.agent_exists("alice") is True


def test_ensure_agent_returns_existing(registry):
    first = registry.register_agent("alice", display_name="Alice")
    second = registry.ensure_agent("alice")
    assert second["display_name"] == "Alice"
    assert second["created_at"] == first["created_at"]


# --- update_stats -----------------------------------------------------------


def test_update_stats_patches_record(registry):
    registry.register_agent("alice")
    updated = registry.update_stats("alice", last_ingest_at=12345, total_chunks=42)
    assert updated["last_ingest_at"] == 12345
    assert updated["total_chunks"] == 42
    # Persisted
    again = registry.get_agent("alice")
    assert again["total_chunks"] == 42


def test_update_stats_partial(registry):
    registry.register_agent("alice")
    registry.update_stats("alice", total_chunks=10)
    registry.update_stats("alice", last_ingest_at=999)
    record = registry.get_agent("alice")
    assert record["total_chunks"] == 10
    assert record["last_ingest_at"] == 999


# --- multi-agent isolation --------------------------------------------------


def test_multiple_agents_get_separate_dirs(registry, tmp_path):
    registry.register_agent("alice")
    registry.register_agent("bob")
    assert (tmp_path / "agent-memory" / "alice").is_dir()
    assert (tmp_path / "agent-memory" / "bob").is_dir()


def test_corrupt_registry_treated_as_empty(tmp_path):
    (tmp_path / "agents.json").write_text("{not json")
    registry = AgentRegistry(tmp_path)
    assert registry.list_agents() == []
    # Recovers by overwriting on next register
    registry.register_agent("alice")
    assert registry.agent_exists("alice")


def test_atomic_write_does_not_leave_tmp_files(registry, tmp_path):
    registry.register_agent("alice")
    registry.register_agent("bob")
    # Tmp file should be cleaned up by the atomic rename
    assert not (tmp_path / "agents.json.tmp").exists()


# --- librarian config -------------------------------------------------------


def test_register_seeds_default_librarian(registry):
    registry.register_agent("alice")
    lib = registry.get_librarian("alice")
    assert lib["enabled"] is True
    assert lib["model"] is None
    assert set(lib["tasks"].keys()) == set(LIBRARIAN_TASKS)
    assert all(lib["tasks"].values())


def test_get_librarian_missing_agent_raises(registry):
    with pytest.raises(AgentNotFoundError):
        registry.get_librarian("nobody")


def test_set_librarian_master_switch(registry):
    registry.register_agent("alice")
    lib = registry.set_librarian("alice", enabled=False)
    assert lib["enabled"] is False
    # Persisted across reads
    assert registry.get_librarian("alice")["enabled"] is False


def test_set_librarian_model_override_and_clear(registry):
    registry.register_agent("alice")
    registry.set_librarian("alice", model="ollama:qwen3:4b")
    assert registry.get_librarian("alice")["model"] == "ollama:qwen3:4b"
    registry.set_librarian("alice", clear_model=True)
    assert registry.get_librarian("alice")["model"] is None


def test_set_librarian_per_task_toggles(registry):
    registry.register_agent("alice")
    registry.set_librarian("alice", tasks={"reflect": False, "crystallise": False})
    lib = registry.get_librarian("alice")
    assert lib["tasks"]["reflect"] is False
    assert lib["tasks"]["crystallise"] is False
    # Other tasks untouched
    assert lib["tasks"]["fact_extraction"] is True


def test_set_librarian_unknown_task_rejected(registry):
    registry.register_agent("alice")
    with pytest.raises(ValueError):
        registry.set_librarian("alice", tasks={"hallucinate_more": True})


def test_is_task_enabled_master_off_overrides_per_task(registry):
    registry.register_agent("alice")
    # Per-task is on by default but master is off → must be False
    registry.set_librarian("alice", enabled=False)
    assert registry.is_task_enabled("alice", "fact_extraction") is False
    # Master back on, per-task off → False
    registry.set_librarian("alice", enabled=True, tasks={"fact_extraction": False})
    assert registry.is_task_enabled("alice", "fact_extraction") is False
    # Both on → True
    registry.set_librarian("alice", tasks={"fact_extraction": True})
    assert registry.is_task_enabled("alice", "fact_extraction") is True


def test_is_task_enabled_unknown_agent(registry):
    assert registry.is_task_enabled("nobody", "fact_extraction") is False


def test_is_task_enabled_unknown_task(registry):
    registry.register_agent("alice")
    assert registry.is_task_enabled("alice", "not_a_real_task") is False


def test_legacy_record_without_librarian_field(registry, tmp_path):
    # Simulate an older agents.json written before the librarian field
    # existed. Reading it should return the default config.
    legacy = {
        "agents": [
            {
                "name": "legacy-bob",
                "display_name": "Legacy Bob",
                "created_at": 1700000000,
                "last_ingest_at": 0,
                "total_chunks": 0,
            }
        ]
    }
    (tmp_path / "agents.json").write_text(json.dumps(legacy))
    lib = registry.get_librarian("legacy-bob")
    assert lib["enabled"] is True
    assert set(lib["tasks"].keys()) == set(LIBRARIAN_TASKS)


# --- fanout config ----------------------------------------------------------


def test_default_librarian_includes_fanout_block(registry):
    registry.register_agent("alice")
    lib = registry.get_librarian("alice")
    assert "fanout" in lib
    assert lib["fanout"]["default"] == "low"
    assert lib["fanout"]["auto_scale"] is True


def test_set_librarian_patches_fanout_level(registry):
    registry.register_agent("alice")
    lib = registry.set_librarian("alice", fanout="med")
    assert lib["fanout"]["default"] == "med"
    # Other fanout sub-key untouched
    assert lib["fanout"]["auto_scale"] is True
    # Persisted
    assert registry.get_librarian("alice")["fanout"]["default"] == "med"


def test_set_librarian_patches_fanout_auto_scale(registry):
    registry.register_agent("alice")
    lib = registry.set_librarian("alice", fanout_auto_scale=False)
    assert lib["fanout"]["auto_scale"] is False
    # Fanout level untouched
    assert lib["fanout"]["default"] == "low"
    # Persisted
    assert registry.get_librarian("alice")["fanout"]["auto_scale"] is False


def test_set_librarian_unknown_fanout_level_raises(registry):
    registry.register_agent("alice")
    with pytest.raises(ValueError, match="unknown fanout level"):
        registry.set_librarian("alice", fanout="ultra")


# --- effective_fanout -------------------------------------------------------


def test_effective_fanout_pi_worker_returns_low(registry):
    # Pi-class: no GPU, no turboquant — should stay at default (low = K=3)
    registry.register_agent("alice")
    k = registry.effective_fanout("alice", worker_capabilities={"gpu_vram_gb": 0, "turboquant": False})
    assert k == FANOUT_LEVELS["low"]
    assert k == 3


def test_effective_fanout_no_caps_returns_low(registry):
    # No caps at all — should stay at default (low = K=3)
    registry.register_agent("alice")
    k = registry.effective_fanout("alice", worker_capabilities=None)
    assert k == FANOUT_LEVELS["low"]


def test_effective_fanout_gpu_worker_bumps_tier(registry):
    # GPU worker with TurboQuant + ≥12 GB VRAM: low → med (K=10)
    registry.register_agent("alice")
    k = registry.effective_fanout("alice", worker_capabilities={"gpu_vram_gb": 12, "turboquant": True})
    assert k == FANOUT_LEVELS["med"]
    assert k == 10


def test_effective_fanout_auto_scale_off_no_bump(registry):
    # Even with a capable GPU worker, auto_scale=False keeps the configured level
    registry.register_agent("alice")
    registry.set_librarian("alice", fanout_auto_scale=False)
    k = registry.effective_fanout("alice", worker_capabilities={"gpu_vram_gb": 16, "turboquant": True})
    assert k == FANOUT_LEVELS["low"]
    assert k == 3


def test_effective_fanout_high_baseline_no_over_scale(registry):
    # high + GPU worker should not exceed high (K=20)
    registry.register_agent("alice")
    registry.set_librarian("alice", fanout="high")
    k = registry.effective_fanout("alice", worker_capabilities={"gpu_vram_gb": 24, "turboquant": True})
    assert k == FANOUT_LEVELS["high"]
    assert k == 20


def test_legacy_record_without_fanout_defaults_to_low(registry, tmp_path):
    # Librarian record that predates the fanout key should auto-populate it.
    legacy = {
        "agents": [
            {
                "name": "old-agent",
                "display_name": "Old Agent",
                "created_at": 1700000000,
                "last_ingest_at": 0,
                "total_chunks": 0,
                "librarian": {
                    "enabled": True,
                    "model": None,
                    "tasks": {t: True for t in LIBRARIAN_TASKS},
                    # no "fanout" key — simulates a record saved before this feature
                },
            }
        ]
    }
    (tmp_path / "agents.json").write_text(json.dumps(legacy))
    lib = registry.get_librarian("old-agent")
    assert "fanout" in lib
    assert lib["fanout"]["default"] == "low"
    assert lib["fanout"]["auto_scale"] is True
    # effective_fanout should still resolve correctly
    k = registry.effective_fanout("old-agent")
    assert k == FANOUT_LEVELS["low"]
