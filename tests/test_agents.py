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
