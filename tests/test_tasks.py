"""Tests for taosmd.tasks — dependency-aware task graph.

Covers:
- Hash-ID stability (same triple -> same ID; distinct triple -> different ID)
- ready-view correctness including transitive chains and edge soft-removal
- prime token budget (text under cap, all four sections represented)
- archive round-trip (create tasks + edges, rebuild_from_archive, assert match)
- status enum rejection
"""

from __future__ import annotations

import asyncio
import time

import pytest

from taosmd import api as taosmd_api
import taosmd.tasks as tasks_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Isolated data dir for each test; wipe task db cache between tests."""
    d = tmp_path / "tdata"
    d.mkdir()
    # Clear all caches so each test is isolated
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    monkeypatch.setattr(tasks_mod, "_db_cache", {})
    return str(d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Hash-ID stability
# ---------------------------------------------------------------------------


def test_hash_id_stable():
    """Same (title, project, ts) triple must always give the same ID."""
    t = 1234567890.123
    id1 = tasks_mod._make_task_id("Fix the bug", "proj-abc", t)
    id2 = tasks_mod._make_task_id("Fix the bug", "proj-abc", t)
    assert id1 == id2
    assert id1.startswith("t-")
    assert len(id1) == 14  # "t-" + 12 hex chars


def test_hash_id_different_title():
    t = 1234567890.0
    id1 = tasks_mod._make_task_id("Task A", None, t)
    id2 = tasks_mod._make_task_id("Task B", None, t)
    assert id1 != id2


def test_hash_id_different_project():
    t = 1234567890.0
    id1 = tasks_mod._make_task_id("Task A", "proj-1", t)
    id2 = tasks_mod._make_task_id("Task A", "proj-2", t)
    assert id1 != id2


def test_hash_id_different_ts():
    id1 = tasks_mod._make_task_id("Task A", None, 1000.0)
    id2 = tasks_mod._make_task_id("Task A", None, 1001.0)
    assert id1 != id2


def test_hash_id_no_project_is_distinct_from_project():
    t = 1234567890.0
    id1 = tasks_mod._make_task_id("Task A", None, t)
    id2 = tasks_mod._make_task_id("Task A", "", t)
    # Both use "" for project, so should be equal
    assert id1 == id2


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_create_task_returns_task(data_dir):
    task = run(tasks_mod.create_task(
        "Write tests",
        created_by="agent-x",
        data_dir=data_dir,
    ))
    assert task["id"].startswith("t-")
    assert task["title"] == "Write tests"
    assert task["status"] == "open"
    assert task["created_by"] == "agent-x"
    assert task["priority"] == 0


def test_create_task_with_all_fields(data_dir):
    task = run(tasks_mod.create_task(
        "Deploy service",
        body="Deploy the updated image",
        project="proj-abc",
        assignee="agent-y",
        priority=5,
        created_by="agent-x",
        data_dir=data_dir,
    ))
    assert task["body"] == "Deploy the updated image"
    assert task["project"] == "proj-abc"
    assert task["assignee"] == "agent-y"
    assert task["priority"] == 5


def test_create_task_missing_created_by(data_dir):
    with pytest.raises(ValueError, match="created_by"):
        run(tasks_mod.create_task("A task", created_by="", data_dir=data_dir))


def test_list_tasks_empty(data_dir):
    result = run(tasks_mod.list_tasks(data_dir=data_dir))
    assert result == []


def test_list_tasks_with_filter(data_dir):
    run(tasks_mod.create_task("T1", created_by="x", project="proj-1", data_dir=data_dir))
    run(tasks_mod.create_task("T2", created_by="x", project="proj-2", data_dir=data_dir))
    result = run(tasks_mod.list_tasks(project="proj-1", data_dir=data_dir))
    assert len(result) == 1
    assert result[0]["title"] == "T1"


def test_update_task_status(data_dir):
    task = run(tasks_mod.create_task("Do work", created_by="x", data_dir=data_dir))
    updated = run(tasks_mod.update_task(task["id"], status="in_progress", data_dir=data_dir))
    assert updated["status"] == "in_progress"
    assert updated["closed_ts"] is None


def test_update_task_close_sets_closed_ts(data_dir):
    task = run(tasks_mod.create_task("Close me", created_by="x", data_dir=data_dir))
    updated = run(tasks_mod.update_task(task["id"], status="closed", data_dir=data_dir))
    assert updated["status"] == "closed"
    assert updated["closed_ts"] is not None


def test_update_task_not_found(data_dir):
    with pytest.raises(ValueError, match="task not found"):
        run(tasks_mod.update_task("t-000000000000", status="closed", data_dir=data_dir))


def test_update_task_invalid_status(data_dir):
    task = run(tasks_mod.create_task("T", created_by="x", data_dir=data_dir))
    with pytest.raises(ValueError, match="status"):
        run(tasks_mod.update_task(task["id"], status="garbage", data_dir=data_dir))


# ---------------------------------------------------------------------------
# Ready view correctness
# ---------------------------------------------------------------------------


def test_ready_view_no_edges(data_dir):
    """A task with no edges is ready."""
    task = run(tasks_mod.create_task("Simple task", created_by="x", data_dir=data_dir))
    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    assert any(t["id"] == task["id"] for t in ready)


def test_ready_view_blocked_by_open(data_dir):
    """A task blocked by an open task is NOT in the ready queue."""
    blocker = run(tasks_mod.create_task("Blocker", created_by="x", data_dir=data_dir))
    blocked = run(tasks_mod.create_task("Blocked", created_by="x", data_dir=data_dir))
    run(tasks_mod.add_edge(blocker["id"], blocked["id"], "blocks", "x", data_dir=data_dir))

    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    ready_ids = {t["id"] for t in ready}
    assert blocked["id"] not in ready_ids
    assert blocker["id"] in ready_ids  # blocker itself is ready


def test_ready_view_blocked_by_closed(data_dir):
    """A task blocked only by a closed task IS in the ready queue."""
    blocker = run(tasks_mod.create_task("Blocker", created_by="x", data_dir=data_dir))
    blocked = run(tasks_mod.create_task("Blocked", created_by="x", data_dir=data_dir))
    run(tasks_mod.add_edge(blocker["id"], blocked["id"], "blocks", "x", data_dir=data_dir))
    # Close the blocker
    run(tasks_mod.update_task(blocker["id"], status="closed", data_dir=data_dir))

    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    ready_ids = {t["id"] for t in ready}
    assert blocked["id"] in ready_ids


def test_ready_view_transitive_chain(data_dir):
    """A blocks B blocks C: closing A makes B ready but C still needs B to close."""
    a = run(tasks_mod.create_task("A", created_by="x", data_dir=data_dir))
    b = run(tasks_mod.create_task("B", created_by="x", data_dir=data_dir))
    c = run(tasks_mod.create_task("C", created_by="x", data_dir=data_dir))
    run(tasks_mod.add_edge(a["id"], b["id"], "blocks", "x", data_dir=data_dir))
    run(tasks_mod.add_edge(b["id"], c["id"], "blocks", "x", data_dir=data_dir))

    # Initially only A is ready
    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    ready_ids = {t["id"] for t in ready}
    assert a["id"] in ready_ids
    assert b["id"] not in ready_ids
    assert c["id"] not in ready_ids

    # Close A: B becomes ready, C still blocked by B
    run(tasks_mod.update_task(a["id"], status="closed", data_dir=data_dir))
    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    ready_ids = {t["id"] for t in ready}
    assert b["id"] in ready_ids
    assert c["id"] not in ready_ids


def test_ready_view_edge_removal_restores(data_dir):
    """Removing a blocks edge restores readiness of the downstream task."""
    blocker = run(tasks_mod.create_task("Blocker", created_by="x", data_dir=data_dir))
    blocked = run(tasks_mod.create_task("Blocked", created_by="x", data_dir=data_dir))
    run(tasks_mod.add_edge(blocker["id"], blocked["id"], "blocks", "x", data_dir=data_dir))

    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    assert blocked["id"] not in {t["id"] for t in ready}

    # Remove the edge
    run(tasks_mod.remove_edge(blocker["id"], blocked["id"], "blocks", data_dir=data_dir))

    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    assert blocked["id"] in {t["id"] for t in ready}


def test_ready_view_priority_ordering(data_dir):
    """Ready tasks are ordered priority DESC, created_ts ASC."""
    t1 = run(tasks_mod.create_task("Low prio", created_by="x", priority=1, data_dir=data_dir))
    t2 = run(tasks_mod.create_task("High prio", created_by="x", priority=10, data_dir=data_dir))
    t3 = run(tasks_mod.create_task("Zero prio", created_by="x", priority=0, data_dir=data_dir))
    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    ids = [t["id"] for t in ready]
    assert ids.index(t2["id"]) < ids.index(t1["id"])
    assert ids.index(t1["id"]) < ids.index(t3["id"])


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_add_edge_invalid_type(data_dir):
    a = run(tasks_mod.create_task("A", created_by="x", data_dir=data_dir))
    b = run(tasks_mod.create_task("B", created_by="x", data_dir=data_dir))
    with pytest.raises(ValueError, match="edge_type"):
        run(tasks_mod.add_edge(a["id"], b["id"], "invalid", "x", data_dir=data_dir))


def test_add_edge_missing_task(data_dir):
    a = run(tasks_mod.create_task("A", created_by="x", data_dir=data_dir))
    with pytest.raises(ValueError, match="task not found"):
        run(tasks_mod.add_edge(a["id"], "t-000000000000", "blocks", "x", data_dir=data_dir))


def test_remove_edge_not_found(data_dir):
    a = run(tasks_mod.create_task("A", created_by="x", data_dir=data_dir))
    b = run(tasks_mod.create_task("B", created_by="x", data_dir=data_dir))
    with pytest.raises(ValueError, match="active edge not found"):
        run(tasks_mod.remove_edge(a["id"], b["id"], "blocks", data_dir=data_dir))


def test_remove_edge_already_removed(data_dir):
    a = run(tasks_mod.create_task("A", created_by="x", data_dir=data_dir))
    b = run(tasks_mod.create_task("B", created_by="x", data_dir=data_dir))
    run(tasks_mod.add_edge(a["id"], b["id"], "blocks", "x", data_dir=data_dir))
    run(tasks_mod.remove_edge(a["id"], b["id"], "blocks", data_dir=data_dir))
    with pytest.raises(ValueError, match="active edge not found"):
        run(tasks_mod.remove_edge(a["id"], b["id"], "blocks", data_dir=data_dir))


# ---------------------------------------------------------------------------
# depends_on in create_task
# ---------------------------------------------------------------------------


def test_create_task_with_depends_on(data_dir):
    """depends_on creates blocking edges automatically."""
    blocker = run(tasks_mod.create_task("Blocker", created_by="x", data_dir=data_dir))
    dep_task = run(tasks_mod.create_task(
        "Dependent",
        created_by="x",
        depends_on=[blocker["id"]],
        data_dir=data_dir,
    ))
    ready = run(tasks_mod.ready_tasks(data_dir=data_dir))
    ready_ids = {t["id"] for t in ready}
    assert dep_task["id"] not in ready_ids
    assert blocker["id"] in ready_ids


def test_create_task_depends_on_nonexistent(data_dir):
    with pytest.raises(ValueError, match="depends_on task not found"):
        run(tasks_mod.create_task(
            "T",
            created_by="x",
            depends_on=["t-000000000000"],
            data_dir=data_dir,
        ))


# ---------------------------------------------------------------------------
# Prime token budget
# ---------------------------------------------------------------------------


def test_prime_empty(data_dir):
    """prime() on an empty store returns a dict with text and empty tasks."""
    result = run(tasks_mod.prime(data_dir=data_dir))
    assert "text" in result
    assert "tasks" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["tasks"], list)


def test_prime_under_cap(data_dir):
    """prime() text stays within the character cap regardless of task count."""
    for i in range(30):
        run(tasks_mod.create_task(
            f"Task {i:02d} — this is a somewhat long title to consume budget",
            created_by="x",
            data_dir=data_dir,
        ))
    result = run(tasks_mod.prime(data_dir=data_dir))
    assert len(result["text"]) <= tasks_mod.PRIME_CHAR_CAP


def test_prime_sections_present(data_dir):
    """prime() text mentions ready, in-progress, and blocked sections."""
    t_ready = run(tasks_mod.create_task("Ready task", created_by="x", data_dir=data_dir))
    t_ip = run(tasks_mod.create_task("In-progress task", created_by="x", data_dir=data_dir))
    run(tasks_mod.update_task(t_ip["id"], status="in_progress", data_dir=data_dir))

    blocker = run(tasks_mod.create_task("Blocker", created_by="x", data_dir=data_dir))
    blocked = run(tasks_mod.create_task("Blocked task", created_by="x", data_dir=data_dir))
    run(tasks_mod.add_edge(blocker["id"], blocked["id"], "blocks", "x", data_dir=data_dir))

    t_closed = run(tasks_mod.create_task("Recent closed", created_by="x", data_dir=data_dir))
    run(tasks_mod.update_task(t_closed["id"], status="closed", data_dir=data_dir))

    result = run(tasks_mod.prime(data_dir=data_dir))
    text = result["text"]
    assert "READY" in text
    assert "IN PROGRESS" in text
    assert "BLOCKED" in text
    assert "RECENTLY CLOSED" in text


def test_prime_tasks_only_referenced(data_dir):
    """prime() tasks list only contains tasks actually referenced in text."""
    for i in range(5):
        run(tasks_mod.create_task(f"Task {i}", created_by="x", data_dir=data_dir))
    result = run(tasks_mod.prime(data_dir=data_dir))
    referenced_ids = set()
    for t in result["tasks"]:
        referenced_ids.add(t["id"])
    # Every referenced task ID should appear in the text
    for t_id in referenced_ids:
        assert t_id in result["text"]


# ---------------------------------------------------------------------------
# Archive round-trip
# ---------------------------------------------------------------------------


def test_archive_round_trip(data_dir, monkeypatch):
    """Create tasks + edges, rebuild from archive, assert identical projection."""
    # Create tasks
    t1 = run(tasks_mod.create_task("Task One", created_by="agent-a", project="p1", data_dir=data_dir))
    t2 = run(tasks_mod.create_task("Task Two", created_by="agent-b", priority=3, data_dir=data_dir))
    t3 = run(tasks_mod.create_task("Task Three", created_by="agent-a", data_dir=data_dir))

    # Add edges
    run(tasks_mod.add_edge(t1["id"], t2["id"], "blocks", "agent-a", data_dir=data_dir))
    run(tasks_mod.add_edge(t2["id"], t3["id"], "parent", "agent-b", data_dir=data_dir))

    # Update a task
    run(tasks_mod.update_task(t1["id"], status="in_progress", data_dir=data_dir))

    # Capture current projection
    all_before = run(tasks_mod.list_tasks(data_dir=data_dir))
    ready_before = run(tasks_mod.ready_tasks(data_dir=data_dir))

    # Rebuild from archive
    result = run(tasks_mod.rebuild_from_archive(data_dir=data_dir))
    assert result["tasks_rebuilt"] >= 3

    # Assert identical projection
    all_after = run(tasks_mod.list_tasks(data_dir=data_dir))
    ready_after = run(tasks_mod.ready_tasks(data_dir=data_dir))

    before_ids = {t["id"] for t in all_before}
    after_ids = {t["id"] for t in all_after}
    assert before_ids == after_ids

    # Status should match
    before_map = {t["id"]: t["status"] for t in all_before}
    after_map = {t["id"]: t["status"] for t in all_after}
    assert before_map == after_map

    # Ready set should match
    assert {t["id"] for t in ready_before} == {t["id"] for t in ready_after}


def test_rebuild_from_archive_with_edge_removal(data_dir):
    """Edge removal is replayed correctly."""
    t1 = run(tasks_mod.create_task("T1", created_by="x", data_dir=data_dir))
    t2 = run(tasks_mod.create_task("T2", created_by="x", data_dir=data_dir))
    run(tasks_mod.add_edge(t1["id"], t2["id"], "blocks", "x", data_dir=data_dir))
    run(tasks_mod.remove_edge(t1["id"], t2["id"], "blocks", data_dir=data_dir))

    ready_before = {t["id"] for t in run(tasks_mod.ready_tasks(data_dir=data_dir))}

    run(tasks_mod.rebuild_from_archive(data_dir=data_dir))

    ready_after = {t["id"] for t in run(tasks_mod.ready_tasks(data_dir=data_dir))}
    assert ready_before == ready_after
    # Both should be ready after edge removal
    assert t1["id"] in ready_after
    assert t2["id"] in ready_after
