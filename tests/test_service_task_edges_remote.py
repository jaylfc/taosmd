"""Service-layer test: project scope survives the remote-path hop for /tasks/edges.

PR #405 (commit 08a33b3d) added GET /tasks/edges and correctly scoped the local
SQL path in ``taosmd.tasks.list_edges``.  But ``service.task_list_edges`` dropped
``project`` when delegating to ``remote.task_list_edges`` (which also lacked the
parameter), so a project-bound token reading via a shared-server deployment could
see sibling-project edges.

This test injects a *recording* fake remote and asserts that ``project``
reaches ``remote.task_list_edges``.  ``service.task_list`` is exercised in the
same test as a positive control: it forwards ``project`` correctly on every
branch, so its assertion passing proves the recording probe itself is sound and
a drop in ``task_list_edges`` cannot report a false green.
"""

from __future__ import annotations

import asyncio

import pytest

from taosmd import service as taosmd_service


class _RecordingRemote:
    """Fake RemoteClient that records every call for later assertion."""

    def __init__(self) -> None:
        self.task_list_calls: list[dict] = []
        self.task_list_edges_calls: list[dict] = []

    async def task_list(self, *, status=None, project=None, assignee=None,
                        limit=50, **_opts) -> list[dict]:
        self.task_list_calls.append(
            {"status": status, "project": project,
             "assignee": assignee, "limit": limit}
        )
        return []

    async def task_list_edges(self, *, from_id=None, to_id=None,
                              edge_type=None, limit=500, project=None, **_opts) -> list[dict]:
        self.task_list_edges_calls.append(
            {"from_id": from_id, "to_id": to_id, "edge_type": edge_type,
             "limit": limit, "project": project}
        )
        return []


@pytest.fixture
def fake_remote(monkeypatch):
    """Inject a recording fake remote into the service layer."""
    fake = _RecordingRemote()
    monkeypatch.setattr(
        taosmd_service, "_get_remote", lambda data_dir: fake
    )
    return fake


def test_task_list_edges_forwards_project_to_remote(fake_remote):
    """service.task_list_edges must forward ``project`` to the remote client.

    Positive control: service.task_list (which already forwards project) is
    exercised in the same test so a broken recording probe cannot report a pass.
    """
    # --- target: task_list_edges must forward project ---
    asyncio.run(taosmd_service.task_list_edges(
        from_id="t-src", to_id="t-dst", edge_type="blocks",
        limit=10, project="proj-a", data_dir="/fake",
    ))
    assert len(fake_remote.task_list_edges_calls) == 1
    assert fake_remote.task_list_edges_calls[0]["project"] == "proj-a"

    # --- positive control: task_list already forwards project ---
    asyncio.run(taosmd_service.task_list(
        status="open", project="proj-a", assignee="agent-1",
        limit=20, data_dir="/fake",
    ))
    assert len(fake_remote.task_list_calls) == 1
    assert fake_remote.task_list_calls[0]["project"] == "proj-a"
