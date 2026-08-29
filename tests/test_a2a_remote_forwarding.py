"""RED proof for Defect 1: service.a2a_* functions forward ``data_dir``
positionally to ``RemoteClient``, but the remote methods declare ``**_opts``
which only captures keyword arguments.  Every one of the four operations raises
``TypeError`` whenever a remote server URL is configured.

These tests drive a recording fake remote and assert both that the call arrives
AND that it carries ``data_dir`` as a keyword argument.  An existing forwarded
function (``a2a_threads``) serves as a positive control and stays green.
"""
from __future__ import annotations

import asyncio

import pytest

from taosmd import service


class RecordingRemote:
    """Minimal stand-in for :class:`~taosmd.remote.RemoteClient`.

    Mirrors the four membership method signatures (each ends in ``**_opts``)
    plus ``a2a_threads`` as a positive control.  Records every call so the
    tests can assert on the received arguments.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def a2a_create_thread(self, thread, participants, agent, **_opts) -> dict:
        self.calls.append(("a2a_create_thread", thread, participants, agent, _opts))
        return {"thread": thread, "created": True, "active_members": []}

    async def a2a_list_members(self, thread, **_opts) -> list[dict]:
        self.calls.append(("a2a_list_members", thread, _opts))
        return []

    async def a2a_add_member(self, thread, principal_id, agent, **_opts) -> dict:
        self.calls.append(("a2a_add_member", thread, principal_id, agent, _opts))
        return {"thread": thread, "principal_id": principal_id, "added": True}

    async def a2a_remove_member(self, thread, principal_id, agent, **_opts) -> dict:
        self.calls.append(("a2a_remove_member", thread, principal_id, agent, _opts))
        return {"thread": thread, "principal_id": principal_id, "removed": True}

    async def a2a_threads(self, *, principal=None, **_opts) -> list[dict]:
        self.calls.append(("a2a_threads", principal, _opts))
        return [{"thread": "t", "kind": "channel", "participants": [],
                 "last_message": {}}]


@pytest.fixture
def patched_remote(monkeypatch):
    """Patch ``_get_remote`` so the service layer always uses our fake remote."""
    remote = RecordingRemote()
    monkeypatch.setattr(service, "_get_remote", lambda data_dir=None: remote)
    return remote


DD = "/fake/data/dir"


def test_a2a_create_thread_forwards_data_dir_by_keyword(patched_remote):
    """``service.a2a_create_thread`` must forward ``data_dir`` as a keyword so
    the remote method (which only accepts ``**_opts`` for extras) receives it."""
    asyncio.run(service.a2a_create_thread(
        "proj-x", ["alice"], "carol", data_dir=DD,
    ))
    assert len(patched_remote.calls) == 1
    name = patched_remote.calls[0][0]
    assert name == "a2a_create_thread"
    opts = patched_remote.calls[0][-1]
    assert opts.get("data_dir") == DD


def test_a2a_list_members_forwards_data_dir_by_keyword(patched_remote):
    """``service.a2a_list_members`` must forward ``data_dir`` as a keyword."""
    asyncio.run(service.a2a_list_members("thread-1", data_dir=DD))
    assert len(patched_remote.calls) == 1
    assert patched_remote.calls[0][0] == "a2a_list_members"
    opts = patched_remote.calls[0][-1]
    assert opts.get("data_dir") == DD


def test_a2a_add_member_forwards_data_dir_by_keyword(patched_remote):
    """``service.a2a_add_member`` must forward ``data_dir`` as a keyword."""
    asyncio.run(service.a2a_add_member("t1", "dave", "carol", data_dir=DD))
    assert len(patched_remote.calls) == 1
    assert patched_remote.calls[0][0] == "a2a_add_member"
    opts = patched_remote.calls[0][-1]
    assert opts.get("data_dir") == DD


def test_a2a_remove_member_forwards_data_dir_by_keyword(patched_remote):
    """``service.a2a_remove_member`` must forward ``data_dir`` as a keyword."""
    asyncio.run(service.a2a_remove_member("t1", "alice", "carol", data_dir=DD))
    assert len(patched_remote.calls) == 1
    assert patched_remote.calls[0][0] == "a2a_remove_member"
    opts = patched_remote.calls[0][-1]
    assert opts.get("data_dir") == DD


def test_a2a_threads_positive_control(patched_remote):
    """``service.a2a_threads`` already forwards ``principal`` by keyword --
    positive control that must stay green while the four membership forwards
    are broken."""
    asyncio.run(service.a2a_threads(principal="carol", data_dir=DD))
    assert len(patched_remote.calls) == 1
    assert patched_remote.calls[0][0] == "a2a_threads"
    principal, opts = patched_remote.calls[0][1], patched_remote.calls[0][2]
    assert principal == "carol"
    assert opts.get("data_dir") == DD
