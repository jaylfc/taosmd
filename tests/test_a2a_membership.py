"""Tests for A2A thread membership: store, service layer, and HTTP endpoints.

Covers the four membership service functions added in tsk-a3rwa4:
``a2a_create_thread``, ``a2a_list_members``, ``a2a_add_member``,
``a2a_remove_member``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server, service
from taosmd.a2a_membership import MembershipStore
from taosmd.archive import ArchiveStore, EVENT_A2A


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder -- no ONNX/QMD model required."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Isolated data dir with a clean stores cache for each test."""
    data_dir = tmp_path / "taosmd-a2a-mem"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    yield data_dir
    for s in list(taosmd_api._stores_cache.values()):
        for store in (s.get("archive"), s.get("vector"), s.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass
        mstore = MembershipStore(str(data_dir))
        try:
            asyncio.run(mstore.close())
        except Exception:
            pass


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """HTTP server on an ephemeral port against an isolated data dir."""
    data_dir = tmp_path / "taosmd-a2a-mem-http"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        httpd.service_loop.close()


def _http_post(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _http_delete(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# MembershipStore unit tests
# ---------------------------------------------------------------------------

def test_membership_store_data_dir_none(tmp_path, monkeypatch):
    """MembershipStore tolerates data_dir=None by falling back to the default."""
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    store = MembershipStore(None)
    assert store._path == tmp_path / "a2a-membership.db"
    assert store._path.exists()
    asyncio.run(store.close())


def test_membership_store_creates_db(tmp_path):
    """Store creates the SQLite file and schema on open."""
    data_dir = tmp_path / "store"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='a2a_membership'"
    ).fetchone()
    assert row is not None
    asyncio.run(store.close())


def test_membership_store_migration_applied(tmp_path):
    """The a2a_membership migration runs and stamps user_version."""
    from taosmd import _db, migrations

    data_dir = tmp_path / "mig"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    conn = _db.connect(str(data_dir / "a2a-membership.db"))
    try:
        version = migrations.migrate(conn, "a2a_membership")
        assert version.to_version == migrations.latest_version("a2a_membership")
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == \
            migrations.latest_version("a2a_membership")
    finally:
        conn.close()
        asyncio.run(store.close())


def test_membership_store_add_and_get(tmp_path):
    """add_membership writes a row; get_membership retrieves it."""
    data_dir = tmp_path / "crud"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        mid = asyncio.run(store.add_membership("thread-1", "alice", role="owner"))
        assert mid > 0
        m = asyncio.run(store.get_membership("thread-1", "alice"))
        assert m is not None
        assert m.thread == "thread-1"
        assert m.principal_id == "alice"
        assert m.role == "owner"
        assert m.removed_at is None
    finally:
        asyncio.run(store.close())


def test_membership_store_falsy_timestamp_created(tmp_path):
    """created_at=0 (epoch) is preserved, not replaced with time.time()."""
    data_dir = tmp_path / "ts"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        asyncio.run(store.add_membership("t1", "alice", role="owner", created_at=0))
        m = asyncio.run(store.get_membership("t1", "alice"))
        assert m is not None
        assert m.created_at == 0
    finally:
        asyncio.run(store.close())


def test_membership_store_falsy_timestamp_removed(tmp_path):
    """removed_at=0 (epoch) is preserved, not replaced with time.time()."""
    data_dir = tmp_path / "ts2"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        asyncio.run(store.add_membership("t1", "alice", role="owner", created_at=100))
        result = asyncio.run(store.remove_membership("t1", "alice", removed_at=0))
        assert result is True
        row = store._conn.execute(
            "SELECT removed_at FROM a2a_membership WHERE thread='t1' AND principal_id='alice'"
        ).fetchone()
        assert row["removed_at"] == 0
    finally:
        asyncio.run(store.close())


def test_membership_store_remove_is_zero_loss(tmp_path):
    """remove_membership marks inactive but the row persists."""
    data_dir = tmp_path / "zl"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        asyncio.run(store.add_membership("t1", "alice", role="owner"))
        asyncio.run(store.remove_membership("t1", "alice"))
        m = asyncio.run(store.get_membership("t1", "alice"))
        assert m is None
        has_any = asyncio.run(store.has_any_membership("t1"))
        assert has_any is True
    finally:
        asyncio.run(store.close())


def test_membership_store_list_active_members(tmp_path):
    """list_active_members returns active rows ordered by role then name."""
    data_dir = tmp_path / "list"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        asyncio.run(store.add_membership("t1", "zoe", role="member"))
        asyncio.run(store.add_membership("t1", "alice", role="owner"))
        asyncio.run(store.add_membership("t1", "bob", role="member"))
        members = asyncio.run(store.list_active_members("t1"))
        assert len(members) == 3
        assert members[0].role == "owner"
        assert members[0].principal_id == "alice"
        assert [m.role for m in members] == ["owner", "member", "member"]
    finally:
        asyncio.run(store.close())


def test_membership_store_get_thread_owners(tmp_path):
    """get_thread_owners returns only active owners."""
    data_dir = tmp_path / "owners"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        asyncio.run(store.add_membership("t1", "alice", role="owner"))
        asyncio.run(store.add_membership("t1", "bob", role="member"))
        owners = asyncio.run(store.get_thread_owners("t1"))
        assert len(owners) == 1
        assert owners[0].principal_id == "alice"
    finally:
        asyncio.run(store.close())


def test_membership_store_count_active_members(tmp_path):
    """count_active_members counts only active rows."""
    data_dir = tmp_path / "count"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        asyncio.run(store.add_membership("t1", "alice", role="owner"))
        asyncio.run(store.add_membership("t1", "bob", role="member"))
        asyncio.run(store.add_membership("t1", "carol", role="member"))
        assert asyncio.run(store.count_active_members("t1")) == 3
        asyncio.run(store.remove_membership("t1", "bob"))
        assert asyncio.run(store.count_active_members("t1")) == 2
    finally:
        asyncio.run(store.close())


def test_membership_store_open_thread_no_rows(tmp_path):
    """A thread with no membership rows reports has_any_membership=False."""
    data_dir = tmp_path / "open"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        assert asyncio.run(store.has_any_membership("ghost")) is False
    finally:
        asyncio.run(store.close())


def test_membership_store_re_add_after_removal(tmp_path):
    """Adding a previously removed member reactivates the row (zero-loss)."""
    data_dir = tmp_path / "react"
    data_dir.mkdir()
    store = MembershipStore(str(data_dir))
    try:
        asyncio.run(store.add_membership("t1", "alice", role="owner"))
        asyncio.run(store.remove_membership("t1", "alice"))
        assert asyncio.run(store.has_any_membership("t1")) is True
        assert asyncio.run(store.get_membership("t1", "alice")) is None
        asyncio.run(store.add_membership("t1", "alice", role="member"))
        m = asyncio.run(store.get_membership("t1", "alice"))
        assert m is not None
        assert m.role == "member"
        assert m.removed_at is None
    finally:
        asyncio.run(store.close())


# ---------------------------------------------------------------------------
# Service-layer: a2a_create_thread
# ---------------------------------------------------------------------------

def test_create_thread_creator_is_owner(isolated_data_dir):
    """The caller is added as owner; participants are members."""
    dd = str(isolated_data_dir)
    result = asyncio.run(service.a2a_create_thread(
        "proj-x", ["alice", "bob"], "carol", data_dir=dd,
    ))
    assert result["thread"] == "proj-x"
    assert result["created"] is True
    members = result["active_members"]
    assert len(members) == 3
    owner = next(m for m in members if m["principal_id"] == "carol")
    assert owner["role"] == "owner"
    alice = next(m for m in members if m["principal_id"] == "alice")
    assert alice["role"] == "member"


def test_create_thread_caller_not_in_participants(isolated_data_dir):
    """Caller listed in participants is not duplicated."""
    dd = str(isolated_data_dir)
    result = asyncio.run(service.a2a_create_thread(
        "proj-y", ["alice", "carol"], "carol", data_dir=dd,
    ))
    ids = [m["principal_id"] for m in result["active_members"]]
    assert ids.count("carol") == 1


def test_create_thread_empty_participants_raises(isolated_data_dir):
    """Empty participants list raises ValueError."""
    dd = str(isolated_data_dir)
    with pytest.raises(ValueError, match="participants"):
        asyncio.run(service.a2a_create_thread("t", [], "alice", data_dir=dd))


def test_create_thread_duplicate_raises(isolated_data_dir):
    """Creating a thread that already exists raises ValueError."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice"], "bob", data_dir=dd))
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(service.a2a_create_thread("t1", ["carol"], "dave", data_dir=dd))


# ---------------------------------------------------------------------------
# Service-layer: a2a_list_members
# ---------------------------------------------------------------------------

def test_list_members_returns_owners_first(isolated_data_dir):
    """list_members returns active members with owners first, then members alphabetical."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["zoe", "alice", "bob"], "carol", data_dir=dd))
    members = asyncio.run(service.a2a_list_members("t1", data_dir=dd))
    ids = [m["principal_id"] for m in members]
    # owner (carol) should come first, then members alphabetically
    assert ids[0] == "carol"
    assert ids[1:] == sorted(ids[1:])
    assert set(ids) == {"alice", "bob", "carol", "zoe"}


def test_list_members_open_thread_returns_empty(isolated_data_dir):
    """A thread with no membership rows returns [] (open/legacy)."""
    dd = str(isolated_data_dir)
    members = asyncio.run(service.a2a_list_members("ghost", data_dir=dd))
    assert members == []


# ---------------------------------------------------------------------------
# Service-layer: a2a_add_member
# ---------------------------------------------------------------------------

def test_add_member_by_owner(isolated_data_dir):
    """An owner can add a member to the thread."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice"], "carol", data_dir=dd))
    result = asyncio.run(service.a2a_add_member("t1", "dave", "carol", data_dir=dd))
    assert result["added"] is True
    assert result["principal_id"] == "dave"


def test_add_member_while_add_member(isolated_data_dir):
    """A member cannot add another member."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice", "bob"], "carol", data_dir=dd))
    with pytest.raises(PermissionError, match="not an owner"):
        asyncio.run(service.a2a_add_member("t1", "dave", "alice", data_dir=dd))


def test_add_member_already_member(isolated_data_dir):
    """Adding a principal who is already an active member returns already_member."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice", "bob"], "carol", data_dir=dd))
    result = asyncio.run(service.a2a_add_member("t1", "alice", "carol", data_dir=dd))
    assert result["added"] is False
    assert result["already_member"] is True


# ---------------------------------------------------------------------------
# Service-layer: a2a_remove_member
# ---------------------------------------------------------------------------

def test_remove_member_by_owner(isolated_data_dir):
    """An owner can remove a member."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice", "bob"], "carol", data_dir=dd))
    result = asyncio.run(service.a2a_remove_member("t1", "alice", "carol", data_dir=dd))
    assert result["removed"] is True
    remaining = asyncio.run(service.a2a_list_members("t1", data_dir=dd))
    ids = [m["principal_id"] for m in remaining]
    assert "alice" not in ids
    assert "bob" in ids
    assert "carol" in ids


def test_remove_member_non_owner_raises(isolated_data_dir):
    """A member cannot remove another member."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice", "bob"], "carol", data_dir=dd))
    with pytest.raises(PermissionError, match="not an owner"):
        asyncio.run(service.a2a_remove_member("t1", "bob", "alice", data_dir=dd))


def test_remove_last_owner_raises(isolated_data_dir):
    """Cannot remove the last owner of a thread."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice"], "carol", data_dir=dd))
    with pytest.raises(ValueError, match="last owner"):
        asyncio.run(service.a2a_remove_member("t1", "carol", "carol", data_dir=dd))


def test_remove_not_found_returns_false(isolated_data_dir):
    """Removing a principal who is not a member returns not_found."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice"], "carol", data_dir=dd))
    result = asyncio.run(service.a2a_remove_member("t1", "ghost", "carol", data_dir=dd))
    assert result["removed"] is False
    assert result["not_found"] is True


def test_remove_then_readd_same_role(isolated_data_dir):
    """After removal, the principal can be re-added."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice", "bob"], "carol", data_dir=dd))
    asyncio.run(service.a2a_remove_member("t1", "alice", "carol", data_dir=dd))
    result = asyncio.run(service.a2a_add_member("t1", "alice", "carol", data_dir=dd))
    assert result["added"] is True


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

def test_http_a2a_create_thread(live_server):
    """POST /a2a/threads creates a thread and returns active members."""
    status, body = _http_post(f"{live_server}/a2a/threads",
                              {"thread": "ht-1", "participants": ["alice"], "agent": "carol"})
    assert status == 200, body
    assert body["thread"] == "ht-1"
    assert body["created"] is True
    assert len(body["active_members"]) == 2
    assert any(m["role"] == "owner" for m in body["active_members"])


def test_http_a2a_create_thread_missing_fields(live_server):
    """POST /a2a/threads without required fields returns 400."""
    status, body = _http_post(f"{live_server}/a2a/threads", {"thread": "x", "agent": "y"})
    assert status == 400


def test_http_a2a_list_members(live_server):
    """GET /a2a/threads/{thread}/members returns members."""
    _http_post(f"{live_server}/a2a/threads",
               {"thread": "ht-2", "participants": ["alice", "bob"], "agent": "carol"})
    status, body = _get(f"{live_server}/a2a/threads/ht-2/members")
    assert status == 200, body
    ids = [m["principal_id"] for m in body["members"]]
    assert set(ids) == {"alice", "bob", "carol"}


def test_http_a2a_list_members_open_thread(live_server):
    """GET /a2a/threads/{thread}/members for unknown thread returns empty."""
    status, body = _get(f"{live_server}/a2a/threads/ghost/members")
    assert status == 200, body
    assert body["members"] == []


def test_http_a2a_add_member(live_server):
    """POST /a2a/threads/{thread}/members adds a member."""
    _http_post(f"{live_server}/a2a/threads",
               {"thread": "ht-3", "participants": ["alice"], "agent": "carol"})
    status, body = _http_post(f"{live_server}/a2a/threads/ht-3/members",
                              {"principal_id": "dave", "agent": "carol"})
    assert status == 200, body
    assert body["added"] is True


def test_http_a2a_add_member_non_owner(live_server):
    """POST /a2a/threads/{thread}/members by a non-owner returns 403 (PermissionError)."""
    _http_post(f"{live_server}/a2a/threads",
               {"thread": "ht-4", "participants": ["alice", "bob"], "agent": "carol"})
    status, body = _http_post(f"{live_server}/a2a/threads/ht-4/members",
                              {"principal_id": "dave", "agent": "alice"})
    assert status == 403


def test_http_a2a_remove_member(live_server):
    """DELETE /a2a/threads/{thread}/members/{principal} removes a member."""
    _http_post(f"{live_server}/a2a/threads",
               {"thread": "ht-5", "participants": ["alice", "bob"], "agent": "carol"})
    status, body = _http_delete(
        f"{live_server}/a2a/threads/ht-5/members/alice", {"agent": "carol"},
    )
    assert status == 200, body
    assert body["removed"] is True

    # Verify the member is gone
    status, body = _get(f"{live_server}/a2a/threads/ht-5/members")
    assert status == 200, body
    ids = [m["principal_id"] for m in body["members"]]
    assert "alice" not in ids
    assert "bob" in ids
    assert "carol" in ids


def test_http_a2a_remove_last_owner(live_server):
    """DELETE last owner returns 400 (ValueError -> 400)."""
    _http_post(f"{live_server}/a2a/threads",
               {"thread": "ht-6", "participants": ["alice"], "agent": "carol"})
    status, body = _http_delete(
        f"{live_server}/a2a/threads/ht-6/members/carol", {"agent": "carol"},
    )
    assert status == 400


def test_http_a2a_remove_member_non_owner(live_server):
    """DELETE by a non-owner returns 403 (PermissionError)."""
    _http_post(f"{live_server}/a2a/threads",
               {"thread": "ht-7", "participants": ["alice", "bob"], "agent": "carol"})
    status, body = _http_delete(
        f"{live_server}/a2a/threads/ht-7/members/bob", {"agent": "alice"},
    )
    assert status == 403


def test_archive_event_lands_on_add_and_remove(isolated_data_dir):
    """a2a_add_member and a2a_remove_member must actually record archive
    events, not merely return \"archived\": true."""
    dd = str(isolated_data_dir)
    asyncio.run(service.a2a_create_thread("t1", ["alice", "bob"], "carol", data_dir=dd))
    asyncio.run(service.a2a_add_member("t1", "dave", "carol", data_dir=dd))
    asyncio.run(service.a2a_remove_member("t1", "dave", "carol", data_dir=dd))

    archive = ArchiveStore(
        archive_dir=str(isolated_data_dir / "archive"),
        index_path=str(isolated_data_dir / "archive-index.db"),
    )
    asyncio.run(archive.init())
    try:
        events = asyncio.run(archive.query(event_type=EVENT_A2A, limit=1000))
        actions = []
        for e in events:
            data = json.loads(e.get("data_json", "{}"))
            actions.append(data.get("admin_action"))
        assert "membership_created" in actions
        assert "membership_removed" in actions
    finally:
        asyncio.run(archive.close())
