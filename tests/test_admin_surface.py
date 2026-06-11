"""Tests for the admin surface: shelf lifecycle and A2A channel admin.

The admin endpoints require a configured server token and FAIL CLOSED when
none is set (403). These tests exercise both the no-token-403 path and the
correct-token happy path.

Uses the live_server fixture pattern from tests/test_http_server.py with a
variant that sets a server token via env or config.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from taosmd import api as taosmd_api
from taosmd import config as taosmd_config
from taosmd import http_server


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_TOKEN = "test-admin-token-abc123"


def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder; same as test_http_server.py."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _post(url: str, payload, token: str | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return _send(req)


def _get(url: str, token: str | None = None) -> tuple[int, dict]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return _send(urllib.request.Request(url, headers=headers, method="GET"))


def _send(req) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# Fixtures: server with token, server without token
# ---------------------------------------------------------------------------

def _make_token_server(tmp_path, monkeypatch):
    """Helper that creates a token-gated server and returns (url, httpd, data_dir)."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    monkeypatch.setenv("TAOSMD_TOKEN", _TOKEN)

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f"http://{host}:{port}", httpd, str(data_dir), thread


def _teardown_server(httpd, thread):
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)
    for s in list(taosmd_api._stores_cache.values()):
        for store in (s.get("archive"), s.get("vector"), s.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    httpd.service_loop.run(store.close())
                except Exception:
                    pass
    httpd.service_loop.close()


@pytest.fixture
def live_server_with_token(tmp_path, monkeypatch):
    """Live server with a configured bearer token for admin tests."""
    base, httpd, data_dir, thread = _make_token_server(tmp_path, monkeypatch)
    try:
        yield base
    finally:
        _teardown_server(httpd, thread)


@pytest.fixture
def live_server_with_token_and_loop(tmp_path, monkeypatch):
    """Live server with token that also yields the httpd (for service loop access)."""
    base, httpd, data_dir, thread = _make_token_server(tmp_path, monkeypatch)
    try:
        yield base, httpd
    finally:
        _teardown_server(httpd, thread)


@pytest.fixture
def live_server_no_token(tmp_path, monkeypatch):
    """Live server with NO configured token (admin endpoints should return 403)."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    monkeypatch.delenv("TAOSMD_TOKEN", raising=False)

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Auth rule: no-token-configured returns 403 on every admin route
# ---------------------------------------------------------------------------

_ADMIN_ROUTES = [
    ("/shelves", {"shelf_id": "testshelf"}),
    ("/shelves/testshelf/archive", {}),
    ("/shelves/testshelf/unarchive", {}),
    ("/a2a/admin/delete-channel", {"channel": "general"}),
    ("/a2a/admin/rename-channel", {"from": "old", "to": "new"}),
    ("/a2a/admin/supersede-message", {"id": 1}),
]


@pytest.mark.parametrize("path,payload", _ADMIN_ROUTES)
def test_no_token_configured_returns_403(live_server_no_token, path, payload):
    """When no server token is configured all admin endpoints return 403."""
    status, body = _post(f"{live_server_no_token}{path}", payload, token=None)
    assert status == 403, f"expected 403 for {path}, got {status}: {body}"
    assert "admin surface requires a configured server token" in body.get("error", "")


@pytest.mark.parametrize("path,payload", _ADMIN_ROUTES)
def test_wrong_token_returns_401(live_server_with_token, path, payload):
    """When a token IS configured, a wrong token returns 401."""
    status, body = _post(
        f"{live_server_with_token}{path}", payload, token="wrong-token"
    )
    assert status == 401, f"expected 401 for {path} with wrong token, got {status}: {body}"


# ---------------------------------------------------------------------------
# Shelf create: idempotence
# ---------------------------------------------------------------------------

def test_shelf_create_new(live_server_with_token):
    status, body = _post(
        f"{live_server_with_token}/shelves",
        {"shelf_id": "myshelf", "display_name": "My Shelf"},
        token=_TOKEN,
    )
    assert status == 200, body
    assert body["created"] is True
    assert body["shelf"]["name"] == "myshelf"
    assert body["shelf"]["display_name"] == "My Shelf"


def test_shelf_create_idempotent(live_server_with_token):
    _post(
        f"{live_server_with_token}/shelves",
        {"shelf_id": "myshelf2"},
        token=_TOKEN,
    )
    status, body = _post(
        f"{live_server_with_token}/shelves",
        {"shelf_id": "myshelf2"},
        token=_TOKEN,
    )
    assert status == 200, body
    assert body["created"] is False
    assert body["shelf"]["name"] == "myshelf2"


def test_shelf_create_with_project_id(live_server_with_token):
    status, body = _post(
        f"{live_server_with_token}/shelves",
        {"shelf_id": "proj-shelf", "project_id": "proj-xyz"},
        token=_TOKEN,
    )
    assert status == 200, body
    assert body["created"] is True
    assert body["shelf"].get("project_id") == "proj-xyz"


def test_shelf_create_invalid_id_returns_400(live_server_with_token):
    status, body = _post(
        f"{live_server_with_token}/shelves",
        {"shelf_id": "InvalidCaps"},
        token=_TOKEN,
    )
    assert status == 400, body
    assert "shelf_id" in body.get("error", "").lower() or "must match" in body.get("error", "")


def test_shelf_create_missing_id_returns_400(live_server_with_token):
    status, body = _post(
        f"{live_server_with_token}/shelves",
        {},
        token=_TOKEN,
    )
    assert status == 400
    assert "shelf_id" in body.get("error", "")


# ---------------------------------------------------------------------------
# Shelf archive / unarchive: hides and restores rows
# ---------------------------------------------------------------------------

def _ingest(base_url: str, text: str, agent: str, token: str) -> None:
    status, body = _post(
        f"{base_url}/ingest",
        {"text": text, "agent": agent},
        token=token,
    )
    assert status == 200, f"ingest failed: {body}"


def _search(base_url: str, query: str, agent: str, token: str) -> list:
    # Use mode=bm25 so we only hit the vector store BM25 index and do not get
    # archive FTS results, which come from the zero-loss archive and are NOT
    # suppressed by shelf-archive (the spec only hides vector rows).
    status, body = _post(
        f"{base_url}/search",
        {"query": query, "agent": agent, "limit": 10, "mode": "bm25"},
        token=token,
    )
    assert status == 200, f"search failed: {body}"
    return body.get("hits", [])


def test_archive_hides_rows_from_search(live_server_with_token):
    base = live_server_with_token
    shelf = "archivetest"

    # Create shelf and ingest some content
    _post(f"{base}/shelves", {"shelf_id": shelf}, token=_TOKEN)
    _ingest(base, "Unique text for archive test alpha", shelf, _TOKEN)
    _ingest(base, "Another unique text for archive test beta", shelf, _TOKEN)

    # Rows should be visible before archive
    hits_before = _search(base, "unique text for archive test", shelf, _TOKEN)
    assert hits_before, "expected hits before archive"

    # Archive the shelf
    status, body = _post(f"{base}/shelves/{shelf}/archive", {}, token=_TOKEN)
    assert status == 200, body
    assert body["archived"] is True
    assert body["rows_hidden"] >= 2

    # Rows should now be hidden from search
    hits_after = _search(base, "unique text for archive test", shelf, _TOKEN)
    assert not hits_after, f"expected no hits after archive, got {hits_after}"


def test_unarchive_restores_rows(live_server_with_token):
    base = live_server_with_token
    shelf = "restoretest"

    _post(f"{base}/shelves", {"shelf_id": shelf}, token=_TOKEN)
    _ingest(base, "Restore test content gamma", shelf, _TOKEN)

    # Archive
    status, body = _post(f"{base}/shelves/{shelf}/archive", {}, token=_TOKEN)
    assert status == 200, body
    assert body["rows_hidden"] >= 1

    # Confirm hidden
    hits = _search(base, "Restore test content gamma", shelf, _TOKEN)
    assert not hits, "expected no hits after archive"

    # Unarchive
    status, body = _post(f"{base}/shelves/{shelf}/unarchive", {}, token=_TOKEN)
    assert status == 200, body
    assert body["archived"] is False
    assert body["rows_restored"] >= 1

    # Rows should be visible again
    hits = _search(base, "Restore test content gamma", shelf, _TOKEN)
    assert hits, "expected hits after unarchive"
    assert "Restore test content gamma" in hits[0]["text"]


def test_unarchive_does_not_resurrect_non_shelf_superseded_rows(
    live_server_with_token_and_loop,
):
    """Rows superseded for non-archive reasons are not restored by unarchive.

    Strategy: create two rows, manually supersede one via the server's service
    loop (simulating a correction-supersede), then archive+unarchive the shelf
    and verify only the shelf-archived row is restored.
    """
    base, httpd = live_server_with_token_and_loop
    shelf = "supersede-test"

    _post(f"{base}/shelves", {"shelf_id": shelf}, token=_TOKEN)
    _ingest(base, "Row superseded for contradiction reason", shelf, _TOKEN)
    _ingest(base, "Row to be shelf-archived delta", shelf, _TOKEN)

    # Use the server's own service loop to supersede the first row, simulating
    # a correction-supersede (not a shelf-archive). The service loop owns the
    # SQLite connection so we must run async operations there.
    stores_cache = taosmd_api._stores_cache
    data_dir_key = next(iter(stores_cache))
    stores = stores_cache[data_dir_key]
    vmem = stores["vector"]

    async def _do_supersede():
        await vmem.supersede_matching("Row superseded for contradiction reason")

    httpd.service_loop.run(_do_supersede())

    # Now archive the shelf
    status, body = _post(f"{base}/shelves/{shelf}/archive", {}, token=_TOKEN)
    assert status == 200, body
    rows_hidden_by_archive = body["rows_hidden"]
    # Only the "delta" row should have been hidden (the other was already superseded)
    assert rows_hidden_by_archive == 1

    # Unarchive
    status, body = _post(f"{base}/shelves/{shelf}/unarchive", {}, token=_TOKEN)
    assert status == 200, body
    # Only the shelf-archive-hidden row should be restored
    assert body["rows_restored"] == 1

    # The "contradiction" row should remain hidden (it was superseded, not shelf-archived).
    # Check by text rather than relying on BM25 not returning the other row.
    hits = _search(base, "Row to be shelf-archived delta", shelf, _TOKEN)
    hit_texts = [h["text"] for h in hits]
    assert "Row superseded for contradiction reason" not in hit_texts, (
        "contradiction-superseded row should remain hidden after unarchive"
    )

    # The "delta" row should now be visible
    assert any("delta" in t for t in hit_texts), (
        f"shelf-archived row should be restored after unarchive; hits: {hit_texts}"
    )


def test_archive_no_op_when_already_archived(live_server_with_token):
    base = live_server_with_token
    shelf = "noop-archive"
    _post(f"{base}/shelves", {"shelf_id": shelf}, token=_TOKEN)
    _post(f"{base}/shelves/{shelf}/archive", {}, token=_TOKEN)

    # Second archive is a no-op
    status, body = _post(f"{base}/shelves/{shelf}/archive", {}, token=_TOKEN)
    assert status == 200, body
    assert body["archived"] is True
    assert body["rows_hidden"] == 0


def test_archive_expect_empty_409_when_rows_exist(live_server_with_token):
    base = live_server_with_token
    shelf = "expect-empty-test"
    _post(f"{base}/shelves", {"shelf_id": shelf}, token=_TOKEN)
    _ingest(base, "Content that prevents empty archive", shelf, _TOKEN)

    status, body = _post(
        f"{base}/shelves/{shelf}/archive?expect_empty=true", {}, token=_TOKEN
    )
    assert status == 409, body
    assert "active row" in body.get("error", "").lower() or "expect_empty" in body.get("error", "").lower()


def test_archive_expect_empty_succeeds_when_no_rows(live_server_with_token):
    base = live_server_with_token
    shelf = "empty-shelf-test"
    _post(f"{base}/shelves", {"shelf_id": shelf}, token=_TOKEN)

    status, body = _post(
        f"{base}/shelves/{shelf}/archive?expect_empty=true", {}, token=_TOKEN
    )
    assert status == 200, body
    assert body["archived"] is True
    assert body["rows_hidden"] == 0


# ---------------------------------------------------------------------------
# A2A admin: delete channel
# ---------------------------------------------------------------------------

def test_a2a_delete_channel_hides_from_channels(live_server_with_token):
    base = live_server_with_token

    # Send a message to a channel
    _post(f"{base}/a2a/send", {"from": "alice", "body": "hello", "thread": "to-delete"}, token=_TOKEN)

    # Confirm it's visible
    status, body = _get(f"{base}/a2a/channels", token=_TOKEN)
    assert status == 200
    channels = {c["channel"] for c in body["channels"]}
    assert "to-delete" in channels

    # Delete the channel
    status, body = _post(
        f"{base}/a2a/admin/delete-channel",
        {"channel": "to-delete"},
        token=_TOKEN,
    )
    assert status == 200, body
    assert body["deleted"] is True

    # Channel should no longer appear in /a2a/channels
    status, body = _get(f"{base}/a2a/channels", token=_TOKEN)
    assert status == 200
    channels_after = {c["channel"] for c in body["channels"]}
    assert "to-delete" not in channels_after


def test_a2a_delete_channel_hides_from_messages(live_server_with_token):
    base = live_server_with_token

    _post(f"{base}/a2a/send", {"from": "bob", "body": "secret message", "thread": "secret-chan"}, token=_TOKEN)
    _post(f"{base}/a2a/admin/delete-channel", {"channel": "secret-chan"}, token=_TOKEN)

    # Messages on the deleted channel should not appear
    status, body = _get(f"{base}/a2a/messages?thread=secret-chan", token=_TOKEN)
    assert status == 200
    assert body.get("messages") == [] or not any(
        m.get("thread") == "secret-chan" for m in body.get("messages", [])
    )


# ---------------------------------------------------------------------------
# A2A admin: rename channel
# ---------------------------------------------------------------------------

def test_a2a_rename_redirects_sends(live_server_with_token):
    base = live_server_with_token

    # Send some old-name history
    _post(f"{base}/a2a/send", {"from": "alice", "body": "old msg", "thread": "old-name"}, token=_TOKEN)

    # Rename old-name -> new-name
    status, body = _post(
        f"{base}/a2a/admin/rename-channel",
        {"from": "old-name", "to": "new-name"},
        token=_TOKEN,
    )
    assert status == 200, body
    assert body["renamed"] is True

    # Send a new message to old-name; it should be redirected to new-name
    _post(f"{base}/a2a/send", {"from": "bob", "body": "redirected msg", "thread": "old-name"}, token=_TOKEN)

    # new-name should contain both the old history and the redirected message
    status, body = _get(f"{base}/a2a/messages?thread=new-name", token=_TOKEN)
    assert status == 200
    texts = [m.get("body") for m in body.get("messages", [])]
    assert "old msg" in texts, f"old history should appear in new-name: {texts}"
    assert "redirected msg" in texts, f"redirected send should appear in new-name: {texts}"


def test_a2a_rename_merges_old_history_into_new(live_server_with_token):
    """Reading the new channel name includes messages originally in the old name."""
    base = live_server_with_token

    _post(f"{base}/a2a/send", {"from": "x", "body": "history item", "thread": "alpha"}, token=_TOKEN)

    _post(f"{base}/a2a/admin/rename-channel", {"from": "alpha", "to": "beta"}, token=_TOKEN)

    status, body = _get(f"{base}/a2a/messages?thread=beta", token=_TOKEN)
    assert status == 200
    texts = [m.get("body") for m in body.get("messages", [])]
    assert "history item" in texts


# ---------------------------------------------------------------------------
# A2A admin: supersede message
# ---------------------------------------------------------------------------

def test_a2a_supersede_message_hides_it(live_server_with_token):
    base = live_server_with_token

    status, receipt = _post(
        f"{base}/a2a/send",
        {"from": "alice", "body": "message to suppress", "thread": "gen"},
        token=_TOKEN,
    )
    assert status == 200
    msg_id = receipt["id"]

    # Confirm it's visible
    status, body = _get(f"{base}/a2a/messages?thread=gen", token=_TOKEN)
    ids_before = [m["id"] for m in body.get("messages", [])]
    assert msg_id in ids_before

    # Supersede it
    status, body = _post(
        f"{base}/a2a/admin/supersede-message",
        {"id": msg_id},
        token=_TOKEN,
    )
    assert status == 200, body
    assert body["superseded"] is True
    assert body["id"] == msg_id

    # It should no longer appear in the feed
    status, body = _get(f"{base}/a2a/messages?thread=gen", token=_TOKEN)
    ids_after = [m["id"] for m in body.get("messages", [])]
    assert msg_id not in ids_after


def test_rename_then_delete_old_name_keeps_history_under_new(live_server_with_token):
    """The migrate-and-delete end state: after renaming old to new and then
    deleting the old name, the old name reads empty (delisted) while the new
    name still surfaces the old history. The zero-loss archive is never
    mutated; only the alias plus deleted sets change."""
    base = live_server_with_token

    _post(f"{base}/a2a/send", {"from": "x", "body": "carried item", "thread": "oldc"}, token=_TOKEN)
    _post(f"{base}/a2a/admin/rename-channel", {"from": "oldc", "to": "newc"}, token=_TOKEN)
    _post(f"{base}/a2a/admin/delete-channel", {"channel": "oldc"}, token=_TOKEN)

    # New name keeps the history.
    status, body = _get(f"{base}/a2a/messages?thread=newc", token=_TOKEN)
    assert status == 200
    assert "carried item" in [m.get("body") for m in body.get("messages", [])]

    # Old name is delisted: direct read returns nothing.
    status, body = _get(f"{base}/a2a/messages?thread=oldc", token=_TOKEN)
    assert status == 200
    assert body.get("messages", []) == []

    # Old name does not appear in the channel list.
    status, body = _get(f"{base}/a2a/channels", token=_TOKEN)
    assert "oldc" not in [c["channel"] for c in body.get("channels", [])]
