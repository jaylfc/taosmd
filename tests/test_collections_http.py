"""HTTP surface tests for collections.

Contract (docs/specs/codebase-indexing-collections-design.md + the Jul 19
decisions): POST /collections and POST /collections/{id}/index are
admin-token-gated (fail closed); list/get/link/unlink/grants and search are
data-plane. DELETE /collections/{id} archives (admin). Indexing is async:
202 then poll GET /collections/{id} until status is ready|error. Search
gains collection/collections params with per-agent grant enforcement.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import config as taosmd_config
from taosmd import http_server

_TOKEN = "test-admin-token-abc123"


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _req(method: str, url: str, payload=None, token: str | None = None):
    headers = {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "readme.md").write_text("# Widget\n\nThe widget frobnicates the sprocket.")
    (d / "guide.txt").write_text("Install with pip and enjoy.")
    return d


@pytest.fixture
def live_server(tmp_path, monkeypatch, source_dir):
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    monkeypatch.setenv("TAOSMD_TOKEN", _TOKEN)
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(data_dir))
    monkeypatch.delenv("TAOSMD_COLLECTIONS_ALLOWED_ROOTS", raising=False)
    taosmd_config.set_collections_allowed_roots(
        [str(source_dir)], data_dir=str(data_dir)
    )

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}", str(data_dir), source_dir
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


@pytest.fixture
def live_server_no_token(tmp_path, monkeypatch, source_dir):
    data_dir = tmp_path / "taosmd-data-nt"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    monkeypatch.delenv("TAOSMD_TOKEN", raising=False)
    monkeypatch.delenv("TAOSMD_ADMIN_TOKEN", raising=False)
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


def _create(base, source_dir, name="repo docs"):
    status, body = _req(
        "POST", f"{base}/collections",
        {"name": name, "kind": "docs", "source_path": str(source_dir)},
        token=_TOKEN,
    )
    assert status == 200, body
    return body["collection"]


def _wait_ready(base, cid, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body = _req("GET", f"{base}/collections/{cid}", token=_TOKEN)
        assert status == 200
        if body["collection"]["status"] in ("ready", "error"):
            return body["collection"]
        time.sleep(0.05)
    raise AssertionError("collection never left 'indexing'")


# ---------------------------------------------------------------------------
# Admin gating
# ---------------------------------------------------------------------------

def test_create_fails_closed_without_any_token(live_server_no_token):
    status, body = _req(
        "POST", f"{live_server_no_token}/collections",
        {"name": "x", "kind": "docs", "source_path": "/tmp"},
    )
    assert status == 403


def test_create_requires_admin_token(live_server):
    base, _, source_dir = live_server
    status, _ = _req(
        "POST", f"{base}/collections",
        {"name": "x", "kind": "docs", "source_path": str(source_dir)},
    )
    assert status == 401
    status, _ = _req(
        "POST", f"{base}/collections",
        {"name": "x", "kind": "docs", "source_path": str(source_dir)},
        token="wrong-token",
    )
    assert status == 401


def test_index_requires_admin_token(live_server):
    base, _, source_dir = live_server
    col = _create(base, source_dir)
    status, _ = _req("POST", f"{base}/collections/{col['id']}/index")
    assert status == 401


def test_delete_requires_admin_token(live_server):
    base, _, source_dir = live_server
    col = _create(base, source_dir)
    status, _ = _req("DELETE", f"{base}/collections/{col['id']}")
    assert status == 401


# ---------------------------------------------------------------------------
# Create / list / get
# ---------------------------------------------------------------------------

def test_create_and_get(live_server):
    base, _, source_dir = live_server
    col = _create(base, source_dir)
    assert col["status"] == "created"
    assert col["kind"] == "docs"
    status, body = _req("GET", f"{base}/collections/{col['id']}", token=_TOKEN)
    assert status == 200
    assert body["collection"]["name"] == "repo docs"
    assert body["collection"]["embedder"] is None


def test_create_rejects_disallowed_path(live_server, tmp_path):
    base, _, _ = live_server
    outside = tmp_path / "definitely-elsewhere"
    outside.mkdir()
    status, body = _req(
        "POST", f"{base}/collections",
        {"name": "x", "kind": "docs", "source_path": str(outside)},
        token=_TOKEN,
    )
    assert status == 400
    assert "allowed root" in body["error"]


def test_get_unknown_404(live_server):
    base, _, _ = live_server
    status, _ = _req("GET", f"{base}/collections/col-000000000000", token=_TOKEN)
    assert status == 404


def test_list_and_project_filter(live_server):
    base, _, source_dir = live_server
    a = _create(base, source_dir, name="a")
    b = _create(base, source_dir, name="b")
    status, body = _req(
        "POST", f"{base}/collections/{a['id']}/link",
        {"type": "taos", "id": "prj-123"}, token=_TOKEN,
    )
    assert status == 200
    status, body = _req("GET", f"{base}/collections", token=_TOKEN)
    assert status == 200
    assert {c["id"] for c in body["collections"]} == {a["id"], b["id"]}
    status, body = _req("GET", f"{base}/collections?project=prj-123", token=_TOKEN)
    assert [c["id"] for c in body["collections"]] == [a["id"]]


# ---------------------------------------------------------------------------
# Link / unlink / grants
# ---------------------------------------------------------------------------

def test_link_unlink(live_server):
    base, _, source_dir = live_server
    col = _create(base, source_dir)
    status, body = _req(
        "POST", f"{base}/collections/{col['id']}/link",
        {"type": "git", "id": "abc123def456"}, token=_TOKEN,
    )
    assert status == 200
    assert {"type": "git", "id": "abc123def456"} in body["collection"]["links"]
    status, body = _req(
        "POST", f"{base}/collections/{col['id']}/unlink",
        {"type": "git", "id": "abc123def456"}, token=_TOKEN,
    )
    assert status == 200
    assert body["collection"]["links"] == []


def test_link_bad_type_400(live_server):
    base, _, source_dir = live_server
    col = _create(base, source_dir)
    status, _ = _req(
        "POST", f"{base}/collections/{col['id']}/link",
        {"type": "jira", "id": "X-1"}, token=_TOKEN,
    )
    assert status == 400


def test_grants_add_and_revoke(live_server):
    base, _, source_dir = live_server
    col = _create(base, source_dir)
    status, body = _req(
        "POST", f"{base}/collections/{col['id']}/grants",
        {"agent": "dev"}, token=_TOKEN,
    )
    assert status == 200
    assert "dev" in body["collection"]["grants"]
    status, body = _req(
        "DELETE", f"{base}/collections/{col['id']}/grants/dev", token=_TOKEN,
    )
    assert status == 200
    assert body["collection"]["grants"] == []


# ---------------------------------------------------------------------------
# Index (async, 202 + poll) and search integration
# ---------------------------------------------------------------------------

def test_index_flow_and_search_grant_enforcement(live_server):
    base, _, source_dir = live_server
    col = _create(base, source_dir)
    status, body = _req(
        "POST", f"{base}/collections/{col['id']}/index", token=_TOKEN,
    )
    assert status == 202
    assert body["status"] == "indexing"
    assert body["job"] == col["id"]

    ready = _wait_ready(base, col["id"])
    assert ready["status"] == "ready"
    assert ready["stats"]["files_indexed"] == 2
    assert ready["last_indexed"] is not None

    # No grant: the collection contributes nothing.
    status, body = _req(
        "POST", f"{base}/search",
        {"query": "widget frobnicates", "agent": "dev", "mode": "bm25",
         "collections": [col["id"]], "collections_only": True},
        token=_TOKEN,
    )
    assert status == 200
    assert body["hits"] == []

    status, _ = _req(
        "POST", f"{base}/collections/{col['id']}/grants",
        {"agent": "dev"}, token=_TOKEN,
    )
    assert status == 200
    status, body = _req(
        "POST", f"{base}/search",
        {"query": "widget frobnicates", "agent": "dev", "mode": "bm25",
         "collections": [col["id"]], "collections_only": True},
        token=_TOKEN,
    )
    assert status == 200
    assert body["hits"]
    top = body["hits"][0]
    assert top["metadata"]["file_path"] == "readme.md"
    assert top["metadata"]["collection_id"] == col["id"]

    # Singular form works too (GET-style single collection).
    status, body = _req(
        "POST", f"{base}/search",
        {"query": "widget frobnicates", "agent": "dev", "mode": "bm25",
         "collection": col["id"], "collections_only": True},
        token=_TOKEN,
    )
    assert status == 200
    assert body["hits"]


def test_index_unknown_404(live_server):
    base, _, _ = live_server
    status, _ = _req(
        "POST", f"{base}/collections/col-000000000000/index", token=_TOKEN,
    )
    assert status == 404


# ---------------------------------------------------------------------------
# Delete (archive)
# ---------------------------------------------------------------------------

def test_delete_archives_reversibly(live_server):
    base, _, source_dir = live_server
    col = _create(base, source_dir)
    status, body = _req("DELETE", f"{base}/collections/{col['id']}", token=_TOKEN)
    assert status == 200
    assert body["collection"]["status"] == "archived"
    # Hidden from the default listing, still fetchable by id (nothing destroyed).
    status, body = _req("GET", f"{base}/collections", token=_TOKEN)
    assert body["collections"] == []
    status, body = _req("GET", f"{base}/collections/{col['id']}", token=_TOKEN)
    assert status == 200
    assert body["collection"]["status"] == "archived"
