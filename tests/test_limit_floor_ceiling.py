"""RED-FIRST tests for floorless limit parsing across int(limit) sites.

These tests are written BEFORE the fix. They must FAIL on master (91af00bb)
to prove the bug exists, then PASS after the floor+ceiling fix is applied.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server


def _post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    return _send(req)


def _get(url):
    return _send(urllib.request.Request(url, method="GET"))


def _send(req):
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


class _ServerCtx:
    def __init__(self, url, httpd, stores):
        self.url = url
        self.httpd = httpd
        self.stores = stores

    def run(self, coro):
        return self.httpd.service_loop.run(coro)


@pytest.fixture
def live_server_ctx(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))

    async def _fake_embed(text, task="search_document"):
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    stores["vector"].embed = _fake_embed  # type: ignore[assignment]

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield _ServerCtx(f"http://{host}:{port}", httpd, stores)
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
def live_server(live_server_ctx):
    return live_server_ctx.url


def _seed_memories(ctx, n, agent="user"):
    async def _seed():
        arc = ctx.stores["archive"]
        for i in range(n):
            await arc.record(
                event_type="conversation",
                data={"text": f"seed memory {i}"},
                agent_name=agent,
                summary=f"seed memory {i}",
            )
    ctx.run(_seed())


def _seed_tasks(ctx, n, assignee="a"):
    for i in range(n):
        _post(f"{ctx.url}/tasks", {"title": f"T{i}", "created_by": assignee})


class TestLimitNegativeRejection:
    """All int(limit) sites must reject negative limits with 400."""

    def test_search_negative_limit_rejected(self, live_server):
        status, body = _post(
            f"{live_server}/search",
            {"query": "hello", "agent": "user", "limit": -1},
        )
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_memories_negative_limit_rejected(self, live_server_ctx):
        _seed_memories(live_server_ctx, 10)
        status, body = _get(f"{live_server_ctx.url}/memories?limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_graph_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/graph?limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_graph_activations_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/graph/activations?limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_pending_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/pending?agent=user&limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_a2a_messages_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/a2a/messages?limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_a2a_mentions_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/a2a/mentions?reader=user&limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_a2a_inbox_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/a2a/inbox?consumer=foo&limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_a2a_inbox_unhandled_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/a2a/inbox/unhandled?consumer=foo&limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_a2a_thread_messages_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/a2a/threads/lim/messages?limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_tasks_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/tasks?limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_tasks_ready_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/tasks/ready?limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()

    def test_tasks_edges_negative_limit_rejected(self, live_server):
        status, body = _get(f"{live_server}/tasks/edges?limit=-1")
        assert status == 400, body
        assert "limit" in body["error"].lower()


class TestLimitCeilingEnforcement:
    """Ceiling caps must hold when more rows than the cap are seeded."""

    def test_tasks_limit_capped_at_200(self, live_server_ctx):
        """GET /tasks caps limit at 200 when 300 tasks are seeded."""
        _seed_tasks(live_server_ctx, 300)
        status, body = _get(f"{live_server_ctx.url}/tasks?limit=9999")
        assert status == 200, body
        assert len(body["tasks"]) == 200, (
            f"expected 200 tasks (capped), got {len(body['tasks'])}"
        )

    def test_tasks_ready_limit_capped_at_200(self, live_server_ctx):
        """GET /tasks/ready caps limit at 200 when 300 open tasks are seeded."""
        _seed_tasks(live_server_ctx, 300)
        status, body = _get(f"{live_server_ctx.url}/tasks/ready?limit=9999")
        assert status == 200, body
        assert len(body["tasks"]) == 200, (
            f"expected 200 tasks (capped), got {len(body['tasks'])}"
        )
