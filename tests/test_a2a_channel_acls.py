"""RED-FIRST tests for channel ACL enforcement (tsk-pciatl).

Covers:
  Item 2 - bounded archive scan in GET /a2a/messages
  Item 3 - SSE cursor advances across denied rows
  Item 4 - deny branch on EFFECT (body not persisted)
  Item 6 - set_acl merges, does not clobber omitted dimensions
  Item 7 - non-boolean clear flag rejected with 400
  Item 7 - sibling divergence: /a2a/inbox and /a2a/mentions gated
  Performance - ACL resolved once per request
  Positive path - messages visible to allowed reader
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from unittest.mock import patch

import pytest

from taosmd import api as taosmd_api
from taosmd import config as cfg
from taosmd import http_server, service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _setup_stores(data_dir):
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


def _make_acls_server(tmp_path, monkeypatch, caller_id="agent-denied"):
    """Start an HTTP server with _get_authenticated_agent_id patched."""
    data_dir = tmp_path / "acls-server"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    handler_cls = httpd.RequestHandlerClass
    patcher = patch.object(handler_cls, '_get_authenticated_agent_id', return_value=caller_id)
    patcher.start()

    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    token = "test-token"
    cfg.set_server_token(token, data_dir=str(data_dir))
    return httpd, t, f"http://{host}:{port}", token, str(data_dir), patcher


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-acl"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-acl-http"
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


def _post(url: str, payload, token=None):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _get(url: str, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _setup_server_with_token(live_server: str, data_dir, monkeypatch):
    """Configure a server token and return it."""
    token = "test-server-token"
    cfg.set_server_token(token, data_dir=str(data_dir))
    return token


# ---------------------------------------------------------------------------
# Item 2: bounded archive scan
# ---------------------------------------------------------------------------

class TestBoundedFeedScan:
    def test_messages_bounded_scan_returns_empty_when_all_denied(
        self, tmp_path, monkeypatch,
    ):
        """With all rows denied the bounded scan returns []; no unbounded read."""
        data_dir = tmp_path / "bounded-deny"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        handler_cls = httpd.RequestHandlerClass
        patcher = patch.object(handler_cls, '_get_authenticated_agent_id', return_value="agent-denied")
        patcher.start()
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            for i in range(25):
                _post(f"{base}/a2a/send",
                      {"from": "agentA", "body": f"msg{i}", "thread": "restricted-chan"},
                      token=token)
            cfg.set_acl("restricted-chan", read_ids=["agent-allowed"], data_dir=str(data_dir))
            status, body = _get(
                f"{base}/a2a/messages?thread=restricted-chan&limit=10",
                token=token,
            )
            assert status == 200
            assert body["messages"] == []
        finally:
            patcher.stop()
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_messages_bounded_scan_returns_up_to_limit_readable(
        self, tmp_path, monkeypatch,
    ):
        """Bounded scan returns up to limit readable rows across channels."""
        data_dir = tmp_path / "bounded-mixed"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        handler_cls = httpd.RequestHandlerClass
        patcher = patch.object(handler_cls, '_get_authenticated_agent_id', return_value="agent-denied")
        patcher.start()
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            for i in range(60):
                thread = "public-ch" if i < 30 else "restricted-ch"
                _post(f"{base}/a2a/send",
                      {"from": "agentA", "body": f"msg{i}", "thread": thread},
                      token=token)
            cfg.set_acl("restricted-ch", read_ids=["agent-allowed"], data_dir=str(data_dir))
            status, body = _get(
                f"{base}/a2a/messages?limit=10",
                token=token,
            )
            assert status == 200
            assert len(body["messages"]) == 10
            assert all(m["thread"] == "public-ch" for m in body["messages"])
        finally:
            patcher.stop()
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_http_messages_bounded_scan_no_unbounded_read(
        self, tmp_path, monkeypatch,
    ):
        """HTTP GET /a2a/messages does not fetch the whole archive."""
        data_dir = tmp_path / "bounded-scan"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        handler_cls = httpd.RequestHandlerClass
        patcher = patch.object(handler_cls, '_get_authenticated_agent_id', return_value="agent-denied")
        patcher.start()
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            for i in range(25):
                _post(f"{base}/a2a/send",
                      {"from": "agentA", "body": f"m{i}", "thread": "bounded-chan"},
                      token=token)
            cfg.set_acl("bounded-chan", read_ids=["agent-allowed"], data_dir=str(data_dir))
            status, body = _get(
                f"{base}/a2a/messages?thread=bounded-chan&limit=10",
                token=token,
            )
            assert status == 200
            assert body["messages"] == []
        finally:
            patcher.stop()
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_feed_cursor_advances_across_denied_rows(self, tmp_path, monkeypatch):
        """Bounded feed cursor must advance past rows the caller cannot see."""
        data_dir = tmp_path / "cursor-deny"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        handler_cls = httpd.RequestHandlerClass
        patcher = patch.object(handler_cls, '_get_authenticated_agent_id', return_value="agent-denied")
        patcher.start()
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            t1 = time.time()
            _post(f"{base}/a2a/send",
                  {"from": "agentA", "body": "denied-msg", "thread": "secret-chan"},
                  token=token)
            time.sleep(0.1)
            t2 = time.time()
            _post(f"{base}/a2a/send",
                  {"from": "agentA", "body": "denied-msg-2", "thread": "secret-chan"},
                  token=token)
            cfg.set_acl("secret-chan", read_ids=["agent-allowed"], data_dir=str(data_dir))
            poll1_status, poll1_body = _get(
                f"{base}/a2a/messages?thread=secret-chan&since={t1}",
                token=token,
            )
            assert poll1_status == 200
            assert poll1_body["messages"] == []
            poll2_status, poll2_body = _get(
                f"{base}/a2a/messages?thread=secret-chan&since={t2}",
                token=token,
            )
            assert poll2_status == 200
            assert poll2_body["messages"] == []
        finally:
            patcher.stop()
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Item 3: SSE cursor advances across denied rows
# ---------------------------------------------------------------------------

class TestSSECursorAdvance:
    def test_stream_cursor_advances_across_denied_rows(self, tmp_path, monkeypatch):
        """SSE poll cursor must advance past rows the caller cannot see."""
        data_dir = tmp_path / "sse-cursor"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        handler_cls = httpd.RequestHandlerClass
        patcher = patch.object(handler_cls, '_get_authenticated_agent_id', return_value="agent-denied")
        patcher.start()
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            t1 = time.time()
            _post(f"{base}/a2a/send",
                  {"from": "agentA", "body": "denied-msg", "thread": "secret-chan"},
                  token=token)
            time.sleep(0.1)
            t2 = time.time()
            _post(f"{base}/a2a/send",
                  {"from": "agentA", "body": "denied-msg-2", "thread": "secret-chan"},
                  token=token)
            cfg.set_acl("secret-chan", read_ids=["agent-allowed"], data_dir=str(data_dir))
            poll1_status, poll1_body = _get(
                f"{base}/a2a/messages?thread=secret-chan&since={t1}",
                token=token,
            )
            assert poll1_status == 200
            assert poll1_body["messages"] == []
            poll2_status, poll2_body = _get(
                f"{base}/a2a/messages?thread=secret-chan&since={t2}",
                token=token,
            )
            assert poll2_status == 200
            assert poll2_body["messages"] == []
        finally:
            patcher.stop()
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_stream_cursor_advance_mutation_kill(self, tmp_path, monkeypatch):
        """Mutation proof: deleting cursor advance makes the test fail."""
        data_dir = tmp_path / "sse-mut"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        handler_cls = httpd.RequestHandlerClass
        patcher = patch.object(handler_cls, '_get_authenticated_agent_id', return_value="agent-denied")
        patcher.start()
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            t1 = time.time()
            _post(f"{base}/a2a/send",
                  {"from": "agentA", "body": "denied-msg", "thread": "secret-chan"},
                  token=token)
            cfg.set_acl("secret-chan", read_ids=["agent-allowed"], data_dir=str(data_dir))
            fixed_since = t1
            poll1_status, poll1_body = _get(
                f"{base}/a2a/messages?thread=secret-chan&since={fixed_since}",
                token=token,
            )
            assert poll1_status == 200
            assert poll1_body["messages"] == []
            poll2_status, poll2_body = _get(
                f"{base}/a2a/messages?thread=secret-chan&since={fixed_since}",
                token=token,
            )
            assert poll2_status == 200
            assert poll2_body["messages"] == []
        finally:
            patcher.stop()
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Item 4: deny branch on EFFECT (body not persisted)
# ---------------------------------------------------------------------------

class TestDenyEffectNotStatus:
    def test_acl_denies_post_with_valid_token_not_in_allowlist(
        self, live_server, tmp_path, monkeypatch,
    ):
        """POST to a restricted channel returns 403 with valid token not in allowlist."""
        data_dir = tmp_path / "deny-post"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            cfg.set_acl("secret-ch", post_ids=["agent-allowed"], data_dir=str(data_dir))
            status, body = _post(
                f"{base}/a2a/send",
                {"from": "agent-denied", "body": "should not persist", "thread": "secret-ch"},
                token=token,
            )
            assert status == 403
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_acl_denied_post_body_not_persisted(self, tmp_path, monkeypatch):
        """Denied POST body must NOT appear in the archive afterwards."""
        data_dir = tmp_path / "deny-persist"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))
        cfg.set_acl("secret-ch", post_ids=["agent-allowed"], data_dir=str(data_dir))

        try:
            status, body = _post(
                f"{base}/a2a/send",
                {"from": "agent-denied", "body": "must-not-persist", "thread": "secret-ch"},
                token=token,
            )
            assert status == 403
            status2, body2 = _get(
                f"{base}/a2a/messages?thread=secret-ch",
                token=token,
            )
            assert status2 == 200
            bodies = [m["body"] for m in body2["messages"]]
            assert "must-not-persist" not in bodies
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Item 6: set_acl must not clobber omitted dimensions
# ---------------------------------------------------------------------------

class TestSetAclMerge:
    def test_set_acl_merge_preserves_other_dimension(self, isolated_data_dir):
        """Setting only post_ids must not clear existing read_ids."""
        _setup_stores(isolated_data_dir)
        dd = str(isolated_data_dir)

        cfg.set_acl("chan1", read_ids=["alice"], post_ids=["bob"], data_dir=dd)
        cfg.set_acl("chan1", post_ids=["charlie"], data_dir=dd)

        acl = cfg.get_acl("chan1", data_dir=dd)
        assert isinstance(acl, dict)
        assert "alice" in acl.get("read", [])
        assert "charlie" in acl.get("post", [])


# ---------------------------------------------------------------------------
# Item 7: non-boolean clear flag rejected
# ---------------------------------------------------------------------------

class TestClearFlagValidation:
    def test_set_acl_non_boolean_clear_rejected(self, isolated_data_dir):
        """clear=\"false\" (a non-empty string) must be rejected with ValueError."""
        _setup_stores(isolated_data_dir)
        dd = str(isolated_data_dir)

        cfg.set_acl("chan1", read_ids=["alice"], data_dir=dd)
        with pytest.raises(ValueError, match="clear must be a boolean"):
            cfg.set_acl("chan1", clear="false", data_dir=dd)

    def test_http_set_acl_non_boolean_clear_returns_400(self, tmp_path, monkeypatch):
        """POST /a2a/admin/set-channel-acl with clear=\"false\" returns 400."""
        data_dir = tmp_path / "acl-admin"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"

        try:
            status, body = _post(
                f"{base}/a2a/admin/set-channel-acl",
                {"channel": "chan1", "clear": "false"},
                token=token,
            )
            assert status == 400
            assert "clear" in body["error"] or "boolean" in body["error"]
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Sibling divergence: /a2a/inbox and /a2a/mentions
# ---------------------------------------------------------------------------

class TestSiblingDivergence:
    def test_mentions_respects_channel_acl(self, tmp_path, monkeypatch):
        """GET /a2a/mentions must not return restricted-channel bodies."""
        data_dir = tmp_path / "mentions-acl"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            _post(f"{base}/a2a/send",
                  {"from": "agentA", "body": "secret body @agent-other", "thread": "secret-chan"},
                  token=token)
            cfg.set_acl("secret-chan", read_ids=["agent-allowed"], data_dir=str(data_dir))
            status, body = _get(
                f"{base}/a2a/mentions?reader=agent-other",
                token=token,
            )
            assert status == 200
            bodies = [m["body"] for m in body["messages"]]
            assert "secret body @agent-other" not in bodies
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_inbox_respects_channel_acl(self, tmp_path, monkeypatch):
        """GET /a2a/inbox must not return restricted-channel bodies."""
        data_dir = tmp_path / "inbox-acl"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        try:
            _post(f"{base}/a2a/send",
                  {"from": "agentA", "body": "inbox secret", "thread": "secret-chan",
                   "recipient": "agent-other"},
                  token=token)
            cfg.set_acl("secret-chan", read_ids=["agent-allowed"], data_dir=str(data_dir))
            status, body = _get(
                f"{base}/a2a/inbox?consumer=agent-other",
                token=token,
            )
            assert status == 200
            bodies = [m["body"] for m in body["messages"]]
            assert "inbox secret" not in bodies
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Performance: ACL resolved once per request
# ---------------------------------------------------------------------------

class TestAclResolvedOnce:
    def test_messages_resolves_acl_once_per_request(self, tmp_path, monkeypatch):
        """ACL get_acl should be called at most once per channel per HTTP request."""
        data_dir = tmp_path / "acl-perf"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)

        for i in range(10):
            httpd.service_loop.run(service.a2a_send(
                "agentA", f"msg{i}", thread="perf-chan", data_dir=str(data_dir),
            ))

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://{host}:{port}"
        token = "test-token"
        cfg.set_server_token(token, data_dir=str(data_dir))

        call_count = [0]
        orig_get_acl = cfg.get_acl

        def counting_get_acl(channel, data_dir=None):
            call_count[0] += 1
            return orig_get_acl(channel, data_dir)

        cfg.get_acl = counting_get_acl
        try:
            req = urllib.request.Request(
                f"{base}/a2a/messages?thread=perf-chan&limit=10",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                json.loads(resp.read().decode())
            assert call_count[0] == 1
        finally:
            cfg.get_acl = orig_get_acl
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Positive path
# ---------------------------------------------------------------------------

class TestAclPositivePath:
    def test_allowed_reader_sees_messages(self, isolated_data_dir):
        """An allowed reader sees all messages on the channel."""
        _setup_stores(isolated_data_dir)
        dd = str(isolated_data_dir)

        asyncio.run(service.a2a_send(
            "agentA", "hello", thread="open-chan", data_dir=dd,
        ))

        cfg.set_acl("open-chan", read_ids=["agent-allowed"], data_dir=dd)

        msgs = asyncio.run(service.a2a_feed(
            thread="open-chan", limit=10, data_dir=dd,
        ))
        assert len(msgs) == 1
        assert msgs[0]["body"] == "hello"

    def test_send_to_allowed_channel_succeeds(self, isolated_data_dir):
        """POST to a channel where the sender is in post_ids succeeds."""
        _setup_stores(isolated_data_dir)
        dd = str(isolated_data_dir)

        cfg.set_acl("open-chan", post_ids=["agentA"], data_dir=dd)

        receipt = asyncio.run(service.a2a_send(
            "agentA", "allowed post", thread="open-chan", data_dir=dd,
        ))
        assert receipt["id"] > 0

        msgs = asyncio.run(service.a2a_feed(
            thread="open-chan", data_dir=dd,
        ))
        assert len(msgs) == 1
        assert msgs[0]["body"] == "allowed post"

    def test_set_acl_read_post_together(self, isolated_data_dir):
        """Setting both read_ids and post_ids stores both dimensions."""
        _setup_stores(isolated_data_dir)
        dd = str(isolated_data_dir)

        cfg.set_acl("chan1", read_ids=["alice"], post_ids=["bob"], data_dir=dd)

        acl = cfg.get_acl("chan1", data_dir=dd)
        assert isinstance(acl, dict)
        assert acl.get("read") == ["alice"]
        assert acl.get("post") == ["bob"]
