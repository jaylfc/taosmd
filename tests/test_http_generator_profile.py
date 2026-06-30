"""Tests for GET/POST /generator-profile endpoints."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server
from taosmd import config, agents


# ---------------------------------------------------------------------------
# Transport helpers (mirrors tests/test_http_server.py)
# ---------------------------------------------------------------------------

def _post(url: str, payload) -> tuple[int, dict]:
    data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    return _send(req)


def _get(url: str) -> tuple[int, dict]:
    return _send(urllib.request.Request(url, method="GET"))


def _send(req) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# Local fixture that yields (base_url, data_dir) so tests can set state
# in the same data dir the server uses.
# ---------------------------------------------------------------------------

@pytest.fixture
def live_server_with_dir(tmp_path, monkeypatch):
    """Start the HTTP server and yield (base_url, data_dir)."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    try:
        yield base_url, data_dir
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
# Tests
# ---------------------------------------------------------------------------

def test_get_lists_profiles_and_global_active(live_server_with_dir):
    base_url, data_dir = live_server_with_dir
    status, body = _get(f"{base_url}/generator-profile")
    assert status == 200, body
    ids = {p["id"] for p in body["profiles"]}
    assert {"balanced", "factual-recall"} <= ids
    assert body["active"] == "balanced"
    assert body["scope"] == "global"


def test_get_reflects_global_set(live_server_with_dir):
    base_url, data_dir = live_server_with_dir
    config.set_generator_profile("factual-recall", data_dir=data_dir)
    status, body = _get(f"{base_url}/generator-profile")
    assert status == 200, body
    assert body["active"] == "factual-recall"


def test_get_reflects_per_agent(live_server_with_dir):
    base_url, data_dir = live_server_with_dir
    agents.AgentRegistry(data_dir).register_agent("alice")
    agents.set_agent_generator_profile("alice", "factual-recall", data_dir=data_dir)
    status, body = _get(f"{base_url}/generator-profile?agent=alice")
    assert status == 200, body
    assert body["active"] == "factual-recall"
    assert body["scope"] == "alice"


def test_post_sets_global(live_server_with_dir):
    base_url, data_dir = live_server_with_dir
    status, body = _post(f"{base_url}/generator-profile", {"profile_id": "factual-recall"})
    assert status == 200, body
    assert config.get_generator_profile(data_dir=data_dir) == "factual-recall"
    assert body["active"] == "factual-recall"


def test_post_sets_per_agent(live_server_with_dir):
    base_url, data_dir = live_server_with_dir
    agents.AgentRegistry(data_dir).register_agent("bob")
    status, body = _post(
        f"{base_url}/generator-profile",
        {"profile_id": "factual-recall", "agent": "bob"},
    )
    assert status == 200, body
    assert agents.get_agent_generator_profile("bob", data_dir=data_dir) == "factual-recall"


def test_post_unknown_profile_400(live_server_with_dir):
    base_url, data_dir = live_server_with_dir
    status, body = _post(f"{base_url}/generator-profile", {"profile_id": "nope"})
    assert status == 400, body
    assert config.get_generator_profile(data_dir=data_dir) is None


def test_get_unknown_agent_falls_back_to_global(live_server_with_dir):
    """GET /generator-profile?agent=<unregistered> must return 200 with global scope."""
    base_url, data_dir = live_server_with_dir
    status, body = _get(f"{base_url}/generator-profile?agent=nonexistent-agent-xyz")
    assert status == 200, body
    assert body["scope"] == "global"
    assert body["active"] == "balanced"
