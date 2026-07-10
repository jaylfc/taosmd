"""Config/profile POST writes must run on the single service-loop thread.

Regression guard for the lost-update race (audit M3): the controls and
generator-profile POST handlers used to read-modify-write config.json /
the agent registry directly on the ThreadingHTTPServer request thread,
racing ``ensure_agent`` which does the same read-modify-write on the
service-loop thread during a concurrent /ingest. Routing every mutation
through the service loop serialises them so no write is lost.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server
from taosmd import config, agents

SERVICE_LOOP_THREAD = "taosmd-service-loop"


def _send(req: urllib.request.Request) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _post(url: str, payload) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send(req)


@pytest.fixture
def live_server_with_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    try:
        yield base_url, data_dir, httpd
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


def test_controls_post_write_runs_on_service_loop(live_server_with_dir, monkeypatch):
    base_url, data_dir, _httpd = live_server_with_dir
    seen: list[str] = []
    orig = config._write

    def spy(data, data_dir=None):
        seen.append(threading.current_thread().name)
        return orig(data, data_dir=data_dir)

    monkeypatch.setattr(config, "_write", spy)
    status, body = _post(f"{base_url}/controls", {"values": {"adjacent_turns": 3}})
    assert status == 200, body
    assert seen, "expected config write to run"
    assert all(name == SERVICE_LOOP_THREAD for name in seen), seen


def test_generator_profile_post_registry_write_runs_on_service_loop(
    live_server_with_dir, monkeypatch
):
    base_url, data_dir, _httpd = live_server_with_dir
    agents.AgentRegistry(data_dir).register_agent("alice")
    seen: list[str] = []
    orig = agents.AgentRegistry._write

    def spy(self, data):
        seen.append(threading.current_thread().name)
        return orig(self, data)

    monkeypatch.setattr(agents.AgentRegistry, "_write", spy)
    status, body = _post(
        f"{base_url}/generator-profile",
        {"profile_id": "factual-recall", "agent": "alice"},
    )
    assert status == 200, body
    assert seen, "expected registry write to run"
    assert all(name == SERVICE_LOOP_THREAD for name in seen), seen


def test_generator_profile_post_global_write_runs_on_service_loop(
    live_server_with_dir, monkeypatch
):
    base_url, data_dir, _httpd = live_server_with_dir
    seen: list[str] = []
    orig = config._write

    def spy(data, data_dir=None):
        seen.append(threading.current_thread().name)
        return orig(data, data_dir=data_dir)

    monkeypatch.setattr(config, "_write", spy)
    status, body = _post(f"{base_url}/generator-profile", {"profile_id": "factual-recall"})
    assert status == 200, body
    assert seen, "expected config write to run"
    assert all(name == SERVICE_LOOP_THREAD for name in seen), seen


def test_concurrent_profile_post_and_registrations_lose_no_write(live_server_with_dir):
    """A profile POST and concurrent ensure_agent must all survive.

    Registers a target agent, then hammers the per-agent profile POST for
    it (now serialised on the service loop) while another thread drives
    ``ensure_agent`` onto the same loop, exactly where /ingest runs it. The
    two writers touch the one registry file: with both serialised on the
    loop the profile persists and every registration is retained. Before
    the fix the POST wrote from the request thread and could clobber (or be
    clobbered by) a loop-side ensure_agent write.
    """
    base_url, data_dir, httpd = live_server_with_dir
    agents.AgentRegistry(data_dir).register_agent("target")

    errors: list[str] = []

    def set_profile():
        for _ in range(25):
            status, body = _post(
                f"{base_url}/generator-profile",
                {"profile_id": "factual-recall", "agent": "target"},
            )
            if status != 200:
                errors.append(f"profile POST {status}: {body}")

    names = [f"worker{i}" for i in range(40)]

    def register_agents():
        # Mirror /ingest: ensure_agent runs on the service loop, in the
        # server's data dir.
        for n in names:
            async def _reg(name=n):
                agents.ensure_agent(name, data_dir=data_dir)
            httpd.service_loop.run(_reg())

    threads = [threading.Thread(target=set_profile), threading.Thread(target=register_agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    # The profile write survived.
    assert agents.get_agent_generator_profile("target", data_dir=data_dir) == "factual-recall"
    # Every concurrent registration survived (no lost update dropped an agent).
    registered = {a["name"] for a in agents.AgentRegistry(data_dir).list_agents()}
    missing = set(names) - registered
    assert not missing, f"lost registrations: {sorted(missing)}"
