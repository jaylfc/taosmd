"""Tests for taosmd.remote and related config/service dispatch.

Hermetic — uses an ephemeral port and isolated tmp dirs so nothing touches
~/.taosmd or port 7900. The ONNX/QMD embedder is patched with a
deterministic hash vector, matching test_http_server.py style.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import config as taosmd_config
from taosmd import http_server, service as taosmd_service
from taosmd.remote import RemoteClient


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder — no ONNX/QMD model needed."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """Start a real HTTP server on an ephemeral port with an isolated data dir.

    Yields (base_url, data_dir_str).  Cleans up fully on teardown.
    """
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}", str(data_dir)
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
def client(live_server):
    """RemoteClient pointing at the live ephemeral server."""
    base_url, _ = live_server
    return RemoteClient(base_url)


# ---------------------------------------------------------------------------
# RemoteClient round-trips
# ---------------------------------------------------------------------------

def test_remote_ingest_and_search(client):
    """ingest → search returns a hit for the ingested text."""
    result = asyncio.run(client.ingest("Remote memory test content.", agent="remote-test"))
    assert result["archived"] == 1
    assert result["agent"] == "remote-test"

    hits = asyncio.run(client.search("Remote memory test content.", agent="remote-test", limit=3))
    assert hits, "expected at least one hit"
    assert any("Remote memory" in h["text"] for h in hits)


def test_remote_pending_list(client):
    """pending_list returns a list (may be empty)."""
    pending = asyncio.run(client.pending_list(agent="remote-test"))
    assert isinstance(pending, list)


def test_remote_a2a_send_and_feed(client):
    """a2a_send -> a2a_feed returns the sent message."""
    receipt = asyncio.run(
        client.a2a_send("agent-alpha", "Hello from remote!", thread="test-chan")
    )
    assert receipt["from"] == "agent-alpha"
    assert receipt["thread"] == "test-chan"

    msgs = asyncio.run(client.a2a_feed(thread="test-chan", limit=10))
    assert any(m["from"] == "agent-alpha" and "Hello from remote!" in m["body"] for m in msgs)


def test_remote_a2a_send_with_refs_and_blocks(client):
    """a2a_send with refs+blocks -> a2a_feed returns them verbatim."""
    refs = [
        {"kind": "doc", "title": "Spec", "uri": "https://example.com/spec",
         "sha256": "deadbeef", "doc_id": "spec-1", "version": 2, "for": ["agentA"],
         "summary": "API spec"},
        {"kind": "log", "title": "CI Log", "uri": "https://example.com/log",
         "sha256": None, "doc_id": None, "version": None, "for": None,
         "summary": None},
    ]
    blocks = [
        {"kind": "tool_call", "tool": "search", "args": {"query": "hello"}},
        {"kind": "tool_result", "result": "found 3 items"},
    ]
    receipt = asyncio.run(
        client.a2a_send(
            "agent-alpha", "remote envelope msg", thread="remote-env",
            refs=refs, blocks=blocks,
        )
    )
    assert receipt["refs"] == refs
    assert receipt["blocks"] == blocks

    msgs = asyncio.run(client.a2a_feed(thread="remote-env", limit=10))
    assert len(msgs) == 1
    assert msgs[0]["refs"] == refs
    assert msgs[0]["blocks"] == blocks


def test_remote_a2a_channels(client):
    """a2a_channels returns a list after posting."""
    asyncio.run(client.a2a_send("agent-beta", "Ping", thread="chan-for-channels"))
    channels = asyncio.run(client.a2a_channels())
    assert isinstance(channels, list)
    names = {c["channel"] for c in channels}
    assert "chan-for-channels" in names


def test_remote_a2a_members(client):
    """a2a_members returns the senders on a channel."""
    asyncio.run(client.a2a_send("agent-gamma", "Hi", thread="chan-for-members"))
    asyncio.run(client.a2a_send("agent-delta", "Hey", thread="chan-for-members"))
    members = asyncio.run(client.a2a_members(channel="chan-for-members"))
    assert "agent-gamma" in members
    assert "agent-delta" in members


def test_remote_stats_health(client):
    """stats returns reachable=True and a server_version."""
    stats = asyncio.run(client.stats(agent="remote-test"))
    assert stats["reachable"] is True
    assert "server_version" in stats


def test_remote_non_200_raises_runtime_error(live_server):
    """A request to a non-existent endpoint raises RuntimeError with status."""
    base_url, _ = live_server
    bad = RemoteClient(base_url)
    # POST to a non-existent JSON endpoint returns 404
    with pytest.raises(RuntimeError, match="404"):
        asyncio.run(bad._run("POST", "/no-such-endpoint", {"foo": "bar"}))


# ---------------------------------------------------------------------------
# config: server_url / server_token get/set + env override
# ---------------------------------------------------------------------------

@pytest.fixture
def config_data_dir(tmp_path, monkeypatch):
    d = tmp_path / "cfg"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(d))
    # Clear env overrides so config-file path is exercised cleanly.
    monkeypatch.delenv("TAOSMD_SERVER_URL", raising=False)
    monkeypatch.delenv("TAOSMD_TOKEN", raising=False)
    return d


def test_config_server_url_round_trip(config_data_dir):
    taosmd_config.set_server_url("http://pi.local:7900")
    assert taosmd_config.get_server_url() == "http://pi.local:7900"


def test_config_server_url_clear(config_data_dir):
    taosmd_config.set_server_url("http://pi.local:7900")
    taosmd_config.set_server_url("", clear=True)
    assert taosmd_config.get_server_url() is None


def test_config_server_url_unset_is_none(config_data_dir):
    assert taosmd_config.get_server_url() is None


def test_config_server_url_empty_raises(config_data_dir):
    with pytest.raises(ValueError):
        taosmd_config.set_server_url("")
    with pytest.raises(ValueError):
        taosmd_config.set_server_url("   ")


def test_config_server_url_env_override(config_data_dir, monkeypatch):
    taosmd_config.set_server_url("http://file-url:7900")
    monkeypatch.setenv("TAOSMD_SERVER_URL", "http://env-url:7900")
    assert taosmd_config.get_server_url() == "http://env-url:7900"


def test_config_server_token_round_trip(config_data_dir):
    taosmd_config.set_server_token("supersecret123")
    assert taosmd_config.get_server_token() == "supersecret123"


def test_config_server_token_clear(config_data_dir):
    taosmd_config.set_server_token("tok")
    taosmd_config.set_server_token("", clear=True)
    assert taosmd_config.get_server_token() is None


def test_config_server_token_env_override(config_data_dir, monkeypatch):
    taosmd_config.set_server_token("file-token")
    monkeypatch.setenv("TAOSMD_TOKEN", "env-token")
    assert taosmd_config.get_server_token() == "env-token"


# ---------------------------------------------------------------------------
# service.py dispatch: monkeypatch server_url → goes remote; unset → local
# ---------------------------------------------------------------------------

def test_service_dispatch_goes_remote_when_url_set(live_server, tmp_path, monkeypatch):
    """When server_url is in config.json for a data_dir, service.search hits remote."""
    base_url, data_dir_str = live_server
    cfg_dir = tmp_path / "cfg-dispatch"
    cfg_dir.mkdir()

    # Pre-ingest via remote so there is something to find.
    rc = RemoteClient(base_url)
    asyncio.run(rc.ingest("service dispatch remote test", agent="dispatch-agent"))

    # Write server_url into the data_dir's config.json (the file-only path that
    # bypasses env-var for an explicit data_dir, preventing infinite recursion
    # in the server itself).
    cfg_file = cfg_dir / "config.json"
    cfg_file.write_text(json.dumps({"server_url": base_url}))

    # Flush the remote cache so it re-reads the config file.
    taosmd_service._remote_cache.clear()

    hits = asyncio.run(taosmd_service.search(
        "service dispatch remote test",
        agent="dispatch-agent",
        data_dir=str(cfg_dir),
    ))
    assert hits, "expected hits via remote dispatch"

    # Cleanup: clear remote cache
    taosmd_service._remote_cache.clear()


def test_service_dispatch_local_when_no_url(tmp_path, monkeypatch):
    """When get_server_url() returns None, service.search uses local stores."""
    data_dir = tmp_path / "local-dispatch"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    monkeypatch.delenv("TAOSMD_SERVER_URL", raising=False)
    taosmd_service._remote_cache.clear()

    # Make a local store with a patched embedder.
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    # Ingest and search locally.
    asyncio.run(taosmd_service.ingest("local dispatch test", agent="local-agent", data_dir=str(data_dir)))
    hits = asyncio.run(taosmd_service.search("local dispatch test", agent="local-agent", data_dir=str(data_dir)))
    assert hits, "expected local hits"


# ---------------------------------------------------------------------------
# Token auth: server with token set → no-header 401, header 200, health open
# ---------------------------------------------------------------------------

@pytest.fixture
def token_server(tmp_path, monkeypatch):
    """Start a server with server_token configured, yielding (base_url, token)."""
    data_dir = tmp_path / "token-data"
    data_dir.mkdir()
    cfg_dir = tmp_path / "token-cfg"
    cfg_dir.mkdir()

    TOKEN = "test-bearer-token-xyz"

    # Write the token into config.json in the data_dir so the server reads it.
    cfg_file = data_dir / "config.json"
    cfg_file.write_text(json.dumps({"server_token": TOKEN}))

    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    # Reload http_server module-level _config so it reads the token we just wrote.
    # The token is read at _make_handler() call time, so we pass data_dir.
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}", TOKEN
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


def _raw_get(url: str, headers: dict | None = None) -> tuple[int, dict]:
    """GET with optional extra headers; returns (status, body)."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _raw_post(url: str, payload: dict, headers: dict | None = None) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def test_token_health_always_open(token_server):
    """/health is accessible without a token."""
    base_url, _ = token_server
    status, body = _raw_get(f"{base_url}/health")
    assert status == 200
    assert body["status"] == "ok"


def test_token_data_endpoint_no_header_returns_401(token_server):
    """POST /ingest without Authorization header returns 401."""
    base_url, _ = token_server
    status, body = _raw_post(
        f"{base_url}/ingest",
        {"text": "blocked text", "agent": "tester"},
    )
    assert status == 401
    assert "Unauthorized" in body.get("error", "")


def test_token_data_endpoint_wrong_token_returns_401(token_server):
    """POST /ingest with wrong token returns 401."""
    base_url, _ = token_server
    status, body = _raw_post(
        f"{base_url}/ingest",
        {"text": "blocked text", "agent": "tester"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert status == 401


def test_token_data_endpoint_correct_token_returns_200(token_server):
    """POST /ingest with the correct token returns 200."""
    base_url, token = token_server
    status, body = _raw_post(
        f"{base_url}/ingest",
        {"text": "authenticated content", "agent": "auth-agent"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert status == 200, body
    assert body["archived"] == 1


def test_token_remote_client_sends_token(token_server):
    """RemoteClient with the correct token successfully ingests."""
    base_url, token = token_server
    rc = RemoteClient(base_url, token=token)
    result = asyncio.run(rc.ingest("remote client token test", agent="rc-auth"))
    assert result["archived"] == 1


def test_token_remote_client_no_token_raises(token_server):
    """RemoteClient without a token raises RuntimeError on data endpoints."""
    base_url, _ = token_server
    rc = RemoteClient(base_url)  # no token
    with pytest.raises(RuntimeError, match="401"):
        asyncio.run(rc.ingest("should be blocked", agent="no-token-agent"))


# ---------------------------------------------------------------------------
# install-skill copies into a tmp HOME, idempotent without --force
# ---------------------------------------------------------------------------

def test_install_skill_copies_skill(tmp_path, monkeypatch):
    """install-skill copies SKILL.md into ~/.claude/skills/taosmd-a2a/."""
    import argparse  # noqa: PLC0415
    from taosmd.cli import _install_skill_cmd  # noqa: PLC0415

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    # Patch expanduser so Path("~/.claude/skills/...") resolves under fake_home.
    import pathlib  # noqa: PLC0415
    original_expanduser = pathlib.Path.expanduser

    def _patched_expanduser(self):
        p = str(self)
        if p.startswith("~"):
            return pathlib.Path(str(fake_home) + p[1:])
        return original_expanduser(self)

    monkeypatch.setattr(pathlib.Path, "expanduser", _patched_expanduser)

    args = argparse.Namespace(force=False)
    rc = _install_skill_cmd(args)
    assert rc == 0

    skill_dest = fake_home / ".claude" / "skills" / "taosmd-a2a" / "SKILL.md"
    assert skill_dest.exists(), f"SKILL.md not found at {skill_dest}"
    content = skill_dest.read_text()
    assert "taosmd-a2a" in content


def test_install_skill_idempotent_without_force(tmp_path, monkeypatch):
    """Second install-skill call without --force returns 0 and does not overwrite."""
    import argparse  # noqa: PLC0415
    from taosmd.cli import _install_skill_cmd  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    fake_home = tmp_path / "home2"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    original_expanduser = pathlib.Path.expanduser

    def _patched(self):
        p = str(self)
        if p.startswith("~"):
            return pathlib.Path(str(fake_home) + p[1:])
        return original_expanduser(self)

    monkeypatch.setattr(pathlib.Path, "expanduser", _patched)

    args_first = argparse.Namespace(force=False)
    _install_skill_cmd(args_first)

    # Write a sentinel into the destination to verify idempotency.
    skill_dest = fake_home / ".claude" / "skills" / "taosmd-a2a" / "SKILL.md"
    skill_dest.write_text("sentinel content")

    args_second = argparse.Namespace(force=False)
    rc = _install_skill_cmd(args_second)
    assert rc == 0
    # Content should still be the sentinel (not overwritten).
    assert skill_dest.read_text() == "sentinel content"


def test_install_skill_force_overwrites(tmp_path, monkeypatch):
    """install-skill --force overwrites an existing installation."""
    import argparse  # noqa: PLC0415
    from taosmd.cli import _install_skill_cmd  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    fake_home = tmp_path / "home3"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    original_expanduser = pathlib.Path.expanduser

    def _patched(self):
        p = str(self)
        if p.startswith("~"):
            return pathlib.Path(str(fake_home) + p[1:])
        return original_expanduser(self)

    monkeypatch.setattr(pathlib.Path, "expanduser", _patched)

    # First install, then clobber.
    _install_skill_cmd(argparse.Namespace(force=False))
    skill_dest = fake_home / ".claude" / "skills" / "taosmd-a2a" / "SKILL.md"
    skill_dest.write_text("old content")

    rc = _install_skill_cmd(argparse.Namespace(force=True))
    assert rc == 0
    new_content = skill_dest.read_text()
    assert new_content != "old content"
    assert "taosmd-a2a" in new_content


def test_remote_task_list_edges_cross_project_scoping(client):
    """RemoteClient task_list_edges with project= returns only edges in that project."""
    # Seed tasks in two projects
    ta1 = asyncio.run(client.task_create(title="A1", created_by="a", project="proj-a"))
    ta2 = asyncio.run(client.task_create(title="A2", created_by="a", project="proj-a"))
    tb1 = asyncio.run(client.task_create(title="B1", created_by="a", project="proj-b"))
    tb2 = asyncio.run(client.task_create(title="B2", created_by="a", project="proj-b"))

    # Seed edges in each project
    asyncio.run(client.task_add_edge(ta1["id"], ta2["id"], "blocks", created_by="a"))
    asyncio.run(client.task_add_edge(tb1["id"], tb2["id"], "blocks", created_by="a"))

    # Without project scope, both edges are returned
    all_edges = asyncio.run(client.task_list_edges())
    assert len(all_edges) == 2

    # With project=proj-a, only the proj-a edge is returned
    proj_a_edges = asyncio.run(client.task_list_edges(project="proj-a"))
    assert len(proj_a_edges) == 1
    assert proj_a_edges[0]["from_id"] == ta1["id"]
    assert proj_a_edges[0]["to_id"] == ta2["id"]

    # Positive control: service.task_list is correctly scoped on the same hop
    proj_a_tasks = asyncio.run(client.task_list(project="proj-a"))
    assert len(proj_a_tasks) == 2
    assert {t["id"] for t in proj_a_tasks} == {ta1["id"], ta2["id"]}
