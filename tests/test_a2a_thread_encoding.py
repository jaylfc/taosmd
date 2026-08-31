"""Tests for percent-encoding of {thread} path segments in A2A thread routes.

tsk-5jprfp: GET /a2a/threads/{thread} and GET /a2a/threads/{thread}/messages
were neither encoded by the client nor decoded by the server dispatch branch.
This caused InvalidURL, silent empty results, or UnicodeEncodeError for thread
names containing spaces, hashes, slashes, or non-ASCII characters.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server
from taosmd.remote import RemoteClient


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_a2a_membership.py / test_remote.py)
# ---------------------------------------------------------------------------


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-thread-enc"
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


def _raw_get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _raw_post(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# UNIT TESTS — pin the encode function in RemoteClient.a2a_thread_messages
# ---------------------------------------------------------------------------


class TestRemoteClientThreadEncoding:
    """Unit tests pinning the percent-encoding of {thread} in RemoteClient."""

    @staticmethod
    def _make_rc():
        return RemoteClient("http://fake:9999")

    def _check_path(self, thread: str, expected: str) -> None:
        rc = self._make_rc()
        captured: dict = {}

        async def fake_run(method, path, body=None, params=None):
            captured["path"] = path
            return {"thread": "test", "messages": []}

        with patch.object(rc, "_run", side_effect=fake_run):
            asyncio.run(rc.a2a_thread_messages(thread=thread))
        assert captured["path"] == expected

    def test_encodes_space(self):
        self._check_path("proj x", "/a2a/threads/proj%20x/messages")

    def test_encodes_hash(self):
        self._check_path("proj#x", "/a2a/threads/proj%23x/messages")

    def test_encodes_slash(self):
        self._check_path("proj/x", "/a2a/threads/proj%2Fx/messages")

    def test_encodes_nonascii(self):
        self._check_path("proj\u00e9", "/a2a/threads/proj%C3%A9/messages")


# ---------------------------------------------------------------------------
# UNIT TESTS — pin the unquote() in the server dispatch branch
# ---------------------------------------------------------------------------


class TestServerDispatchDecode:
    """Unit tests pinning the unquote() call in the /a2a/threads/{thread} dispatch."""

    def test_decodes_messages_path_thread(self, live_server):
        """GET /a2a/threads/{encoded}/messages: server decodes thread in response."""
        _raw_post(
            f"{live_server}/a2a/send", {"from": "s", "body": "msg", "thread": "proj x"}
        )
        status, body = _raw_get(f"{live_server}/a2a/threads/proj%20x/messages")
        assert status == 200, body
        assert body["thread"] == "proj x"
        assert len(body["messages"]) == 1

    def test_decodes_bare_thread_path(self, live_server):
        """GET /a2a/threads/{encoded}: server decodes thread in bare path."""
        _raw_post(
            f"{live_server}/a2a/send", {"from": "s", "body": "msg", "thread": "proj x"}
        )
        status, body = _raw_get(f"{live_server}/a2a/threads/proj%20x")
        assert status == 200, body
        assert body["thread"] == "proj x"
        assert len(body["messages"]) == 1


# ---------------------------------------------------------------------------
# END-TO-END PROBE — real RemoteClient against real server
# Reproduces the full DEFECT / SERVER table from tsk-5jprfp.
# Every row must go from broken (pre-fix) to correct (post-fix) and all
# share the same command so a green cannot be a blind probe.
# ---------------------------------------------------------------------------


class TestEndToEndThreadEncoding:
    """Full table probe: CONTROL + DEFECT + SERVER rows in one command."""

    def test_full_encoding_table(self, live_server):
        rc = RemoteClient(live_server)

        # --- Seed each thread with exactly 1 message ---
        # Thread names with special characters are sent via a2a/send
        # (POST body, no path encoding needed), so they store correctly.
        seed_threads = {
            "projx": "CONTROL_safe",
            "proj x": "DEFECT_space",
            "proj#x": "DEFECT_hash",
            "proj/x": "DEFECT_slash",
            "proj\u00e9": "DEFECT_nonascii",
        }
        for name, label in seed_threads.items():
            status, body = _raw_post(
                f"{live_server}/a2a/send",
                {"from": "sender", "body": f"message in {label}", "thread": name},
            )
            assert status == 200, f"seed failed for '{name}': {body}"

        # --- CONTROL_safe (via RemoteClient) ---
        msgs = asyncio.run(rc.a2a_thread_messages(thread="projx"))
        assert len(msgs["messages"]) == 1, (
            f"CONTROL_safe: expected 1 msg, got {len(msgs.get('messages', []))}"
        )

        # --- DEFECT_space (via RemoteClient) ---
        # Pre-fix: urllib raises InvalidURL on raw space in URL path.
        # Post-fix: client encodes space as %20, server decodes back.
        msgs = asyncio.run(rc.a2a_thread_messages(thread="proj x"))
        assert len(msgs["messages"]) == 1, (
            f"DEFECT_space: expected 1 msg, got {len(msgs.get('messages', []))}"
        )

        # --- DEFECT_hash (via RemoteClient) ---
        # Pre-fix: '#' is treated as URL fragment delimiter by urlsplit;
        #   server sees path /a2a/threads/proj, queries thread 'proj', returns 0.
        # Post-fix: client encodes '#' as %23, server decodes back.
        msgs = asyncio.run(rc.a2a_thread_messages(thread="proj#x"))
        assert len(msgs["messages"]) == 1, (
            f"DEFECT_hash: expected 1 msg, got {len(msgs.get('messages', []))}"
        )

        # --- DEFECT_slash (via RemoteClient) ---
        # Pre-fix (client only): client encodes '/' as %2F; server sees 'proj%2Fx'
        #   and fails to match stored 'proj/x'.
        # Post-fix: server decodes %2F back to '/'.
        msgs = asyncio.run(rc.a2a_thread_messages(thread="proj/x"))
        assert len(msgs["messages"]) == 1, (
            f"DEFECT_slash: expected 1 msg, got {len(msgs.get('messages', []))}"
        )

        # --- DEFECT_nonascii (via RemoteClient) ---
        # Pre-fix: urllib raises UnicodeEncodeError on raw non-ASCII bytes.
        # Post-fix: client encodes e-acute as %C3%A9, server decodes back.
        msgs = asyncio.run(rc.a2a_thread_messages(thread="proj\u00e9"))
        assert len(msgs["messages"]) == 1, (
            f"DEFECT_nonascii: expected 1 msg, got {len(msgs.get('messages', []))}"
        )

        # --- SERVER_given_encoded (via raw HTTP, encoded path) ---
        # Pre-fix: server does not unquote, sees 'proj%20x', no match → msgs=0.
        # Post-fix: server unquotes, sees 'proj x', finds the message.
        status, body = _raw_get(f"{live_server}/a2a/threads/proj%20x/messages?limit=50")
        assert status == 200, body
        assert len(body["messages"]) == 1, (
            f"SERVER_given_encoded: expected 1 msg, got {len(body.get('messages', []))}"
        )

        # --- SERVER_bare_threadread_encoded (via raw HTTP, bare path) ---
        # GET /a2a/threads/{thread} (without /messages suffix).
        # Pre-fix: server sees 'proj%20x', no match → msgs=0.
        # Post-fix: server unquotes → finds the message.
        status, body = _raw_get(f"{live_server}/a2a/threads/proj%20x?limit=50")
        assert status == 200, body
        assert len(body["messages"]) == 1, (
            f"SERVER_bare_threadread_encoded: expected 1 msg, got {len(body.get('messages', []))}"
        )

        # --- SERVER_bare_threadread_CONTROL (via raw HTTP, simple thread) ---
        status, body = _raw_get(f"{live_server}/a2a/threads/projx?limit=50")
        assert status == 200, body
        assert len(body["messages"]) == 1, (
            f"SERVER_bare_threadread_CONTROL: expected 1 msg, got {len(body.get('messages', []))}"
        )
