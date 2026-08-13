"""Tests for scripts/check_remote_routes.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scripts.check_remote_routes import (
    MissingRoute,
    check_remote_routes,
    _extract_route_literals,
    _parse_remote_calls,
    parse_waived_routes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(src: str, text: str) -> Path:
    p = src / "tmp.py"
    p.write_text(text, encoding="utf-8")
    return p


REMOTE_BASE = '''
from taosmd.remote import RemoteClient

class Foo:
    def _run(self, method, path):
        pass

    async def ok(self):
        return await self._run("POST", "/ingest")

    async def missing(self):
        return await self._run("POST", "/no-route")

    async def fstring(self):
        return await self._run("POST", f"/tasks/{{x}}")
'''


HTTP_BASE = '''
def _dispatch(self, method, path):
    if method == "POST" and path == "/ingest":
        pass
    if method == "POST" and path == "/no-route":
        pass
'''


# ---------------------------------------------------------------------------
# _parse_remote_calls
# ---------------------------------------------------------------------------

class TestParseRemoteCalls:
    def test_literal_calls_extracted(self):
        calls, unparseable = _parse_remote_calls(REMOTE_BASE)
        methods = {c.method for c in calls}
        paths = {c.path for c in calls}
        assert "POST" in methods
        assert "/ingest" in paths
        assert "/no-route" in paths
        assert unparseable == 1

    def test_unparseable_count(self):
        _, unparseable = _parse_remote_calls(REMOTE_BASE)
        assert unparseable == 1


# ---------------------------------------------------------------------------
# _extract_route_literals
# ---------------------------------------------------------------------------

class TestExtractRouteLiterals:
    def test_exact_match(self):
        src = '''
def _dispatch(self, method, path):
    if method == "POST" and path == "/ingest":
        pass
'''
        assert "/ingest" in _extract_route_literals(src)

    def test_prefix_match(self):
        src = '''
def _dispatch(self, method, path):
    if method == "POST" and path.startswith("/tasks/"):
        pass
'''
        assert "/tasks/" in _extract_route_literals(src)

    def test_suffix_match(self):
        src = '''
def _dispatch(self, method, path):
    if path.endswith("/edges"):
        pass
'''
        assert "/edges" in _extract_route_literals(src)

    def test_in_operator(self):
        src = '''
def _dispatch(self, method, path):
    if path in ("/a", "/b"):
        pass
'''
        assert "/a" in _extract_route_literals(src)
        assert "/b" in _extract_route_literals(src)


# ---------------------------------------------------------------------------
# parse_waived_routes
# ---------------------------------------------------------------------------

class TestParseWaivedRoutes:
    def test_no_trailer(self):
        assert parse_waived_routes(None) == set()
        assert parse_waived_routes("hello world") == set()

    def test_single_route(self):
        body = "Some description\n\nMissing-Route-Intentionally: POST /no-route"
        assert parse_waived_routes(body) == {"POST /no-route"}

    def test_multiple_routes(self):
        body = "Missing-Route-Intentionally: POST /a, GET /b"
        assert parse_waived_routes(body) == {"POST /a", "GET /b"}

    def test_trailer_with_surrounding_text(self):
        body = "Fixed in another PR. Missing-Route-Intentionally: POST /x"
        assert parse_waived_routes(body) == {"POST /x"}


# ---------------------------------------------------------------------------
# check_remote_routes
# ---------------------------------------------------------------------------

class TestCheckRemoteRoutes:
    def test_clean_when_all_routes_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote.py"
            http = Path(tmp) / "http_server.py"
            remote.write_text(REMOTE_BASE, encoding="utf-8")
            http.write_text(HTTP_BASE, encoding="utf-8")
            missing, waived, unparseable = check_remote_routes(remote_path=remote, http_path=http)
            assert missing == []
            assert waived == set()
            assert unparseable == 1

    def test_detects_missing_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote.py"
            http = Path(tmp) / "http_server.py"
            remote.write_text(REMOTE_BASE, encoding="utf-8")
            # http_server.py has only /ingest, not /no-route
            http_src = '''
def _dispatch(self, method, path):
    if method == "POST" and path == "/ingest":
        pass
'''
            http.write_text(http_src, encoding="utf-8")
            missing, _, unparseable = check_remote_routes(remote_path=remote, http_path=http)
            assert len(missing) == 1
            m = missing[0]
            assert isinstance(m, MissingRoute)
            assert m.method == "POST"
            assert m.path == "/no-route"
            assert m.lineno == 12
            assert unparseable == 1

    def test_waiver_trailer_suppresses_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote.py"
            http = Path(tmp) / "http_server.py"
            remote.write_text(REMOTE_BASE, encoding="utf-8")
            http.write_text(HTTP_BASE, encoding="utf-8")
            missing, waived, unparseable = check_remote_routes(
                remote_path=remote,
                http_path=http,
                pr_body="Missing-Route-Intentionally: POST /no-route",
            )
            assert missing == []
            assert waived == {"POST /no-route"}
            assert unparseable == 1

    def test_multiple_missing_routes(self):
        remote_src = '''
class Foo:
    def _run(self, method, path):
        pass

    async def a(self):
        return await self._run("POST", "/alpha")

    async def b(self):
        return await self._run("GET", "/beta")
'''
        http_src = '''
def _dispatch(self, method, path):
    if path == "/alpha":
        pass
'''
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote.py"
            http = Path(tmp) / "http_server.py"
            remote.write_text(remote_src, encoding="utf-8")
            http.write_text(http_src, encoding="utf-8")
            missing, _, _ = check_remote_routes(remote_path=remote, http_path=http)
            assert len(missing) == 1
            assert missing[0].path == "/beta"


# ---------------------------------------------------------------------------
# Integration: current repo files
# ---------------------------------------------------------------------------

class TestCurrentRepo:
    def test_master_is_clean(self):
        missing, _, unparseable = check_remote_routes()
        assert missing == [], f"Expected clean but got missing: {missing}"
        assert unparseable == 3

    def test_at_least_twenty_literal_paths(self):
        calls, _ = _parse_remote_calls(
            Path("/tmp/exec-tsk-bwgr26/taosmd/remote.py").read_text(encoding="utf-8")
        )
        assert len(calls) >= 20
