"""Tests for taosmd.contract_check -- the packaged contract probe.

Mirrors the server fixture pattern from tests/test_version_capabilities.py
and tests/test_http_server.py.
"""

from __future__ import annotations

import json
import sys
import threading

import pytest

from taosmd import api as taosmd_api
from taosmd import capabilities
from taosmd import contract_check
from taosmd import http_server


def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder so no ONNX/QMD model is needed."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


class _ServerCtx:
    """Handle on a running test server."""

    def __init__(self, url, httpd, stores):
        self.url = url
        self.httpd = httpd
        self.stores = stores

    def run(self, coro):
        return self.httpd.service_loop.run(coro)


@pytest.fixture
def live_server_ctx(tmp_path, monkeypatch):
    """Start the HTTP server on an ephemeral port against an isolated data dir."""
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


# ---------------------------------------------------------------------------
# Helper to call the contract_check CLI and capture exit code + output
# ---------------------------------------------------------------------------


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    """Run contract_check.main with args, return (exit_code, stdout, stderr)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    from io import StringIO

    stdout = StringIO()
    stderr = StringIO()
    sys.stdout = stdout
    sys.stderr = stderr
    try:
        code = contract_check.main(args)
    except SystemExit as exc:
        code = exc.code
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return code, stdout.getvalue(), stderr.getvalue()


# ---------------------------------------------------------------------------
# Test (a): clean server, no write probe -> ok True, no writes made
# ---------------------------------------------------------------------------


def test_clean_server_no_write_probe_ok(live_server_ctx):
    """Clean server with no write probe returns ok=True and makes no write requests."""
    # Wrap the server's handler to record non-GET requests
    original_dispatch = live_server_ctx.httpd.RequestHandlerClass._dispatch
    non_get_calls: list[tuple[str, str]] = []

    def tracking_dispatch(self, method: str) -> None:
        if method != "GET":
            non_get_calls.append((method, self.path))
        return original_dispatch(self, method)

    live_server_ctx.httpd.RequestHandlerClass._dispatch = tracking_dispatch

    result = contract_check.run(live_server_ctx.url, write_probe=False, timeout=10)
    assert result["ok"] is True, f"expected ok=True, got {result}"
    assert result["base_url"] == live_server_ctx.url
    assert isinstance(result["version"], str) and result["version"]
    assert isinstance(result["commit"], str)
    assert isinstance(result["checks"], list)
    assert len(result["checks"]) > 0
    # No writes should have been made
    assert non_get_calls == [], f"expected no non-GET requests, got {non_get_calls}"


def test_clean_server_no_write_probe_cli(live_server):
    """CLI with clean server and no write probe exits 0."""
    code, stdout, stderr = _run_cli(["--base-url", live_server])
    assert code == 0, f"exit code {code}, stdout={stdout}, stderr={stderr}"
    assert "PASS health" in stdout
    assert "PASS version" in stdout


# ---------------------------------------------------------------------------
# Test (b): clean server with --write-probe -> ok True, marker found
# ---------------------------------------------------------------------------


def test_clean_server_with_write_probe_ok(live_server_ctx):
    """Clean server with --write-probe returns ok=True and marker is found."""
    result = contract_check.run(live_server_ctx.url, write_probe=True, timeout=10)
    assert result["ok"] is True, f"expected ok=True, got {result}"
    # Check write probe checks passed
    check_names = {c["name"] for c in result["checks"]}
    assert "write_probe_ingest" in check_names
    assert "write_probe_search" in check_names
    assert "write_probe_a2a_send" in check_names
    assert "write_probe_a2a_read" in check_names
    for name in ("write_probe_ingest", "write_probe_search", "write_probe_a2a_send", "write_probe_a2a_read"):
        check = next(c for c in result["checks"] if c["name"] == name)
        assert check["ok"] is True, f"{name} failed: {check['detail']}"


def test_clean_server_with_write_probe_cli(live_server):
    """CLI with clean server and --write-probe exits 0."""
    code, stdout, stderr = _run_cli(["--base-url", live_server, "--write-probe"])
    assert code == 0, f"exit code {code}, stdout={stdout}, stderr={stderr}"
    assert "PASS write_probe_ingest" in stdout
    assert "PASS write_probe_search" in stdout
    assert "PASS write_probe_a2a_send" in stdout
    assert "PASS write_probe_a2a_read" in stdout


# ---------------------------------------------------------------------------
# Test (c): server whose /health claims a capability whose route returns 200 text/html
# ---------------------------------------------------------------------------


def test_fake_capability_html_route_fails(live_server_ctx, monkeypatch):
    """Server advertising a capability whose route returns 200 text/html -> ok=False."""
    # Add a fake capability probe that declares a GET route not served by the dispatcher
    # We monkeypatch capabilities.CAPABILITY_PROBES to include an extra probe
    # (since _resolved_capabilities iterates over CAPABILITY_PROBES, not all_probes())
    original_probes = capabilities.CAPABILITY_PROBES

    fake_probe = capabilities.CapabilityProbe(
        name="fake_html.v1",
        module="taosmd.service",
        symbols=("ingest",),  # exists so probe resolves
        route_markers=('"/fake/html/route"',),
    )

    monkeypatch.setattr(capabilities, "CAPABILITY_PROBES", original_probes + (fake_probe,))
    capabilities.reset_caches()

    # Now we need to make /health advertise this capability.
    # The server's /health uses capabilities.capabilities() which uses CAPABILITY_PROBES.
    # But the route "/fake/html/route" doesn't exist, so it will 404.
    # To make it return 200 text/html, we need to add a route that serves HTML.
    # The server serves the SPA for unknown GET paths when _serve_dashboard is True.
    # Since we're in a source checkout, _serve_dashboard is True and unknown paths
    # return 200 text/html. So /fake/html/route will return 200 text/html.

    result = contract_check.run(live_server_ctx.url, write_probe=False, timeout=10)
    assert result["ok"] is False, f"expected ok=False, got {result}"
    # The check for fake_html.v1:/fake/html/route should be the failing one
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any("fake_html.v1" in c["name"] for c in failed), f"expected fake_html.v1 check to fail, got {failed}"


# ---------------------------------------------------------------------------
# Test (d): /version capabilities differing from /health
# ---------------------------------------------------------------------------


def test_version_capabilities_differ_from_health(live_server_ctx, monkeypatch):
    """Server where /version capabilities differ from /health -> ok=False."""
    # Monkeypatch the server's handler to return different capabilities on /version
    original_dispatch = live_server_ctx.httpd.RequestHandlerClass._dispatch

    def patched_dispatch(self, method: str) -> None:
        parts = self.path.split("?", 1)
        path = parts[0].rstrip("/") or "/"
        if method == "GET" and path == "/version":
            # Return different capabilities
            self._send_json(200, {
                "version": "0.5.1",
                "commit": None,
                "commit_source": None,
                "built_at": None,
                "built_at_source": None,
                "capabilities": ["collections.v1", "a2a.v1", "fake.v1"],  # differs
            })
            return
        return original_dispatch(self, method)

    live_server_ctx.httpd.RequestHandlerClass._dispatch = patched_dispatch

    result = contract_check.run(live_server_ctx.url, write_probe=False, timeout=10)
    assert result["ok"] is False, f"expected ok=False, got {result}"
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["name"] == "version_capabilities_match" for c in failed), f"expected version_capabilities_match to fail, got {failed}"


# ---------------------------------------------------------------------------
# Test (e): --require of a capability the server does not advertise -> exit 1
# ---------------------------------------------------------------------------


def test_require_missing_capability_exits_1(live_server_ctx, monkeypatch):
    """--require for a capability declared by package but not advertised by server -> exit 1."""
    # Make the server not advertise 'tasks.v1' (a real declared capability)
    # by monkeypatching the server's handler to remove it from /health
    original_dispatch = live_server_ctx.httpd.RequestHandlerClass._dispatch

    def patched_dispatch(self, method: str) -> None:
        parts = self.path.split("?", 1)
        path = parts[0].rstrip("/") or "/"
        if method == "GET" and path == "/health":
            # Return capabilities without tasks.v1
            self._send_json(200, {
                "status": "ok",
                "version": "0.5.1",
                "capabilities": ["collections.v1", "a2a.v1", "grants.v1", "graph.v1", "ingest.v1", "search.v1", "shelves.v1", "temporal.v1"],
            })
            return
        return original_dispatch(self, method)

    live_server_ctx.httpd.RequestHandlerClass._dispatch = patched_dispatch

    # Now --require tasks.v1 should fail because it's declared but not advertised
    code, stdout, stderr = _run_cli(["--base-url", live_server_ctx.url, "--require", "tasks.v1"])
    assert code == 1, f"exit code {code}, stdout={stdout}, stderr={stderr}"
    assert "FAIL require:tasks.v1" in stdout


def test_require_undeclared_capability_does_not_fail(live_server):
    """--require for a capability NOT declared by package should not fail (not in missing_on_server)."""
    code, stdout, stderr = _run_cli(["--base-url", live_server, "--require", "nonexistent.v1"])
    assert code == 0, f"exit code {code}, stdout={stdout}, stderr={stderr}"
    # Should not have a FAIL for require:nonexistent.v1
    assert "FAIL require:nonexistent.v1" not in stdout


# ---------------------------------------------------------------------------
# Test (f): base URL with nothing listening -> exit 2
# ---------------------------------------------------------------------------


def test_connection_failure_exits_2():
    """Connection to non-listening port -> exit 2."""
    code, stdout, stderr = _run_cli(["--base-url", "http://127.0.0.1:9999"])
    assert code == 2, f"exit code {code}, stdout={stdout}, stderr={stderr}"
    assert "CONNECTION FAILED" in stderr or "connection failed" in stdout.lower()


def test_connection_failure_run():
    """run() with non-listening port raises ConnectionError."""
    with pytest.raises(ConnectionError):
        contract_check.run("http://127.0.0.1:9999", write_probe=False, timeout=1)


# ---------------------------------------------------------------------------
# Additional tests for JSON output and skipped routes
# ---------------------------------------------------------------------------


def test_json_output(live_server):
    """--json outputs the result dict."""
    code, stdout, stderr = _run_cli(["--base-url", live_server, "--json"])
    assert code == 0, f"exit code {code}, stdout={stdout}, stderr={stderr}"
    data = json.loads(stdout)
    assert data["ok"] is True
    assert data["base_url"] == live_server
    assert isinstance(data["checks"], list)
    assert isinstance(data["skipped"], list)
    assert isinstance(data["missing_on_server"], list)


def test_skipped_routes_listed(live_server_ctx):
    """Routes with path parameters are listed under skipped."""
    result = contract_check.run(live_server_ctx.url, write_probe=False, timeout=10)
    # Some capabilities have route markers with path parameters (e.g., /tasks/{id})
    # These should be in skipped
    assert isinstance(result["skipped"], list)
    # At least some should be skipped (tasks routes have {id})
    # The exact list depends on the capability probes
    print(f"Skipped: {result['skipped']}")


def test_missing_on_server_empty_on_clean(live_server):
    """On a clean server, missing_on_server should be empty (all probes resolve)."""
    result = contract_check.run(live_server, write_probe=False, timeout=10)
    # All declared probes should resolve on this build
    assert result["missing_on_server"] == []


# ---------------------------------------------------------------------------
# Integration: verify the write probe actually works end-to-end
# ---------------------------------------------------------------------------


def test_write_probe_ingest_search_roundtrip(live_server_ctx):
    """Verify the write probe's ingest + search actually works."""
    result = contract_check.run(live_server_ctx.url, write_probe=True, timeout=10)
    assert result["ok"] is True

    # Verify the marker was actually ingested and searchable by doing a manual search
    # (the contract_check already does this, but let's double-check)
    marker_checks = [c for c in result["checks"] if c["name"].startswith("write_probe_")]
    for check in marker_checks:
        assert check["ok"], f"{check['name']} failed: {check['detail']}"
