"""Tests for taosmd.ref_fetch: resolver + verified fetch helper."""

from __future__ import annotations

import asyncio
import hashlib
import os
import urllib.parse

import pytest

from taosmd import config
from taosmd.ref_fetch import (
    HashMismatchError,
    NotFoundError,
    RefFetchError,
    UnauthorizedError,
    fetch_by_ref,
    resolve_ref_uri,
)


# ---------------------------------------------------------------------------
# Resolver tests
# ---------------------------------------------------------------------------

class TestResolveRefUri:
    def test_taos_uri_maps_to_files_endpoint(self):
        ref = {"uri": "taos://myproj/files/docs/readme.md", "sha256": "abc123"}
        url = resolve_ref_uri(ref, "http://controller:8000")
        assert url == "http://controller:8000/api/projects/myproj/files/docs/readme.md"

    def test_taos_uri_with_special_characters_is_quoted(self):
        ref = {"uri": "taos://my proj/files/path with spaces/file.txt", "sha256": "abc123"}
        url = resolve_ref_uri(ref, "http://controller:8000")
        assert "my%20proj" in url
        assert "path%20with%20spaces" in url
        assert url == "http://controller:8000/api/projects/my%20proj/files/path%20with%20spaces/file.txt"

    def test_dot_segment_traversal_rejected(self):
        ref = {"uri": "taos://proj/files/../../admin/secrets", "sha256": "abc123"}
        with pytest.raises(ValueError, match="dot segment"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_encoded_dot_segment_traversal_rejected(self):
        ref = {"uri": "taos://proj/files/%2e%2e/%2e%2e/admin/secrets", "sha256": "abc123"}
        with pytest.raises(ValueError, match="dot segment"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_mixed_dot_segment_traversal_rejected(self):
        ref = {"uri": "taos://proj/files/a/../../../api/registry/tokens", "sha256": "abc123"}
        with pytest.raises(ValueError, match="dot segment"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_leading_slash_path_rejected(self):
        ref = {"uri": "taos://proj/files//etc/passwd", "sha256": "abc123"}
        with pytest.raises(ValueError, match="absolute"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_non_taos_uri_raises(self):
        ref = {"uri": "https://example.com/file", "sha256": "abc123"}
        with pytest.raises(ValueError, match="unsupported uri scheme"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_empty_uri_raises(self):
        ref = {"uri": "", "sha256": "abc123"}
        with pytest.raises(ValueError, match="unsupported uri scheme"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_missing_uri_field_raises(self):
        ref = {"sha256": "abc123"}
        with pytest.raises(ValueError, match="unsupported uri scheme"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_invalid_taos_shape_raises(self):
        ref = {"uri": "taos://myproj", "sha256": "abc123"}
        with pytest.raises(ValueError, match="invalid taos ref uri"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_taos_uri_without_files_segment_raises(self):
        ref = {"uri": "taos://myproj/other/file.txt", "sha256": "abc123"}
        with pytest.raises(ValueError, match="invalid taos ref uri"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_taos_uri_empty_path_raises(self):
        ref = {"uri": "taos://myproj/files/", "sha256": "abc123"}
        with pytest.raises(ValueError, match="path is empty"):
            resolve_ref_uri(ref, "http://controller:8000")

    def test_non_dict_ref_raises(self):
        with pytest.raises(ValueError, match="unsupported uri scheme"):
            resolve_ref_uri("not-a-dict", "http://controller:8000")


# ---------------------------------------------------------------------------
# fetch_by_ref tests
# ---------------------------------------------------------------------------

class TestFetchByRef:
    def test_fetch_match_returns_verified_bytes(self, monkeypatch):
        content = b"hello world"
        expected_sha = hashlib.sha256(content).hexdigest()
        ref = {"uri": "taos://proj/files/hello.txt", "sha256": expected_sha}

        monkeypatch.setenv("TAOSMD_FILES_URL", "http://ctrl:8000")

        captured = {}

        def fake_fetcher(url, agent):
            captured["url"] = url
            captured["agent"] = agent
            return content

        result = asyncio.run(fetch_by_ref(ref, fake_fetcher, "test-agent"))
        assert result == content
        assert captured["url"] == "http://ctrl:8000/api/projects/proj/files/hello.txt"
        assert captured["agent"] == "test-agent"

    def test_fetch_mismatch_error_contains_no_hash(self, monkeypatch):
        content = b"hello world"
        ref = {"uri": "taos://proj/files/hello.txt", "sha256": "deadbeef" * 8}

        monkeypatch.setenv("TAOSMD_FILES_URL", "http://ctrl:8000")

        def fake_fetcher(url, agent):
            return content

        with pytest.raises(HashMismatchError) as exc_info:
            asyncio.run(fetch_by_ref(ref, fake_fetcher, "test-agent"))
        assert "deadbeef" not in str(exc_info.value)
        assert "sha256 mismatch" in str(exc_info.value)

    def test_fetcher_not_found_propagates(self, monkeypatch):
        ref = {"uri": "taos://proj/files/missing.txt", "sha256": "abc123"}

        monkeypatch.setenv("TAOSMD_FILES_URL", "http://ctrl:8000")

        def fake_fetcher(url, agent):
            raise NotFoundError(f"HTTP 404 from {url}")

        with pytest.raises(NotFoundError):
            asyncio.run(fetch_by_ref(ref, fake_fetcher, "test-agent"))

    def test_fetcher_unauthorized_propagates(self, monkeypatch):
        ref = {"uri": "taos://proj/files/secret.txt", "sha256": "abc123"}

        monkeypatch.setenv("TAOSMD_FILES_URL", "http://ctrl:8000")

        def fake_fetcher(url, agent):
            raise UnauthorizedError(f"HTTP 401 from {url}")

        with pytest.raises(UnauthorizedError):
            asyncio.run(fetch_by_ref(ref, fake_fetcher, "test-agent"))

    def test_non_taos_ref_raises_before_fetch(self, monkeypatch):
        ref = {"uri": "https://example.com/file", "sha256": "abc123"}

        monkeypatch.setenv("TAOSMD_FILES_URL", "http://ctrl:8000")

        def fake_fetcher(url, agent):
            raise AssertionError("fetcher should not be called for non-taos:// uri")

        with pytest.raises(ValueError, match="unsupported uri scheme"):
            asyncio.run(fetch_by_ref(ref, fake_fetcher, "test-agent"))

    def test_ref_without_sha256_raises(self, monkeypatch):
        ref = {"uri": "taos://proj/files/hello.txt"}

        monkeypatch.setenv("TAOSMD_FILES_URL", "http://ctrl:8000")

        def fake_fetcher(url, agent):
            return b"data"

        with pytest.raises(RefFetchError, match="no sha256"):
            asyncio.run(fetch_by_ref(ref, fake_fetcher, "test-agent"))

    def test_redirect_no_authorization_leak(self, monkeypatch):
        import http.server
        import threading
        import time
        import urllib.request
        import urllib.error
        import asyncio
        import hashlib

        from taosmd import service as svc
        from taosmd import config as cfg

        # shared state between test and HTTP servers
        recorded = {"headers": {}}

        # Origin B: records every request header it receives
        class OriginBHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                recorded["headers"] = dict(self.headers)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(b"ok")
            def log_message(self, format, *args):
                pass

        # Origin A: returns 302 redirect to origin B
        class OriginAHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{b_port}/")
                self.end_headers()
            def log_message(self, format, *args):
                pass

        # Start origin B on an ephemeral port
        b_server = http.server.HTTPServer(("127.0.0.1", 0), OriginBHandler)
        b_port = b_server.server_address[1]
        b_thread = threading.Thread(target=b_server.serve_forever, daemon=True)
        b_thread.start()
        time.sleep(0.05)

        # Start origin A on another ephemeral port
        a_server = http.server.HTTPServer(("127.0.0.1", 0), OriginAHandler)
        a_port = a_server.server_address[1]
        a_thread = threading.Thread(target=a_server.serve_forever, daemon=True)
        a_thread.start()
        time.sleep(0.05)

        try:
            # Configure registry token and files URL pointing to origin A
            monkeypatch.setattr(cfg, "get_files_url", lambda d: f"http://localhost:{a_port}")
            monkeypatch.setattr(cfg, "get_registry_token", lambda d: "test-token")

            ref = {
                "uri": "taos://proj/files/hello.txt",
                "sha256": hashlib.sha256(b"hello world").hexdigest(),
            }

            # ---- Part 1: with _NoRedirect (the fixed behaviour) ----
            # The real fetch path uses _NoRedirect which returns None from
            # redirect_request, preventing the 302 from being followed.
            # Origin B must NOT receive the Authorization header.
            auth_leaked = False
            try:
                asyncio.run(
                    svc.fetch_by_ref(ref, agent="test", data_dir="/tmp")
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 302:
                    # Expected: _NoRedirect prevents following the 302,
                    # so the HTTPError propagates and origin B is never contacted.
                    pass
                else:
                    raise

            # Assertion: origin B received NO Authorization header
            auth_keys = [k for k in recorded["headers"] if k.lower() == "authorization"]
            assert len(auth_keys) == 0, (
                f"Authorization header leaked to origin B with _NoRedirect! "
                f"Keys found: {auth_keys}"
            )

            # ---- Part 2: control without _NoRedirect ----
            # Build an opener without _NoRedirect (default HTTPRedirectHandler
            # follows redirects). Origin B MUST receive the Authorization header.
            headers = {"Accept": "application/octet-stream", "Authorization": "Bearer test-token"}
            req = urllib.request.Request(
                f"http://localhost:{a_port}/api/projects/proj/files/hello.txt",
                headers=headers,
                method="GET",
            )
            # Default opener follows redirects -> Authorization leaks to origin B
            opener = urllib.request.build_opener()
            try:
                with opener.open(req, timeout=30) as resp:
                    _ = resp.read()
            except urllib.error.HTTPError:
                # Even if an HTTPError occurs, the redirect was followed and
                # the Authorization header was already sent to origin B via
                # the separate TCP connection recorded below.
                pass

            # Assertion: origin B received the Authorization header
            auth_keys = [k for k in recorded["headers"] if k.lower() == "authorization"]
            assert len(auth_keys) > 0, (
                "Control: Authorization header should have been leaked to origin B "
                "without _NoRedirect!"
            )

        finally:
            b_server.shutdown()
            b_server.server_close()
            a_server.shutdown()
            a_server.server_close()

# ---------------------------------------------------------------------------
# Single-controller fallback tests
# ---------------------------------------------------------------------------

class TestGetFilesUrlFallback:
    def test_files_url_preferred_over_registry(self, monkeypatch):
        from taosmd import config as cfg
        monkeypatch.setattr(cfg, "get_files_url", lambda data_dir=None: "http://files:8000")
        monkeypatch.setattr(cfg, "get_registry_url", lambda data_dir=None: "http://reg:8000")
        from taosmd.ref_fetch import _get_files_url
        assert _get_files_url() == "http://files:8000"

    def test_registry_url_fallback_when_files_url_unset(self, monkeypatch):
        from taosmd import config as cfg
        monkeypatch.setattr(cfg, "get_files_url", lambda data_dir=None: None)
        monkeypatch.setattr(cfg, "get_registry_url", lambda data_dir=None: "http://reg:8000")
        from taosmd.ref_fetch import _get_files_url
        assert _get_files_url() == "http://reg:8000"

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("TAOSMD_FILES_URL", "http://env-files:8000")
        from taosmd.ref_fetch import _get_files_url
        assert _get_files_url() == "http://env-files:8000"

    def test_config_files_url_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TAOSMD_FILES_URL", raising=False)
        config.set_files_url("http://files:9000", data_dir=str(tmp_path))
        assert config.get_files_url(str(tmp_path)) == "http://files:9000"

    def test_config_files_url_clear(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TAOSMD_FILES_URL", raising=False)
        config.set_files_url("http://files:9000", data_dir=str(tmp_path))
        config.set_files_url("", clear=True, data_dir=str(tmp_path))
        assert config.get_files_url(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Service remote routing tests
# ---------------------------------------------------------------------------

class TestServiceFetchByRefRouting:
    def test_routes_to_remote_when_configured(self, monkeypatch):
        from taosmd import service as svc

        class FakeRemote:
            async def fetch_by_ref(self, ref, agent, **kw):
                return {"bytes": "dGVzdA==", "sha256": "abc", "size": 4}

        fake_remote = FakeRemote()
        monkeypatch.setattr(svc, "_get_remote", lambda data_dir: fake_remote)

        ref = {"uri": "taos://proj/files/hello.txt", "sha256": "abc"}
        result = asyncio.run(svc.fetch_by_ref(ref, agent="test"))
        assert result["bytes"] == "dGVzdA=="

    def test_local_path_when_no_remote(self, monkeypatch):
        from taosmd import config as cfg
        from taosmd import service as svc
        from taosmd.ref_fetch import fetch_by_ref as _orig_fetch

        async def fake_fetch(ref, fetcher, agent, data_dir=None):
            return b"hello"

        monkeypatch.setattr(svc, "_get_remote", lambda data_dir: None)
        monkeypatch.setattr(cfg, "get_files_url", lambda data_dir: "http://ctrl:8000")
        monkeypatch.setattr("taosmd.ref_fetch.fetch_by_ref", fake_fetch)

        ref = {"uri": "taos://proj/files/hello.txt", "sha256": "abc"}
        result = asyncio.run(svc.fetch_by_ref(ref, agent="test", data_dir="/tmp"))
        assert result["bytes"] == "aGVsbG8="

    def test_local_path_passes_data_dir_to_ref_fetch(self, monkeypatch):
        from taosmd import config as cfg
        from taosmd import service as svc

        captured = {}

        async def fake_fetch(ref, fetcher, agent, data_dir=None):
            captured["data_dir"] = data_dir
            return b"hello"

        monkeypatch.setattr(svc, "_get_remote", lambda data_dir: None)
        monkeypatch.setattr(cfg, "get_files_url", lambda data_dir=None: "http://ctrl:8000")
        monkeypatch.setattr("taosmd.ref_fetch.fetch_by_ref", fake_fetch)

        ref = {"uri": "taos://proj/files/hello.txt", "sha256": "abc"}
    def test_service_fetch_by_ref_does_not_raise_when_files_url_unset_but_registry_set(self, tmp_path, monkeypatch):
        from taosmd import config as cfg
        from taosmd import service as svc

        async def fake_fetch(ref, fetcher, agent, data_dir=None):
            return b"hello"

        monkeypatch.setattr(svc, "_get_remote", lambda data_dir: None)
        monkeypatch.setattr("taosmd.ref_fetch.fetch_by_ref", fake_fetch)

        config.set_files_url("", clear=True, data_dir=str(tmp_path))
        config.set_registry_url("http://reg:8000", data_dir=str(tmp_path))

        ref = {"uri": "taos://proj/files/hello.txt", "sha256": hashlib.sha256(b"hello").hexdigest()}
        result = asyncio.run(svc.fetch_by_ref(ref, agent="test", data_dir=str(tmp_path)))
        assert result["bytes"] == "aGVsbG8="
