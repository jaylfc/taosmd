"""Tests for taosmd.ref_fetch: resolver + verified fetch helper."""
from __future__ import annotations

import asyncio
import hashlib
import os

import pytest

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

    def test_fetch_mismatch_raises_hash_error_bytes_not_returned(self, monkeypatch):
        content = b"hello world"
        ref = {"uri": "taos://proj/files/hello.txt", "sha256": "deadbeef" * 8}

        monkeypatch.setenv("TAOSMD_FILES_URL", "http://ctrl:8000")

        def fake_fetcher(url, agent):
            return content

        with pytest.raises(HashMismatchError, match="sha256 mismatch"):
            asyncio.run(fetch_by_ref(ref, fake_fetcher, "test-agent"))

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
