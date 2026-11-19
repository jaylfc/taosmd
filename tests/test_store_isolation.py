"""Tests for store isolation: a broken vector store must not take down the
archive, the A2A bus, or the health endpoint.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server


def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder — no ONNX/QMD model required."""
    vmem = stores.get("vector")
    if vmem is None:
        return

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _post(url: str, payload) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _make_broken_vector_store(data_dir: Path) -> None:
    """Create a vector-memory.db with a conflicting mode marker.

    This simulates a store built with a different embedder (e.g. arctic-embed-s
    when the server expects minilm-onnx). The mode marker uses the same format
    as _store_mode_signature but with a different embedder value.
    """
    db_path = data_dir / "vector-memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    # Write a mode marker with a conflicting embedder
    conn.execute(
        "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', ?)",
        ("late_interaction=0;binary_quant=0;dense_format=f32;embedder=arctic-embed-s",),
    )
    conn.commit()
    conn.close()


def _make_broken_archive(data_dir: Path) -> None:
    """Make the archive path a regular file so it cannot be opened as a directory."""
    archive_dir = data_dir / "archive"
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_dir.write_text("not a directory")


class TestStoreIsolation:
    """Tests that a broken vector store does not cascade to other components."""

    @pytest.fixture
    def broken_vector_server(self, tmp_path, monkeypatch):
        """HTTP server with a pre-broken vector store."""
        data_dir = tmp_path / "taosmd-broken"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        # Note: we do NOT call _ensure_stores here; the broken store should be
        # encountered on first request.

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            yield f"http://{host}:{port}", data_dir, httpd
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    @pytest.fixture
    def broken_archive_server(self, tmp_path, monkeypatch):
        """HTTP server with a pre-broken archive (path is a file)."""
        data_dir = tmp_path / "taosmd-broken-archive"
        data_dir.mkdir()
        _make_broken_archive(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            yield f"http://{host}:{port}", data_dir, httpd
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    @pytest.fixture
    def healthy_server(self, tmp_path, monkeypatch):
        """Normal healthy server for baseline comparison."""
        data_dir = tmp_path / "taosmd-healthy"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            yield f"http://{host}:{port}", data_dir, httpd
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

    def test_bus_survives_broken_vector_store(self, broken_vector_server):
        """A2A send and read both return 200; /health shows degraded with vector error."""
        base_url, data_dir, httpd = broken_vector_server

        # POST /a2a/send should succeed (archive works)
        status, body = _post(f"{base_url}/a2a/send", {
            "from": "test-agent",
            "body": "hello from broken vector store",
            "thread": "test-thread",
        })
        assert status == 200, f"a2a/send failed: {body}"

        # GET /a2a/messages should return the message
        status, body = _get(f"{base_url}/a2a/messages?thread=test-thread")
        assert status == 200, f"a2a/messages failed: {body}"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["body"] == "hello from broken vector store"

        # GET /health should show degraded with vector error
        status, body = _get(f"{base_url}/health")
        assert status == 200, body
        assert body["status"] == "degraded"
        assert "stores" in body
        assert body["stores"]["archive"] == "ok"
        assert body["stores"]["vector"].startswith("error:")
        assert "StoreModeMismatch" in body["stores"]["vector"]

    def test_ingest_archives_raw_when_vector_broken(self, broken_vector_server):
        """POST /ingest archives the raw text and reports degraded; no 500."""
        base_url, data_dir, httpd = broken_vector_server

        status, body = _post(f"{base_url}/ingest", {
            "text": "this should be archived even with broken vector",
            "agent": "test-agent",
        })
        assert status == 200, f"ingest failed: {body}"
        assert body["archived"] == 1
        assert body["degraded"] is True
        assert body["vector_failures"] == 1

        # Verify the text is in the archive JSONL
        archive_files = list((data_dir / "archive").rglob("*.jsonl*"))
        assert archive_files, "no archive file created"
        import gzip
        if archive_files[0].suffix == ".gz":
            with gzip.open(archive_files[0], "rt") as f:
                lines = f.readlines()
        else:
            with open(archive_files[0]) as f:
                lines = f.readlines()
        found = any("this should be archived even with broken vector" in line for line in lines)
        assert found, "ingested text not found in archive"

    def test_search_reports_vector_unavailable(self, broken_vector_server):
        """POST /search returns a clear error naming the vector store."""
        base_url, data_dir, httpd = broken_vector_server

        status, body = _post(f"{base_url}/search", {
            "query": "anything",
            "agent": "test-agent",
        })
        # Should be a 4xx/5xx with a clear message about vector store
        assert status >= 400, f"expected error status, got {status}: {body}"
        assert "vector" in str(body).lower()
        assert "StoreModeMismatch" in str(body) or "unavailable" in str(body).lower()

    def test_vector_store_recovers_without_restart(self, tmp_path, monkeypatch):
        """After repairing the store and advancing cooldown, /health shows ok."""
        data_dir = tmp_path / "taosmd-recovery"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        # Don't clear _stores_cache - let it cache the broken state so the
        # cache-hit retry path (Finding 1) is exercised through the real HTTP server.
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        try:
            # First request: health shows degraded
            status, body = _get(f"http://{host}:{port}/health")
            assert status == 200
            assert body["status"] == "degraded"
            assert body["stores"]["vector"].startswith("error:")

            # Repair the store: delete the conflicting mode marker
            db_path = data_dir / "vector-memory.db"
            conn = sqlite3.connect(db_path)
            conn.execute("DELETE FROM store_meta WHERE key='mode'")
            conn.commit()
            conn.close()

            # Advance the retry clock past 30s so _ensure_stores retries the vector store
            import time
            base_time = time.time()

            def mock_time():
                return base_time + 31

            with patch("taosmd.api.time.time", side_effect=mock_time):
                # Second request: should recover without restart
                status, body = _get(f"http://{host}:{port}/health")
                assert status == 200
                assert body["status"] == "ok"
                assert body["stores"]["vector"] == "ok"

                # Search should also succeed
                status, body = _post(f"http://{host}:{port}/search", {
                    "query": "anything",
                    "agent": "test-agent",
                })
                assert status < 400, f"expected success, got {status}: {body}"
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_archive_failure_still_raises(self, broken_archive_server):
        """Broken archive must fail loudly (5xx), not silently skip raw write."""
        base_url, data_dir, httpd = broken_archive_server

        # Any request that needs the archive should fail
        status, body = _post(f"{base_url}/a2a/send", {
            "from": "test-agent",
            "body": "this should fail",
            "thread": "test-thread",
        })
        assert status >= 500, f"expected 5xx for broken archive, got {status}: {body}"

    def test_healthy_server_unchanged(self, healthy_server):
        """Normal server: /health status ok, all three stores ok."""
        base_url, data_dir, httpd = healthy_server

        status, body = _get(f"{base_url}/health")
        assert status == 200
        assert body["status"] == "ok"
        assert body["stores"]["archive"] == "ok"
        assert body["stores"]["vector"] == "ok"
        assert body["stores"]["kg"] == "ok"
        assert "version" in body
        assert "capabilities" in body
        assert isinstance(body["capabilities"], list)


# Unit tests for the _ensure_stores retry logic
class TestEnsureStoresRetry:
    """Unit tests for the 30-second retry cooldown in _ensure_stores."""

    @pytest.mark.asyncio
    async def test_vector_store_retry_after_30_seconds(self, tmp_path, monkeypatch):
        """After 30s, a previously failed vector store is retried."""
        from unittest.mock import patch

        data_dir = tmp_path / "retry-test"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        # First call: vector store fails
        stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is None
        assert stores["archive"] is not None

        # Repair the store
        db_path = data_dir / "vector-memory.db"
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM store_meta WHERE key='mode'")
        conn.commit()
        conn.close()

        # Mock time to advance past 30s
        import time
        base_time = time.time()

        def mock_time():
            return base_time + 31

        with patch("taosmd.api.time.time", side_effect=mock_time):
            # Clear cache to force retry
            taosmd_api._stores_cache.clear()
            stores = await taosmd_api._ensure_stores(str(data_dir))
            assert stores["vector"] is not None
            assert stores["archive"] is not None

    @pytest.mark.asyncio
    async def test_vector_store_no_retry_before_30_seconds(self, tmp_path, monkeypatch):
        """Before 30s, a previously failed vector store is NOT retried."""
        from unittest.mock import patch

        data_dir = tmp_path / "retry-test-2"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        # First call: vector store fails
        stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is None

        # Repair the store
        db_path = data_dir / "vector-memory.db"
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM store_meta WHERE key='mode'")
        conn.commit()
        conn.close()

        # Mock time to advance only 10 seconds
        base_time = time.time()

        def mock_time():
            return base_time + 10

        with patch("taosmd.api.time.time", side_effect=mock_time):
            taosmd_api._stores_cache.clear()
            stores = await taosmd_api._ensure_stores(str(data_dir))
            assert stores["vector"] is None  # Still None, no retry yet
