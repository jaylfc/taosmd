"""Tests for store isolation: a broken vector store must not take down the
archive, the A2A bus, or the health endpoint.
"""

from __future__ import annotations

import asyncio
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


def _post(url: str, payload, timeout: int = 60) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _get(url: str, timeout: int = 60) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _wait_for_health(base_url: str, timeout: int = 60, interval: float = 0.2) -> dict:
    """Poll /health until it returns 200, or timeout expires.

    Returns the health response body.
    """
    deadline = time.time() + timeout
    last_body = None
    while time.time() < deadline:
        try:
            status, body = _get(f"{base_url}/health", timeout=5)
            if status == 200:
                return body
            last_body = body
        except Exception:
            pass
        time.sleep(interval)
    # Final attempt to get a body for the error message
    try:
        _, last_body = _get(f"{base_url}/health", timeout=5)
    except Exception:
        pass
    raise AssertionError(f"/health did not return 200 within {timeout}s; last: {last_body}")


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


def _make_broken_kg_store(data_dir: Path) -> None:
    """Create a knowledge-graph.db that fails to open (corrupt schema)."""
    db_path = data_dir / "knowledge-graph.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Create a table with wrong schema to trigger init failure
    conn.execute("CREATE TABLE kg_triples (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()


def _break_vector_store_differently(data_dir: Path) -> str:
    """Replace vector-memory.db with a non-SQLite file so init fails with a
    different error than the mode-mismatch break. Returns expected error fragment."""
    db_path = data_dir / "vector-memory.db"
    if db_path.exists():
        db_path.unlink()
    # Also remove WAL/SHM sidecars if present
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    db_path.write_bytes(b"not a sqlite database\n")
    return "sqlite3.DatabaseError"


def _break_kg_store_differently(data_dir: Path) -> str:
    """Replace knowledge-graph.db with a non-SQLite file so init fails with a
    different error than the schema break. Returns expected error fragment."""
    db_path = data_dir / "knowledge-graph.db"
    if db_path.exists():
        db_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    db_path.write_bytes(b"not a sqlite database\n")
    return "sqlite3.DatabaseError"


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
    def broken_kg_server(self, tmp_path, monkeypatch):
        """HTTP server with a pre-broken KG store."""
        data_dir = tmp_path / "taosmd-broken-kg"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)

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


class TestKGStoreIsolation:
    """Tests for KG store isolation: a broken KG store must not take down
    the archive, vector store, or health endpoint.
    """

    @pytest.fixture
    def broken_kg_server_ready(self, tmp_path, monkeypatch):
        """HTTP server with a pre-broken KG store, waited for /health ready."""
        data_dir = tmp_path / "taosmd-broken-kg"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        # Initialize stores and patch embedder so vector works
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            base_url = f"http://{host}:{port}"
            _wait_for_health(base_url)
            yield base_url, data_dir, httpd
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_health_shows_kg_degraded(self, broken_kg_server_ready):
        """GET /health shows degraded with kg error text."""
        base_url, data_dir, httpd = broken_kg_server_ready

        status, body = _get(f"{base_url}/health")
        assert status == 200
        assert body["status"] == "degraded"
        assert body["stores"]["archive"] == "ok"
        assert body["stores"]["vector"] == "ok"
        assert body["stores"]["kg"].startswith("error:")

    def test_a2a_survives_broken_kg(self, broken_kg_server_ready):
        """A2A send and read both return 200 when KG is broken."""
        base_url, data_dir, httpd = broken_kg_server_ready

        status, body = _post(f"{base_url}/a2a/send", {
            "from": "test-agent",
            "body": "hello from broken kg store",
            "thread": "test-thread",
        })
        assert status == 200, f"a2a/send failed: {body}"

        status, body = _get(f"{base_url}/a2a/messages?thread=test-thread")
        assert status == 200
        assert len(body["messages"]) == 1

    def test_ingest_survives_broken_kg(self, broken_kg_server_ready):
        """POST /ingest works when KG is broken (archive + vector ok)."""
        base_url, data_dir, httpd = broken_kg_server_ready

        status, body = _post(f"{base_url}/ingest", {
            "text": "this should be archived even with broken kg",
            "agent": "test-agent",
        })
        assert status == 200
        assert body["archived"] == 1
        # No degraded flag since vector works
        assert body.get("degraded") is not True

    def test_search_fails_with_kg_unavailable(self, broken_kg_server_ready):
        """POST /search raises StoreUnavailableError when KG is down."""
        base_url, data_dir, httpd = broken_kg_server_ready

        status, body = _post(f"{base_url}/search", {
            "query": "anything",
            "agent": "test-agent",
        })
        assert status >= 400
        assert "kg" in str(body).lower()
        assert "unavailable" in str(body).lower()

    def test_graph_fails_with_kg_unavailable(self, broken_kg_server_ready):
        """GET /graph raises StoreUnavailableError when KG is down."""
        base_url, data_dir, httpd = broken_kg_server_ready

        status, body = _get(f"{base_url}/graph")
        assert status >= 400
        assert "kg" in str(body).lower()
        assert "unavailable" in str(body).lower()

    def test_graph_activations_fails_with_kg_unavailable(self, broken_kg_server_ready):
        """GET /graph/activations raises StoreUnavailableError when KG is down."""
        base_url, data_dir, httpd = broken_kg_server_ready

        status, body = _get(f"{base_url}/graph/activations")
        assert status >= 400
        assert "kg" in str(body).lower()
        assert "unavailable" in str(body).lower()

    def test_reconcile_fails_with_kg_unavailable(self, tmp_path, monkeypatch):
        """taosmd.api.reconcile raises StoreUnavailableError when KG is down."""
        data_dir = tmp_path / "taosmd-reconcile-kg"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        async def _test():
            with pytest.raises(taosmd_api.StoreUnavailableError, match="kg store unavailable"):
                await taosmd_api.reconcile(agent="test-agent", data_dir=str(data_dir))

        asyncio.run(_test())

    def test_reindex_fails_with_kg_unavailable(self, tmp_path, monkeypatch):
        """taosmd.api.reindex raises StoreUnavailableError when KG is down."""
        data_dir = tmp_path / "taosmd-reindex-kg"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        async def _test():
            with pytest.raises(taosmd_api.StoreUnavailableError, match="kg store unavailable"):
                await taosmd_api.reindex(agent="test-agent", data_dir=str(data_dir))

        asyncio.run(_test())

    def test_supersede_vectors_fails_with_kg_unavailable(self, tmp_path, monkeypatch):
        """taosmd.api.supersede_vectors raises StoreUnavailableError when KG is down."""
        data_dir = tmp_path / "taosmd-supersede-kg"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        async def _test():
            with pytest.raises(taosmd_api.StoreUnavailableError, match="kg store unavailable"):
                await taosmd_api.supersede_vectors("match", data_dir=str(data_dir))

        asyncio.run(_test())

    def test_kg_no_retry_before_cooldown(self, tmp_path, monkeypatch):
        """Before 30s, a previously failed KG store is NOT retried."""
        data_dir = tmp_path / "taosmd-kg-retry"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        async def _first():
            stores = await taosmd_api._ensure_stores(str(data_dir))
            assert stores["kg"] is None
            return stores

        stores = asyncio.run(_first())
        assert stores["kg"] is None
        assert stores["vector"] is not None
        assert stores["archive"] is not None

        # Repair the store: create a valid KG db
        db_path = data_dir / "knowledge-graph.db"
        import os
        os.remove(db_path)
        # The next _ensure_stores call will create a fresh valid one

        # Mock time to advance only 10 seconds (within cooldown)
        base_time = time.time()

        def mock_time():
            return base_time + 10

        with patch("taosmd.api.time.time", side_effect=mock_time):
            taosmd_api._stores_cache.clear()
            stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
            assert stores["kg"] is None  # Still None, no retry yet

    def test_kg_retry_after_cooldown(self, tmp_path, monkeypatch):
        """After 30s, a previously failed KG store is retried and succeeds."""
        data_dir = tmp_path / "taosmd-kg-retry-2"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        async def _first():
            stores = await taosmd_api._ensure_stores(str(data_dir))
            assert stores["kg"] is None
            return stores

        asyncio.run(_first())

        # Repair the store: remove corrupt db
        db_path = data_dir / "knowledge-graph.db"
        import os
        os.remove(db_path)

        # Mock time to advance past 30s
        base_time = time.time()

        def mock_time():
            return base_time + 31

        with patch("taosmd.api.time.time", side_effect=mock_time):
            taosmd_api._stores_cache.clear()
            stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
            assert stores["kg"] is not None  # Retried and succeeded
            assert stores["vector"] is not None
            assert stores["archive"] is not None

    def test_kg_recovery_via_health(self, tmp_path, monkeypatch):
        """After repairing KG store and advancing cooldown, /health shows ok."""
        data_dir = tmp_path / "taosmd-kg-recovery"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        try:
            base_url = f"http://{host}:{port}"
            # First request: health shows degraded
            body = _wait_for_health(base_url)
            assert body["status"] == "degraded"
            assert body["stores"]["kg"].startswith("error:")

            # Repair the store: remove corrupt db
            db_path = data_dir / "knowledge-graph.db"
            import os
            os.remove(db_path)

            # Advance the retry clock past 30s
            base_time = time.time()

            def mock_time():
                return base_time + 31

            with patch("taosmd.api.time.time", side_effect=mock_time):
                # Second request: should recover without restart
                body = _wait_for_health(base_url)
                assert body["status"] == "ok"
                assert body["stores"]["kg"] == "ok"

                # Search should also succeed
                status, body = _post(f"{base_url}/search", {
                    "query": "anything",
                    "agent": "test-agent",
                })
                assert status < 400, f"expected success, got {status}: {body}"
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()


class TestVectorStoreGuardGaps:
    """Tests for vector store guard gaps: ingest_batch, cache-hit cooldown,
    shelf_archive, shelf_unarchive, collections.ingest_folder.
    """

    @pytest.fixture
    def broken_vector_server_ready(self, tmp_path, monkeypatch):
        """HTTP server with a pre-broken vector store, waited for /health ready."""
        data_dir = tmp_path / "taosmd-broken-vec"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            base_url = f"http://{host}:{port}"
            _wait_for_health(base_url)
            yield base_url, data_dir, httpd
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_ingest_batch_reports_degraded_with_vector_unavailable(self, broken_vector_server_ready):
        """POST /ingest/batch reports degraded (not error) when vector store is down.

        This matches the ingest behavior: archive succeeds, vector failures
        are reported via degraded flag and vector_failures count.
        """
        base_url, data_dir, httpd = broken_vector_server_ready

        status, body = _post(f"{base_url}/ingest/batch", {
            "items": [{"text": "test item", "id": "item-1"}],
            "agent": "test-agent",
        })
        assert status == 200, f"ingest/batch failed: {body}"
        assert body["ingested"] == 1
        assert body["degraded"] is True
        assert body["vector_failures"] == 1

    def test_cache_hit_vector_cooldown(self, tmp_path, monkeypatch):
        """Cache-hit path: no retry before cooldown, retry after cooldown.

        This test does NOT clear _stores_cache between calls, exercising the
        cache-hit retry logic in _ensure_stores.
        """
        data_dir = tmp_path / "taosmd-cache-hit"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        # First call: vector store fails, cached
        stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
        assert stores["vector"] is None
        assert stores["archive"] is not None
        assert stores["kg"] is not None

        # Repair the store
        db_path = data_dir / "vector-memory.db"
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM store_meta WHERE key='mode'")
        conn.commit()
        conn.close()

        # Mock time to advance only 10 seconds (within cooldown)
        base_time = time.time()

        def mock_time():
            return base_time + 10

        with patch("taosmd.api.time.time", side_effect=mock_time):
            # Second call WITHOUT clearing cache: should NOT retry
            stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
            assert stores["vector"] is None  # Still None, no retry yet (cache hit)

        # Now advance past 30s
        def mock_time_2():
            return base_time + 31

        with patch("taosmd.api.time.time", side_effect=mock_time_2):
            # Third call WITHOUT clearing cache: should retry and succeed
            stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
            assert stores["vector"] is not None  # Retried and succeeded

    def test_shelf_archive_fails_with_vector_unavailable(self, tmp_path, monkeypatch):
        """admin.shelf_archive raises StoreUnavailableError when vector store is down."""
        from taosmd import admin

        data_dir = tmp_path / "taosmd-shelf-archive"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        # Create a shelf first (needs healthy stores)
        # We need to temporarily fix the vector store to create the shelf
        # But the test is about shelf_archive failing when vector is down
        # So we create the shelf, then break the vector store
        async def _setup():
            # Create shelf with healthy stores
            # First ensure stores with a fresh cache
            taosmd_api._stores_cache.clear()
            # Create a valid vector store temporarily
            stores = await taosmd_api._ensure_stores(str(data_dir))
            # Now break it by replacing the db
            db_path = Path(data_dir) / "vector-memory.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', ?)",
                ("late_interaction=0;binary_quant=0;dense_format=f32;embedder=arctic-embed-s",),
            )
            conn.commit()
            conn.close()

            # Create shelf
            await admin.shelf_create("testshelf", data_dir=data_dir)
            return stores

        stores = asyncio.run(_setup())

        # Now shelf_archive should fail
        async def _test():
            with pytest.raises(taosmd_api.StoreUnavailableError, match="vector store unavailable"):
                await admin.shelf_archive("testshelf", data_dir=data_dir, stores=stores)

        asyncio.run(_test())

    def test_shelf_unarchive_fails_with_vector_unavailable(self, tmp_path, monkeypatch):
        """admin.shelf_unarchive raises StoreUnavailableError when vector store is down."""
        from taosmd import admin

        data_dir = tmp_path / "taosmd-shelf-unarchive"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})

        async def _setup():
            taosmd_api._stores_cache.clear()
            stores = await taosmd_api._ensure_stores(str(data_dir))
            # Break vector store
            db_path = Path(data_dir) / "vector-memory.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', ?)",
                ("late_interaction=0;binary_quant=0;dense_format=f32;embedder=arctic-embed-s",),
            )
            conn.commit()
            conn.close()

            await admin.shelf_create("testshelf", data_dir=data_dir)
            # Archive it first (with healthy stores before break)
            # Actually we need to archive after breaking... but archive needs vector
            # This tests unarchive with broken vector
            return stores

        stores = asyncio.run(_setup())

        async def _test():
            with pytest.raises(taosmd_api.StoreUnavailableError, match="vector store unavailable"):
                await admin.shelf_unarchive("testshelf", data_dir=data_dir, stores=stores)

        asyncio.run(_test())

    def test_collections_ingest_folder_fails_with_vector_unavailable(self, tmp_path, monkeypatch):
        """collections.ingest_folder raises StoreUnavailableError when vector store is down and files changed."""
        from taosmd import config
        from taosmd.collections import CollectionStore, ingest_folder

        data_dir = tmp_path / "taosmd-coll-ingest"
        data_dir.mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Hello\n\nContent")

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        config.set_collections_allowed_roots([str(source_dir)], data_dir=str(data_dir))

        _make_broken_vector_store(data_dir)

        async def _test():
            store = CollectionStore(str(data_dir))
            col = store.create(name="docs", kind="docs", source_path=str(source_dir))

            with pytest.raises(taosmd_api.StoreUnavailableError, match="vector store unavailable"):
                await ingest_folder(col["id"], data_dir=str(data_dir))

        asyncio.run(_test())


class TestCollectionsUnchangedNoOp:
    """Test for the judgement call: collections.ingest_folder on an UNCHANGED
    collection should succeed as a no-op when the vector store is down.
    """

    @pytest.fixture
    def setup_unchanged_collection(self, tmp_path, monkeypatch):
        """Create a collection and index it, then break the vector store."""
        from taosmd import config
        from taosmd.collections import CollectionStore, ingest_folder

        data_dir = tmp_path / "taosmd-coll-unchanged"
        data_dir.mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Hello\n\nContent")

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        config.set_collections_allowed_roots([str(source_dir)], data_dir=str(data_dir))

        # First, index the collection with a healthy vector store
        store = CollectionStore(str(data_dir))
        col = store.create(name="docs", kind="docs", source_path=str(source_dir))

        # Patch embedder and ingest
        stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
        vmem = stores["vector"]

        async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
            h = hash(text) & 0xFFFFFFFF
            return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

        vmem.embed = _fake_embed  # type: ignore[assignment]

        asyncio.run(ingest_folder(col["id"], data_dir=str(data_dir)))
        assert store.get(col["id"])["status"] == "ready"

        # Now break the vector store AND clear the cache so the next call sees it
        _make_broken_vector_store(data_dir)
        taosmd_api._stores_cache.clear()

        return data_dir, source_dir, col["id"], store

    def test_unchanged_collection_succeeds_as_noop(self, setup_unchanged_collection):
        """Unchanged collection: ingest_folder succeeds as no-op when vector store down."""
        from taosmd.collections import ingest_folder

        data_dir, source_dir, col_id, store = setup_unchanged_collection

        # Do NOT change any files - collection is unchanged
        stats = asyncio.run(ingest_folder(col_id, data_dir=str(data_dir)))

        # Should succeed with no files indexed
        assert stats["files_indexed"] == 0
        assert stats["files_unchanged"] == 1
        assert stats["chunks_ingested"] == 0
        assert store.get(col_id)["status"] == "ready"

    def test_changed_collection_fails_with_vector_unavailable(self, setup_unchanged_collection):
        """Changed collection: ingest_folder raises StoreUnavailableError when vector store down."""
        from taosmd.collections import ingest_folder

        data_dir, source_dir, col_id, store = setup_unchanged_collection

        # Change a file
        (source_dir / "doc.md").write_text("# Hello\n\nChanged content")

        # Should fail because vector store is down and there are changes
        with pytest.raises(taosmd_api.StoreUnavailableError, match="vector store unavailable"):
            asyncio.run(ingest_folder(col_id, data_dir=str(data_dir)))

        # Status should be error
        assert store.get(col_id)["status"] == "error"


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


# Readiness wait helper tests
class TestReadinessWait:
    """Tests for the _wait_for_health helper."""

    def test_wait_for_health_succeeds_on_healthy_server(self, tmp_path, monkeypatch):
        """_wait_for_health returns quickly for a healthy server."""
        data_dir = tmp_path / "taosmd-healthy-wait"
        data_dir.mkdir()
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
        _patch_embedder(stores)

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        try:
            base_url = f"http://{host}:{port}"
            body = _wait_for_health(base_url, timeout=10)
            assert body["status"] == "ok"
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    def test_wait_for_health_succeeds_on_degraded_server(self, tmp_path, monkeypatch):
        """_wait_for_health returns for a degraded server (vector broken)."""
        data_dir = tmp_path / "taosmd-degraded-wait"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))

        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        try:
            base_url = f"http://{host}:{port}"
            body = _wait_for_health(base_url, timeout=10)
            assert body["status"] == "degraded"
            assert body["stores"]["vector"].startswith("error:")
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

class TestCacheHitFailureBranches:
    """Pin the 15 mutants surviving #758: cache-hit failure branches,
    vector guards in reconcile/reindex/supersede_vectors, and collections
    has_changes. Real store failures only; no mocks of the function under test.
    """

    @pytest.mark.asyncio
    async def test_cache_hit_vector_failure_records_error_and_timestamp(self, tmp_path, monkeypatch):
        """Cache-hit vector retry that FAILS must record new error and timestamp."""
        data_dir = tmp_path / "vec-cache-fail"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is None
        resolved = str(data_dir)
        first_err = taosmd_api._store_errors[resolved]["vector"]
        first_ts = taosmd_api._last_attempt[resolved]["vector"]
        assert "StoreModeMismatch" in first_err

        # Repair then break differently so the retry fails with a NEW error.
        _break_vector_store_differently(data_dir)

        base_time = time.time()

        def after_cooldown():
            return base_time + 31

        with patch("taosmd.api.time.time", side_effect=after_cooldown):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is None
        second_err = taosmd_api._store_errors[resolved]["vector"]
        second_ts = taosmd_api._last_attempt[resolved]["vector"]
        assert second_err != first_err
        assert "sqlite3.DatabaseError" in second_err or "DatabaseError" in second_err or "database" in second_err.lower()
        assert second_ts > first_ts

    @pytest.mark.asyncio
    async def test_cache_hit_vector_failure_clears_error_on_recovery(self, tmp_path, monkeypatch):
        """Cache-hit vector retry that fails then succeeds must clear error."""
        data_dir = tmp_path / "vec-cache-recover"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is None
        resolved = str(data_dir)
        assert "vector" in taosmd_api._store_errors.get(resolved, {})

        _break_vector_store_differently(data_dir)
        base_time = time.time()
        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 31):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is None
        assert "vector" in taosmd_api._store_errors[resolved]

        # Repair for real: remove the broken file so a fresh store can init.
        db_path = data_dir / "vector-memory.db"
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()

        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 62):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is not None
        assert "vector" not in taosmd_api._store_errors.get(resolved, {})

    @pytest.mark.asyncio
    async def test_cache_hit_vector_failure_via_http_health(self, tmp_path, monkeypatch):
        """HTTP /health reflects cache-hit vector failure with new error/timestamp, then recovery."""
        data_dir = tmp_path / "vec-cache-http"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            base_url = f"http://{host}:{port}"
            body = _wait_for_health(base_url)
            assert body["status"] == "degraded"
            first_err = body["stores"]["vector"]
            assert "StoreModeMismatch" in first_err

            _break_vector_store_differently(data_dir)
            base_time = time.time()
            with patch("taosmd.api.time.time", side_effect=lambda: base_time + 31):
                body = _wait_for_health(base_url)
            assert body["status"] == "degraded"
            second_err = body["stores"]["vector"]
            assert second_err.startswith("error:")
            assert second_err != first_err
            assert "StoreModeMismatch" not in second_err

            db_path = data_dir / "vector-memory.db"
            db_path.unlink()
            for suffix in ("-wal", "-shm"):
                p = Path(str(db_path) + suffix)
                if p.exists():
                    p.unlink()
            with patch("taosmd.api.time.time", side_effect=lambda: base_time + 62):
                body = _wait_for_health(base_url)
            assert body["status"] == "ok"
            assert body["stores"]["vector"] == "ok"
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    @pytest.mark.asyncio
    async def test_cache_hit_kg_failure_records_error_and_timestamp(self, tmp_path, monkeypatch):
        """Cache-hit KG retry that FAILS must record new error and timestamp."""
        data_dir = tmp_path / "kg-cache-fail"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["kg"] is None
        resolved = str(data_dir)
        first_err = taosmd_api._store_errors[resolved]["kg"]
        first_ts = taosmd_api._last_attempt[resolved]["kg"]

        _break_kg_store_differently(data_dir)
        base_time = time.time()
        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 31):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["kg"] is None
        second_err = taosmd_api._store_errors[resolved]["kg"]
        second_ts = taosmd_api._last_attempt[resolved]["kg"]
        assert second_err != first_err
        assert "sqlite3.DatabaseError" in second_err or "DatabaseError" in second_err or "database" in second_err.lower()
        assert second_ts > first_ts

    @pytest.mark.asyncio
    async def test_cache_hit_kg_failure_clears_error_on_recovery(self, tmp_path, monkeypatch):
        """Cache-hit KG retry that fails then succeeds must clear error."""
        data_dir = tmp_path / "kg-cache-recover"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["kg"] is None
        resolved = str(data_dir)
        assert "kg" in taosmd_api._store_errors.get(resolved, {})

        _break_kg_store_differently(data_dir)
        base_time = time.time()
        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 31):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["kg"] is None
        assert "kg" in taosmd_api._store_errors[resolved]

        db_path = data_dir / "knowledge-graph.db"
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()

        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 62):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["kg"] is not None
        assert "kg" not in taosmd_api._store_errors.get(resolved, {})

    @pytest.mark.asyncio
    async def test_cache_hit_kg_failure_via_http_health(self, tmp_path, monkeypatch):
        """HTTP /health reflects cache-hit KG failure with new error/timestamp, then recovery."""
        data_dir = tmp_path / "kg-cache-http"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
        host, port = httpd.server_address[:2]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            base_url = f"http://{host}:{port}"
            body = _wait_for_health(base_url)
            assert body["status"] == "degraded"
            first_err = body["stores"]["kg"]
            assert first_err.startswith("error:")

            _break_kg_store_differently(data_dir)
            base_time = time.time()
            with patch("taosmd.api.time.time", side_effect=lambda: base_time + 31):
                body = _wait_for_health(base_url)
            assert body["status"] == "degraded"
            second_err = body["stores"]["kg"]
            assert second_err.startswith("error:")
            assert second_err != first_err

            db_path = data_dir / "knowledge-graph.db"
            db_path.unlink()
            for suffix in ("-wal", "-shm"):
                p = Path(str(db_path) + suffix)
                if p.exists():
                    p.unlink()
            with patch("taosmd.api.time.time", side_effect=lambda: base_time + 62):
                body = _wait_for_health(base_url)
            assert body["status"] == "ok"
            assert body["stores"]["kg"] == "ok"
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=5)
            httpd.service_loop.close()

    @pytest.mark.asyncio
    async def test_kg_cooldown_blocks_and_allows_retry_by_attempt_count(self, tmp_path, monkeypatch):
        """KG broken+cached: call inside cooldown must not retry (no new attempt); after cooldown must."""
        data_dir = tmp_path / "kg-cooldown-count"
        data_dir.mkdir()
        _make_broken_kg_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["kg"] is None
        resolved = str(data_dir)
        ts_after_first = taosmd_api._last_attempt[resolved]["kg"]

        # Repair the store on disk so a retry would succeed if it happened.
        db_path = data_dir / "knowledge-graph.db"
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()

        base_time = time.time()
        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 10):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["kg"] is None
        assert taosmd_api._last_attempt[resolved]["kg"] == ts_after_first

        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 31):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["kg"] is not None
        assert taosmd_api._last_attempt.get(resolved, {}).get("kg", ts_after_first) != ts_after_first or "kg" not in taosmd_api._store_errors.get(resolved, {})

    @pytest.mark.asyncio
    async def test_cache_hit_vector_cooldown_blocks_retry_by_attempt_count(self, tmp_path, monkeypatch):
        """Vector broken+cached: inside cooldown must not retry; after cooldown must."""
        data_dir = tmp_path / "vec-cooldown-count"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is None
        resolved = str(data_dir)
        ts_after_first = taosmd_api._last_attempt[resolved]["vector"]

        # Repair on disk.
        db_path = data_dir / "vector-memory.db"
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM store_meta WHERE key='mode'")
        conn.commit()
        conn.close()

        base_time = time.time()
        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 10):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is None
        assert taosmd_api._last_attempt[resolved]["vector"] == ts_after_first

        with patch("taosmd.api.time.time", side_effect=lambda: base_time + 31):
            stores = await taosmd_api._ensure_stores(str(data_dir))
        assert stores["vector"] is not None

    @pytest.mark.asyncio
    async def test_reconcile_raises_store_unavailable_when_vector_down(self, tmp_path, monkeypatch):
        """reconcile raises StoreUnavailableError (not AttributeError) when vector is down."""
        data_dir = tmp_path / "reconcile-vec-down"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        with pytest.raises(taosmd_api.StoreUnavailableError) as excinfo:
            await taosmd_api.reconcile(agent="test-agent", data_dir=str(data_dir))
        assert "vector" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_reindex_raises_store_unavailable_when_vector_down(self, tmp_path, monkeypatch):
        """reindex raises StoreUnavailableError (not AttributeError) when vector is down."""
        data_dir = tmp_path / "reindex-vec-down"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        with pytest.raises(taosmd_api.StoreUnavailableError) as excinfo:
            await taosmd_api.reindex(agent="test-agent", data_dir=str(data_dir))
        assert "vector" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_supersede_vectors_raises_store_unavailable_when_vector_down(self, tmp_path, monkeypatch):
        """supersede_vectors raises StoreUnavailableError (not AttributeError) when vector is down."""
        data_dir = tmp_path / "supersede-vec-down"
        data_dir.mkdir()
        _make_broken_vector_store(data_dir)
        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})

        with pytest.raises(taosmd_api.StoreUnavailableError) as excinfo:
            await taosmd_api.supersede_vectors("match", data_dir=str(data_dir))
        assert "vector" in str(excinfo.value).lower()


class TestCollectionsHasChangesGuard:
    """Pin collections has_changes mutants: unchanged collection must not need vector store."""

    def test_unchanged_collection_noop_when_vector_down(self, tmp_path, monkeypatch):
        """Collection with no changes must not need vector store (has_changes=False)."""
        from taosmd import config
        from taosmd.collections import CollectionStore, ingest_folder

        data_dir = tmp_path / "coll-unchanged-vec-down"
        data_dir.mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Hello\n\nContent")

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})
        config.set_collections_allowed_roots([str(source_dir)], data_dir=str(data_dir))

        # Index with healthy stores
        store = CollectionStore(str(data_dir))
        col = store.create(name="docs", kind="docs", source_path=str(source_dir))
        stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
        vmem = stores["vector"]

        async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
            h = hash(text) & 0xFFFFFFFF
            return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

        vmem.embed = _fake_embed  # type: ignore[assignment]
        asyncio.run(ingest_folder(col["id"], data_dir=str(data_dir)))
        assert store.get(col["id"])["status"] == "ready"

        # Break vector and clear cache so it is down.
        _make_broken_vector_store(data_dir)
        taosmd_api._stores_cache.clear()
        taosmd_api._store_errors.clear()
        taosmd_api._last_attempt.clear()

        # Unchanged: must succeed as no-op, not require vector.
        stats = asyncio.run(ingest_folder(col["id"], data_dir=str(data_dir)))
        assert stats["files_indexed"] == 0
        assert stats["files_unchanged"] == 1
        assert stats["chunks_ingested"] == 0
        assert store.get(col["id"])["status"] == "ready"
        store.close()

    def test_changed_collection_fails_when_vector_down(self, tmp_path, monkeypatch):
        """Changed collection must fail when vector store is down (proves has_changes gates)."""
        from taosmd import config
        from taosmd.collections import CollectionStore, ingest_folder

        data_dir = tmp_path / "coll-changed-vec-down"
        data_dir.mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# Hello\n\nContent")

        monkeypatch.setattr(taosmd_api, "_stores_cache", {})
        monkeypatch.setattr(taosmd_api, "_store_errors", {})
        monkeypatch.setattr(taosmd_api, "_last_attempt", {})
        config.set_collections_allowed_roots([str(source_dir)], data_dir=str(data_dir))

        store = CollectionStore(str(data_dir))
        col = store.create(name="docs", kind="docs", source_path=str(source_dir))
        stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
        vmem = stores["vector"]

        async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
            h = hash(text) & 0xFFFFFFFF
            return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

        vmem.embed = _fake_embed  # type: ignore[assignment]
        asyncio.run(ingest_folder(col["id"], data_dir=str(data_dir)))
        assert store.get(col["id"])["status"] == "ready"

        # Change file, break vector, clear cache.
        (source_dir / "doc.md").write_text("# Hello\n\nChanged content")
        _make_broken_vector_store(data_dir)
        taosmd_api._stores_cache.clear()
        taosmd_api._store_errors.clear()
        taosmd_api._last_attempt.clear()

        with pytest.raises(taosmd_api.StoreUnavailableError, match="vector store unavailable"):
            asyncio.run(ingest_folder(col["id"], data_dir=str(data_dir)))
        assert store.get(col["id"])["status"] == "error"
        store.close()
