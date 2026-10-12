"""Storage-format guard: a vector store must not be reopened in a different
mode (dense vs late-interaction vs binary-quant) than it was built in.

The formats are mutually incompatible on disk, so a mismatch must fail loud
with a re-embed instruction rather than silently serve wrong-mode results.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock

from taosmd.vector_memory import StoreModeMismatch, VectorMemory


def _build(path, **kwargs):
    vm = VectorMemory(db_path=str(path), embed_mode="qmd", **kwargs)
    asyncio.run(vm.init())
    return vm


def test_fresh_store_records_mode(tmp_path):
    vm = _build(tmp_path / "vm.db")
    row = vm._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    assert row["value"] == "late_interaction=0;binary_quant=0;embedder=qmd:qmd"
    asyncio.run(vm.close())


def test_switching_embedder_on_existing_store_raises(tmp_path):
    # A store built with one embedder must refuse to open under another
    # (different vector space, stored vectors are meaningless to the new model).
    vm = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="onnx",
                      onnx_path="models/minilm-onnx")
    asyncio.run(vm.init())
    asyncio.run(vm.close())
    with pytest.raises(StoreModeMismatch, match="embedder|incompatible|embedding space"):
        vm2 = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="onnx",
                           onnx_path="models/arctic-embed-s")
        asyncio.run(vm2.init())


def test_legacy_marker_without_embedder_upgrades(tmp_path):
    # A marker written before the embedder component existed must not be
    # rejected; honoured for format and upgraded to the full signature.
    vm = _build(tmp_path / "vm.db")
    vm._conn.execute(
        "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', 'late_interaction=0;binary_quant=0')"
    )
    vm._conn.commit()
    asyncio.run(vm.close())
    vm2 = _build(tmp_path / "vm.db")  # same dense qmd mode, legacy marker present
    row = vm2._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    assert row["value"] == "late_interaction=0;binary_quant=0;embedder=qmd:qmd"
    asyncio.run(vm2.close())


def test_reopen_same_mode_ok(tmp_path):
    vm = _build(tmp_path / "vm.db")
    asyncio.run(vm.close())
    vm2 = _build(tmp_path / "vm.db")  # same dense mode
    assert vm2._conn is not None
    asyncio.run(vm2.close())


def test_reopen_late_interaction_on_dense_store_raises(tmp_path):
    vm = _build(tmp_path / "vm.db")  # dense
    asyncio.run(vm.close())
    with pytest.raises(StoreModeMismatch, match="incompatible|assumed dense"):
        _build(tmp_path / "vm.db", late_interaction=True,
               colbert_model="answerdotai/answerai-colbert-small-v1")


def test_reopen_binary_quant_on_dense_store_raises(tmp_path):
    vm = _build(tmp_path / "vm.db")
    asyncio.run(vm.close())
    with pytest.raises(StoreModeMismatch):
        _build(tmp_path / "vm.db", binary_quant=True)


def test_legacy_store_with_rows_no_marker_refuses_nondense(tmp_path):
    # Simulate a legacy store: dense rows, then drop the marker.
    vm = _build(tmp_path / "vm.db")
    vm._conn.execute(
        "INSERT INTO vector_memory (text, embedding, created_at) VALUES ('x', '[]', 1.0)"
    )
    vm._conn.execute("DELETE FROM store_meta WHERE key='mode'")
    vm._conn.commit()
    asyncio.run(vm.close())
    # Reopening in a non-dense mode is refused straight away (legacy assumed dense).
    with pytest.raises(StoreModeMismatch, match="assumed dense"):
        _build(tmp_path / "vm.db", binary_quant=True)
    # Reopening dense is fine (legacy assumed dense, matches).
    vm2 = _build(tmp_path / "vm.db")
    asyncio.run(vm2.close())


def test_qmd_fresh_store_with_health_records_model(tmp_path):
    # A: fresh qmd store with health model "m-a" records embedder=qmd:m-a in store_meta (read the row back).
    mock_client = AsyncMock()
    
    # Create an async function that returns a response
    async def mock_get(url, timeout=None):
        assert url == "http://localhost:7832/health"
        # Create a proper response mock that has status_code and json() as methods
        response_mock = AsyncMock()
        response_mock.status_code = 200
        
        async def mock_json():
            return {
                "ok": True,
                "backend": "local",
                "models": {"embed": "m-a"}
            }
        
        response_mock.json = mock_json
        return response_mock
    
    mock_client.get = mock_get
    
    vm = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm.init(http_client=mock_client))
    
    row = vm._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    assert row["value"] == "late_interaction=0;binary_quant=0;embedder=qmd:m-a"
    asyncio.run(vm.close())


def test_qmd_different_models_raise_mode_mismatch(tmp_path):
    # B: reopen that store with health model "m-b" raises StoreModeMismatch.
    # Build with model "m-a"
    mock_client_a = AsyncMock()
    
    async def mock_get_a(url, timeout=None):
        assert url == "http://localhost:7832/health"
        response_mock = AsyncMock()
        response_mock.status_code = 200
        
        async def mock_json_a():
            return {
                "ok": True,
                "backend": "local",
                "models": {"embed": "m-a"}
            }
        
        response_mock.json = mock_json_a
        return response_mock
    
    mock_client_a.get = mock_get_a
    
    vm = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm.init(http_client=mock_client_a))
    asyncio.run(vm.close())
    
    # Now try to open with a different model "m-b"
    mock_client_b = AsyncMock()
    
    async def mock_get_b(url, timeout=None):
        assert url == "http://localhost:7832/health"
        response_mock = AsyncMock()
        response_mock.status_code = 200
        
        async def mock_json_b():
            return {
                "ok": True,
                "backend": "local",
                "models": {"embed": "m-b"}
            }
        
        response_mock.json = mock_json_b
        return response_mock
    
    mock_client_b.get = mock_get_b
    
    with pytest.raises(StoreModeMismatch, match="different QMD models"):
        vm2 = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
        asyncio.run(vm2.init(http_client=mock_client_b))


def test_qmd_health_unavailable_uses_recorded(tmp_path):
    # C: reopen with health raising an exception opens fine and the recorded marker is still qmd:m-a afterwards (read it back).
    # Build with model "m-a"
    mock_client_a = AsyncMock()
    
    async def mock_get_a(url, timeout=None):
        assert url == "http://localhost:7832/health"
        response_mock = AsyncMock()
        response_mock.status_code = 200
        
        async def mock_json_a():
            return {
                "ok": True,
                "backend": "local",
                "models": {"embed": "m-a"}
            }
        
        response_mock.json = mock_json_a
        return response_mock
    
    mock_client_a.get = mock_get_a
    
    vm = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm.init(http_client=mock_client_a))
    asyncio.run(vm.close())
    
    # Now try to open with a health check that raises an exception
    mock_client_b = AsyncMock()
    
    async def mock_get_b(url, timeout=None):
        raise Exception("Health check failed")
    
    mock_client_b.get = mock_get_b
    
    # Should open fine (health unreachable) and keep the recorded marker
    vm2 = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm2.init(http_client=mock_client_b))
    
    # The recorded marker should still be qmd:m-a
    row = vm2._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    assert row["value"] == "late_interaction=0;binary_quant=0;embedder=qmd:m-a"
    asyncio.run(vm2.close())


def test_qmd_legacy_marker_upgrades(tmp_path):
    # D: a store whose marker was written as embedder=qmd:qmd opens with health model "m-a" and the marker becomes qmd:m-a.
    # Create a fresh store (it will have marker as qmd:qmd)
    vm = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm.init())
    
    # Manually overwrite the marker to simulate legacy qmd:qmd
    vm._conn.execute(
        "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', 'late_interaction=0;binary_quant=0;embedder=qmd:qmd')"
    )
    vm._conn.commit()
    asyncio.run(vm.close())
    
    # Now open with a specific model
    mock_client = AsyncMock()
    
    async def mock_get(url, timeout=None):
        assert url == "http://localhost:7832/health"
        response_mock = AsyncMock()
        response_mock.status_code = 200
        
        async def mock_json():
            return {
                "ok": True,
                "backend": "local",
                "models": {"embed": "m-a"}
            }
        
        response_mock.json = mock_json
        return response_mock
    
    mock_client.get = mock_get
    
    vm2 = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm2.init(http_client=mock_client))
    
    # The marker should have been upgraded to qmd:m-a
    row = vm2._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    assert row["value"] == "late_interaction=0;binary_quant=0;embedder=qmd:m-a"
    asyncio.run(vm2.close())


def test_qmd_health_errors_use_recorded(tmp_path):
    # E: health returning 500 and health returning JSON with no models key both behave like C (use recorded marker).
    # Build with model "m-a"
    mock_client_a = AsyncMock()
    
    async def mock_get_a(url, timeout=None):
        assert url == "http://localhost:7832/health"
        response_mock = AsyncMock()
        response_mock.status_code = 200
        
        async def mock_json_a():
            return {
                "ok": True,
                "backend": "local",
                "models": {"embed": "m-a"}
            }
        
        response_mock.json = mock_json_a
        return response_mock
    
    mock_client_a.get = mock_get_a
    
    vm = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm.init(http_client=mock_client_a))
    asyncio.run(vm.close())
    
    # Test case 1: health returning 500
    mock_client_b = AsyncMock()
    
    async def mock_get_b(url, timeout=None):
        assert url == "http://localhost:7832/health"
        response_mock = AsyncMock()
        response_mock.status_code = 500
        return response_mock
    
    mock_client_b.get = mock_get_b
    
    vm2 = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm2.init(http_client=mock_client_b))
    
    row = vm2._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    assert row["value"] == "late_interaction=0;binary_quant=0;embedder=qmd:m-a"
    asyncio.run(vm2.close())
    
    # Test case 2: health returning JSON with no models key
    mock_client_c = AsyncMock()
    
    async def mock_get_c(url, timeout=None):
        assert url == "http://localhost:7832/health"
        response_mock = AsyncMock()
        response_mock.status_code = 200
        
        async def mock_json_c():
            return {"ok": True, "backend": "local"}
        
        response_mock.json = mock_json_c
        return response_mock
    
    mock_client_c.get = mock_get_c
    
    vm3 = VectorMemory(db_path=str(tmp_path / "vm.db"), embed_mode="qmd")
    asyncio.run(vm3.init(http_client=mock_client_c))
    
    row = vm3._conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    assert row["value"] == "late_interaction=0;binary_quant=0;embedder=qmd:m-a"
    asyncio.run(vm3.close())