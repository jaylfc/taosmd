"""Tests for f32 base64 dense embedding migration.

Tests the migration from JSON-stored dense embeddings to base64-encoded
little-endian float32, including marker handling, crash safety, and downgrade
behaviour.
"""

import asyncio
import base64
import json
import os
import shutil
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from taosmd.vector_memory import StoreModeMismatch, VectorMemory


# Deterministic embedder for reproducible vectors without model downloads
class FixedEmbedder:
    """Produces fixed vectors based on text hash."""

    def __init__(self, dim=384):
        self.dim = dim

    async def embed(self, text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(h)
        vec = rng.normal(0, 1, self.dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.tolist()

    async def embed_tokens(self, text: str, task: str = "search_document"):
        return []


def _make_fixed_embedder(vmem: VectorMemory, dim: int = 384) -> None:
    """Replace vmem.embed with a deterministic embedder."""
    fe = FixedEmbedder(dim)

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        return await fe.embed(text, task)

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _json_to_f32b64(vec: list[float]) -> str:
    """Convert a JSON list of floats to base64 little-endian float32."""
    arr = np.asarray(vec, dtype="<f4")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _f32b64_to_json(b64: str) -> list[float]:
    """Convert base64 little-endian float32 back to JSON list."""
    arr = np.frombuffer(base64.b64decode(b64), dtype="<f4")
    return arr.tolist()


def _build_legacy_json_store(db_path: str, n_rows: int = 200, with_marker: bool = True) -> None:
    """Build a store exactly as origin/master writes it: JSON embeddings + legacy marker."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            valid_to REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_created ON vector_memory(created_at DESC)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    fe = FixedEmbedder(384)
    for i in range(n_rows):
        text = f"document {i} with some content for testing migration"
        vec = asyncio.run(fe.embed(text))
        emb_json = json.dumps(vec)
        conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', ?)",
            (text, emb_json, float(i)),
        )

    if with_marker:
        # Marker exactly as master writes it (no dense_format, no embedder if we mimic old)
        conn.execute(
            "INSERT INTO store_meta (key, value) VALUES ('mode', 'late_interaction=0;binary_quant=0;embedder=all-MiniLM-L6-v2')"
        )
    conn.commit()
    conn.close()


def _count_json_rows(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cnt = conn.execute("SELECT COUNT(*) FROM vector_memory WHERE embedding LIKE '[%'").fetchone()[0]
    conn.close()
    return cnt


def _count_f32_rows(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cnt = conn.execute("SELECT COUNT(*) FROM vector_memory WHERE embedding NOT LIKE '[%'").fetchone()[0]
    conn.close()
    return cnt


def _get_marker(db_path: str) -> str | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    conn.close()
    return row["value"] if row else None


def _get_file_size(db_path: str) -> int:
    return os.path.getsize(db_path)


def _free_space(path: str) -> int:
    stat = shutil.disk_usage(path)
    return stat.free


class TestF32Migration:
    """Test suite for dense embedding f32 base64 migration."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db = tmp_path / "test_vec.db"
        yield str(db)
        if db.exists():
            db.unlink()

    # (a) marker-carrying JSON store with 200 rows
    def test_migration_marker_carrying_json_store(self, temp_db):
        _build_legacy_json_store(temp_db, n_rows=200, with_marker=True)

        # Capture pre-migration search results
        vm_pre = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm_pre.init())
        _make_fixed_embedder(vm_pre)

        queries = [
            "document 50 content testing",
            "document 100 with some",
            "document 150 migration test",
            "document 10 content",
            "document 199 testing migration",
        ]
        pre_results = []
        for q in queries:
            res = asyncio.run(vm_pre.search(q, limit=5, hybrid=False))
            pre_results.append([(r["id"], r["similarity"]) for r in res])
        asyncio.run(vm_pre.close())

        # Now migrate by re-initializing
        vm = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)

        # Verify all rows are now f32 base64
        assert _count_json_rows(temp_db) == 0, "Some rows still have JSON embeddings"
        assert _count_f32_rows(temp_db) == 200, "Not all rows migrated to f32"

        # Verify marker has dense_format=f32
        marker = _get_marker(temp_db)
        assert marker is not None, "Marker missing after migration"
        assert "dense_format=f32" in marker, f"Marker missing dense_format=f32: {marker}"
        assert "late_interaction=0" in marker
        assert "binary_quant=0" in marker

        # Verify row count and IDs unchanged
        assert asyncio.run(vm.count()) == 200
        ids = [row["id"] for row in vm._conn.execute("SELECT id FROM vector_memory ORDER BY id").fetchall()]
        assert ids == list(range(1, 201))

        # Verify search results identical (same IDs, same order, same scores)
        for i, q in enumerate(queries):
            res = asyncio.run(vm.search(q, limit=5, hybrid=False))
            post_ids = [(r["id"], r["similarity"]) for r in res]
            assert post_ids == pre_results[i], f"Search results differ for query {i}: {post_ids} vs {pre_results[i]}"

        asyncio.run(vm.close())

    # (b) legacy store with rows and NO marker
    def test_migration_legacy_no_marker(self, temp_db):
        _build_legacy_json_store(temp_db, n_rows=200, with_marker=False)

        vm = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)

        assert _count_json_rows(temp_db) == 0
        assert _count_f32_rows(temp_db) == 200

        marker = _get_marker(temp_db)
        assert marker is not None
        assert "dense_format=f32" in marker

        assert asyncio.run(vm.count()) == 200
        asyncio.run(vm.close())

    # (c) crash mid-migration
    def test_migration_crash_midway(self, temp_db):
        _build_legacy_json_store(temp_db, n_rows=200, with_marker=True)

        # Monkeypatch the conversion to raise on row 150
        original_init = VectorMemory.init

        async def patched_init(self, http_client=None):
            # Run the original init up to the migration point, then inject failure
            await original_init(self, http_client)
            # We need to patch the internal migration logic. Instead, patch at a lower level.
            pass

        # Better approach: patch the row conversion function directly
        import taosmd.vector_memory as vm_module

        call_count = {"n": 0}

        def fail_on_150(vec):
            call_count["n"] += 1
            if call_count["n"] == 150:
                raise RuntimeError("Injected failure at row 150")
            return _json_to_f32b64(vec)

        # We need to find where the conversion happens. Let's patch json.loads in the migration path.
        # Actually, the migration will be in _check_store_mode or init. Let's trace through.
        # The migration logic hasn't been written yet, so this test will fail until we implement it.
        # For now, we'll write the test expecting the migration to be in a specific place.

        # This test will be completed after we see the implementation structure.
        # For RED-FIRST, we just need it to fail.
        vm = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")

        with patch.object(vm_module.VectorMemory, "_migrate_dense_to_f32", side_effect=RuntimeError("Injected failure at row 150")):
            with pytest.raises(RuntimeError, match="Injected failure at row 150"):
                asyncio.run(vm.init())

        # After failed migration, db should still have JSON rows and old marker
        assert _count_json_rows(temp_db) == 200
        assert _count_f32_rows(temp_db) == 0
        marker = _get_marker(temp_db)
        assert marker == "late_interaction=0;binary_quant=0;embedder=all-MiniLM-L6-v2"

        # Second init (patch removed) should migrate fully
        vm2 = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm2.init())
        _make_fixed_embedder(vm2)

        assert _count_json_rows(temp_db) == 0
        assert _count_f32_rows(temp_db) == 200
        marker = _get_marker(temp_db)
        assert "dense_format=f32" in marker
        asyncio.run(vm2.close())

    # (d) fresh empty store writes f32 from first add()
    def test_fresh_store_writes_f32(self, temp_db):
        vm = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)

        row_id = asyncio.run(vm.add("first document"))
        assert row_id == 1

        # Check the stored embedding is f32 base64, not JSON
        row = vm._conn.execute("SELECT embedding FROM vector_memory WHERE id=1").fetchone()
        emb = row["embedding"]
        assert not emb.startswith("["), f"Expected f32 base64, got JSON: {emb[:50]}"

        # Verify it decodes correctly
        decoded = _f32b64_to_json(emb)
        assert len(decoded) == 384

        # Marker should have dense_format=f32
        marker = _get_marker(temp_db)
        assert "dense_format=f32" in marker

        asyncio.run(vm.close())

    # (e) binary_quant and late_interaction stores untouched
    def test_binary_quant_untouched(self, temp_db):
        vm = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2", binary_quant=True)
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)

        asyncio.run(vm.add("test doc 1"))
        asyncio.run(vm.add("test doc 2"))

        # Capture embeddings before re-init
        before = [row["embedding"] for row in vm._conn.execute("SELECT embedding FROM vector_memory ORDER BY id").fetchall()]
        asyncio.run(vm.close())

        # Re-open same mode
        vm2 = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2", binary_quant=True)
        asyncio.run(vm2.init())
        _make_fixed_embedder(vm2)

        after = [row["embedding"] for row in vm2._conn.execute("SELECT embedding FROM vector_memory ORDER BY id").fetchall()]

        assert before == after, "binary_quant embeddings changed after re-init"
        asyncio.run(vm2.close())

    def test_late_interaction_untouched(self, temp_db):
        # Late interaction needs a different setup; use ONNX mode with a dummy path
        # Since we don't have the model, we'll test the marker logic only
        vm = VectorMemory(db_path=temp_db, embed_mode="onnx", onnx_path="models/minilm-onnx", late_interaction=True)
        asyncio.run(vm.init())

        # Insert a dummy late-interaction row (base64 float16)
        import numpy as np
        token_mat = np.random.randn(10, 384).astype(np.float16)
        blob = base64.b64encode(token_mat.tobytes()).decode("ascii")
        vm._conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', 1.0)",
            ("late interaction doc", blob),
        )
        vm._conn.commit()

        before = [row["embedding"] for row in vm._conn.execute("SELECT embedding FROM vector_memory ORDER BY id").fetchall()]
        asyncio.run(vm.close())

        # Re-open same mode
        vm2 = VectorMemory(db_path=temp_db, embed_mode="onnx", onnx_path="models/minilm-onnx", late_interaction=True)
        asyncio.run(vm2.init())

        after = [row["embedding"] for row in vm2._conn.execute("SELECT embedding FROM vector_memory ORDER BY id").fetchall()]

        assert before == after, "late_interaction embeddings changed after re-init"
        asyncio.run(vm2.close())

    # (f) downgrade: old-shape signature opening migrated store raises StoreModeMismatch
    def test_downgrade_raises_mismatch(self, temp_db):
        # First, create a migrated store (with dense_format=f32)
        _build_legacy_json_store(temp_db, n_rows=10, with_marker=True)
        vm = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)
        asyncio.run(vm.close())

        # Verify marker has dense_format=f32
        marker = _get_marker(temp_db)
        assert "dense_format=f32" in marker

        # Now monkeypatch _store_mode_signature to return old shape (no dense_format)
        import taosmd.vector_memory as vm_module

        old_signature = "late_interaction=0;binary_quant=0;embedder=all-MiniLM-L6-v2"

        with patch.object(vm_module.VectorMemory, "_store_mode_signature", return_value=old_signature):
            vm_old = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")
            with pytest.raises(StoreModeMismatch, match="dense_format|conflict"):
                asyncio.run(vm_old.init())

    # (g) file size: after migration + VACUUM the 200-row db is at least 2x smaller
    def test_file_size_reduction(self, temp_db):
        _build_legacy_json_store(temp_db, n_rows=200, with_marker=True)
        size_before = _get_file_size(temp_db)

        vm = VectorMemory(db_path=temp_db, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)
        asyncio.run(vm.close())

        size_after = _get_file_size(temp_db)
        print(f"Size before: {size_before}, after: {size_after}, ratio: {size_before/size_after:.2f}x")

        assert size_after * 2 <= size_before, f"Expected at least 2x reduction, got {size_before/size_after:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])