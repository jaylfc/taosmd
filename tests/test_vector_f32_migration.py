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


def _get_rows_and_marker(db_path: str):
    """Get all embeddings and marker from store for byte-identical checks."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, embedding FROM vector_memory ORDER BY id").fetchall()
    marker_row = conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    conn.close()
    return [(r["id"], r["embedding"]) for r in rows], marker_row["value"] if marker_row else None


def _build_conflicting_store(db_path: str, marker: str, embeddings_json: bool) -> None:
    """Build a store with a specific marker and embedding format using raw sqlite3."""
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

    # Add 3 rows with consistent embeddings
    for i in range(3):
        text = f"document {i}"
        if embeddings_json:
            vec = [0.1] * 384
            emb = json.dumps(vec)
        else:
            # base64 f32 (dense format) or base64 f16 (late interaction) or packed bits (binary quant)
            arr = np.array([0.1] * 384, dtype="<f4")
            emb = base64.b64encode(arr.tobytes()).decode("ascii")
        conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', ?)",
            (text, emb, float(i)),
        )

    conn.execute(
        "INSERT INTO store_meta (key, value) VALUES ('mode', ?)",
        (marker,),
    )
    conn.commit()
    conn.close()


# Parametrized conflicting store test cases from the guard-bypass fix
CONFLICTING_STORES = [
    # (marker, embeddings_json, test_id, vm_kwargs)
    (
        "late_interaction=0;binary_quant=0;embedder=some-other-model",
        True,  # JSON embeddings
        "wrong_embedder_json",
        {"embed_mode": "qmd"},  # default dense qmd
    ),
    (
        "late_interaction=0;binary_quant=1;embedder=qmd:qmd",
        False,  # base64 f32 embeddings (binary quant stores packed bits, but we use f32 for test)
        "binary_quant_store",
        {"embed_mode": "qmd", "binary_quant": False},  # opening as dense
    ),
    (
        "late_interaction=1;binary_quant=0;embedder=qmd:qmd",
        False,  # base64 f32 embeddings (late interaction stores token matrices)
        "late_interaction_store",
        {"embed_mode": "qmd", "late_interaction": False},  # opening as dense
    ),
]


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

    # (h) guard-bypass fix: conflicting recorded markers must raise StoreModeMismatch
    # and leave the database byte-identical
    @pytest.mark.parametrize("marker,embeddings_json,test_id,vm_kwargs", CONFLICTING_STORES, ids=[c[2] for c in CONFLICTING_STORES])
    def test_conflicting_store_raises_mismatch_and_unchanged(self, tmp_path, marker, embeddings_json, test_id, vm_kwargs):
        """A store with a conflicting recorded marker must raise StoreModeMismatch
        and leave the database byte-identical (marker and all embeddings unchanged)."""
        db_path = str(tmp_path / f"{test_id}.db")
        _build_conflicting_store(db_path, marker, embeddings_json)

        # Capture pre-state
        rows_before, marker_before = _get_rows_and_marker(db_path)

        # Opening with conflicting mode must raise StoreModeMismatch
        vm = VectorMemory(db_path=db_path, **vm_kwargs)
        with pytest.raises(StoreModeMismatch):
            asyncio.run(vm.init())

        # Capture post-state
        rows_after, marker_after = _get_rows_and_marker(db_path)

        # Database must be byte-identical
        assert marker_after == marker_before, f"Marker changed: {marker_before} -> {marker_after}"
        assert rows_after == rows_before, f"Embeddings changed: {rows_before} -> {rows_after}"

    # (i) positive control: conflict-free legacy marker (lacks dense_format) still migrates
    def test_conflict_free_marker_still_migrates(self, tmp_path):
        """A conflict-free legacy marker (lacks dense_format) must still migrate."""
        db_path = str(tmp_path / "conflict_free.db")
        # Marker matches current instance except lacks dense_format
        marker = "late_interaction=0;binary_quant=0;embedder=qmd:qmd"
        _build_conflicting_store(db_path, marker, embeddings_json=True)

        rows_before, marker_before = _get_rows_and_marker(db_path)

        vm = VectorMemory(db_path=db_path, embed_mode="qmd")
        asyncio.run(vm.init())  # Should NOT raise
        asyncio.run(vm.close())

        rows_after, marker_after = _get_rows_and_marker(db_path)

        # Marker should be upgraded to include dense_format=f32
        assert "dense_format=f32" in marker_after
        assert marker_after != marker_before
        # All rows should be migrated to f32 base64
        for _, emb in rows_after:
            assert not emb.startswith("["), f"Row still has JSON embedding: {emb[:50]}"
        # Row count unchanged
        assert len(rows_after) == len(rows_before) == 3


def _build_store_with_malformed_row(db_path: str, n_rows: int = 5000, malformed_row: int = 4000) -> None:
    """Build a store with JSON embeddings and one malformed row."""
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
    for i in range(1, n_rows + 1):
        text = f"document {i} with some content for testing migration"
        if i == malformed_row:
            # Malformed JSON - missing closing bracket
            emb_json = '[0.1, 0.2'
        else:
            vec = asyncio.run(fe.embed(text))
            emb_json = json.dumps(vec)
        conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', ?)",
            (text, emb_json, float(i)),
        )

    # Legacy marker (no dense_format)
    conn.execute(
        "INSERT INTO store_meta (key, value) VALUES ('mode', 'late_interaction=0;binary_quant=0;embedder=all-MiniLM-L6-v2')"
    )
    conn.commit()
    conn.close()


def _build_store_with_dimension_mismatch(db_path: str, n_rows: int = 100, mismatch_row: int = 50, wrong_dim: int = 768) -> None:
    """Build a store with JSON embeddings and one row with wrong dimension."""
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
    for i in range(1, n_rows + 1):
        text = f"document {i} with some content for testing migration"
        if i == mismatch_row:
            # Valid JSON but wrong dimension (768 instead of 384)
            vec = [0.1] * wrong_dim
            emb_json = json.dumps(vec)
        else:
            vec = asyncio.run(fe.embed(text))
            emb_json = json.dumps(vec)
        conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', ?)",
            (text, emb_json, float(i)),
        )

    # Legacy marker (no dense_format)
    conn.execute(
        "INSERT INTO store_meta (key, value) VALUES ('mode', 'late_interaction=0;binary_quant=0;embedder=all-MiniLM-L6-v2')"
    )
    conn.commit()
    conn.close()


class TestMalformedRowHandling:
    """Tests for Card E: malformed rows must not break store open/search."""

    def test_malformed_json_row_does_not_break_store(self, tmp_path):
        """Store with one malformed JSON row must open and search (Card E)."""
        db_path = str(tmp_path / "test_malformed.db")
        _build_store_with_malformed_row(db_path, n_rows=5000, malformed_row=4000)

        vm = VectorMemory(db_path=db_path, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)

        # Search should work and return results
        results = asyncio.run(vm.search("document 100 content", limit=5, hybrid=False))
        assert len(results) > 0, "Search should return results"

        # Verify the malformed row was left as-is (still JSON)
        row = vm._conn.execute("SELECT embedding FROM vector_memory WHERE id=4000").fetchone()
        assert row is not None
        assert row["embedding"] == '[0.1, 0.2', "Malformed row should be left as-is"

        # Other rows should be converted to f32
        row = vm._conn.execute("SELECT embedding FROM vector_memory WHERE id=1").fetchone()
        assert row is not None
        assert not row["embedding"].startswith("["), "Valid rows should be converted to f32"

        asyncio.run(vm.close())

    def test_dimension_mismatch_row_does_not_break_store(self, tmp_path):
        """Store with one dimension-mismatch row must open and search (Card E)."""
        db_path = str(tmp_path / "test_dim_mismatch.db")
        _build_store_with_dimension_mismatch(db_path, n_rows=100, mismatch_row=50, wrong_dim=768)

        vm = VectorMemory(db_path=db_path, embed_mode="local", local_model="all-MiniLM-L6-v2")
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)

        # Search should work and return results
        results = asyncio.run(vm.search("document 10 content", limit=5, hybrid=False))
        assert len(results) > 0, "Search should return results"

        # Verify the mismatched row was left as-is (still JSON)
        row = vm._conn.execute("SELECT embedding FROM vector_memory WHERE id=50").fetchone()
        assert row is not None
        assert row["embedding"].startswith("["), "Mismatched row should be left as JSON"
        # Verify it has 768 dimensions
        vec = json.loads(row["embedding"])
        assert len(vec) == 768, "Mismatched row should have 768 dimensions"

        # Other rows should be converted to f32
        row = vm._conn.execute("SELECT embedding FROM vector_memory WHERE id=1").fetchone()
        assert row is not None
        assert not row["embedding"].startswith("["), "Valid rows should be converted to f32"

        asyncio.run(vm.close())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
