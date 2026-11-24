"""Tests for the f32 base64 dense embedding format and store-mode guard.

Conversion is no longer implicit on open (see test_legacy_store_served_as_is.py);
this file covers the format written by fresh stores, the untouched binary-quant
and late-interaction paths, and store-mode mismatch protection.
"""

import asyncio
import base64
import json
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


def _build_legacy_json_store(
    db_path: str, n_rows: int = 200, with_marker: bool = True
) -> None:
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_vm_created ON vector_memory(created_at DESC)"
    )
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
    cnt = conn.execute(
        "SELECT COUNT(*) FROM vector_memory WHERE embedding LIKE '[%'"
    ).fetchone()[0]
    conn.close()
    return cnt


def _count_f32_rows(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cnt = conn.execute(
        "SELECT COUNT(*) FROM vector_memory WHERE embedding NOT LIKE '[%'"
    ).fetchone()[0]
    conn.close()
    return cnt


def _get_marker(db_path: str) -> str | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    conn.close()
    return row["value"] if row else None


def _get_rows_and_marker(db_path: str):
    """Get all embeddings and marker from store for byte-identical checks."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, embedding FROM vector_memory ORDER BY id"
    ).fetchall()
    marker_row = conn.execute(
        "SELECT value FROM store_meta WHERE key='mode'"
    ).fetchone()
    conn.close()
    return [(r["id"], r["embedding"]) for r in rows], marker_row[
        "value"
    ] if marker_row else None


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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_vm_created ON vector_memory(created_at DESC)"
    )
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
    """Test suite for the dense f32 base64 format and mode guard."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db = tmp_path / "test_vec.db"
        yield str(db)
        if db.exists():
            db.unlink()

    # (d) fresh empty store writes f32 from first add()
    def test_fresh_store_writes_f32(self, temp_db):
        vm = VectorMemory(db_path=temp_db, embed_mode="qmd")
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)

        row_id = asyncio.run(vm.add("first document"))
        assert row_id == 1

        # Check the stored embedding is f32 base64, not JSON
        row = vm._conn.execute(
            "SELECT embedding FROM vector_memory WHERE id=1"
        ).fetchone()
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
        vm = VectorMemory(db_path=temp_db, embed_mode="qmd", binary_quant=True)
        asyncio.run(vm.init())
        _make_fixed_embedder(vm)

        asyncio.run(vm.add("test doc 1"))
        asyncio.run(vm.add("test doc 2"))

        # Capture embeddings before re-init
        before = [
            row["embedding"]
            for row in vm._conn.execute(
                "SELECT embedding FROM vector_memory ORDER BY id"
            ).fetchall()
        ]
        asyncio.run(vm.close())

        # Re-open same mode
        vm2 = VectorMemory(db_path=temp_db, embed_mode="qmd", binary_quant=True)
        asyncio.run(vm2.init())
        _make_fixed_embedder(vm2)

        after = [
            row["embedding"]
            for row in vm2._conn.execute(
                "SELECT embedding FROM vector_memory ORDER BY id"
            ).fetchall()
        ]

        assert before == after, "binary_quant embeddings changed after re-init"
        asyncio.run(vm2.close())

    def test_late_interaction_untouched(self, temp_db):
        # Late interaction needs a different setup; use ONNX mode with a dummy path
        # Since we don't have the model, we'll test the marker logic only
        vm = VectorMemory(
            db_path=temp_db,
            embed_mode="onnx",
            onnx_path="models/minilm-onnx",
            late_interaction=True,
        )
        asyncio.run(vm.init())

        # Insert a dummy late-interaction row (base64 float16)
        token_mat = np.random.randn(10, 384).astype(np.float16)
        blob = base64.b64encode(token_mat.tobytes()).decode("ascii")
        vm._conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', 1.0)",
            ("late interaction doc", blob),
        )
        vm._conn.commit()

        before = [
            row["embedding"]
            for row in vm._conn.execute(
                "SELECT embedding FROM vector_memory ORDER BY id"
            ).fetchall()
        ]
        asyncio.run(vm.close())

        # Re-open same mode
        vm2 = VectorMemory(
            db_path=temp_db,
            embed_mode="onnx",
            onnx_path="models/minilm-onnx",
            late_interaction=True,
        )
        asyncio.run(vm2.init())

        after = [
            row["embedding"]
            for row in vm2._conn.execute(
                "SELECT embedding FROM vector_memory ORDER BY id"
            ).fetchall()
        ]

        assert before == after, "late_interaction embeddings changed after re-init"
        asyncio.run(vm2.close())

    # (f) downgrade: old-shape signature opening a converted store raises StoreModeMismatch
    def test_downgrade_raises_mismatch(self, temp_db):
        # Build a converted store manually: f32 rows + marker carrying dense_format=f32.
        conn = sqlite3.connect(temp_db)
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        fe = FixedEmbedder(384)
        for i in range(5):
            vec = asyncio.run(fe.embed(f"doc {i}"))
            emb = _json_to_f32b64(vec)
            conn.execute(
                "INSERT INTO vector_memory (text, embedding, created_at) VALUES (?, ?, ?)",
                (f"doc {i}", emb, float(i)),
            )
        conn.execute(
            "INSERT INTO store_meta (key, value) VALUES ('mode', 'late_interaction=0;binary_quant=0;dense_format=f32;embedder=qmd:qmd')"
        )
        conn.commit()
        conn.close()

        # Opening normally is accepted (dense f32 store, qmd instance).
        vm = VectorMemory(db_path=temp_db, embed_mode="qmd")
        asyncio.run(vm.init())
        marker = _get_marker(temp_db)
        assert "dense_format=f32" in marker
        asyncio.run(vm.close())

        # Downgrade: an instance whose signature reports the old shape (no
        # dense_format) opening a store whose marker carries dense_format=f32
        # must raise StoreModeMismatch.
        import taosmd.vector_memory as vm_module

        old_signature = "late_interaction=0;binary_quant=0;embedder=qmd:qmd"

        with patch.object(
            vm_module.VectorMemory, "_store_mode_signature", return_value=old_signature
        ):
            vm_old = VectorMemory(db_path=temp_db, embed_mode="qmd")
            with pytest.raises(StoreModeMismatch, match="dense_format|conflict"):
                asyncio.run(vm_old.init())

    # (h) guard-bypass fix: conflicting recorded markers must raise StoreModeMismatch
    # and leave the database byte-identical
    @pytest.mark.parametrize(
        "marker,embeddings_json,test_id,vm_kwargs",
        CONFLICTING_STORES,
        ids=[c[2] for c in CONFLICTING_STORES],
    )
    def test_conflicting_store_raises_mismatch_and_unchanged(
        self, tmp_path, marker, embeddings_json, test_id, vm_kwargs
    ):
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
        assert marker_after == marker_before, (
            f"Marker changed: {marker_before} -> {marker_after}"
        )
        assert rows_after == rows_before, (
            f"Embeddings changed: {rows_before} -> {rows_after}"
        )
