"""Legacy JSON vector stores are served as-is on open (Card A).

A legacy store (marker without ``dense_format``) must be opened read/write
without converting its rows: ``init`` performs zero conversions, ``search``
reads the JSON embeddings, and ``add()`` keeps writing JSON. The store remains
readable by 0.4.x. A fresh store still writes float32.
"""

import asyncio
import json
import sqlite3


from taosmd.vector_memory import StoreModeMismatch, VectorMemory

_LEGACY_MARKER = "late_interaction=0;binary_quant=0;embedder=qmd:qmd"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vector_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    valid_to REAL
);
CREATE INDEX IF NOT EXISTS idx_vm_created ON vector_memory(created_at DESC);
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class FixedEmbedder:
    """Deterministic embedder for reproducible vectors without model downloads."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed(self, text: str, task: str = "search_document") -> list[float]:
        import numpy as np

        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(h)
        vec = rng.normal(0, 1, self.dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.tolist()

    async def embed_tokens(self, text: str, task: str = "search_document"):
        return []


def _build_legacy_json_store(
    db_path: str, n_rows: int = 20, with_marker: bool = True
) -> None:
    """Build a legacy 0.4.0-shaped store: JSON embeddings + legacy marker."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    fe = FixedEmbedder(384)
    for i in range(n_rows):
        text = f"legacy document {i} with some content"
        vec = asyncio.run(fe.embed(text))
        conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', ?)",
            (text, json.dumps(vec), float(i)),
        )
    if with_marker:
        conn.execute(
            "INSERT INTO store_meta (key, value) VALUES ('mode', ?)", (_LEGACY_MARKER,)
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


def _get_marker(db_path: str) -> str | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    conn.close()
    return row["value"] if row else None


def _install_fixed_embedder(vmem: VectorMemory) -> None:
    fe = FixedEmbedder()

    async def _fake(text: str, task: str = "search_document") -> list[float]:
        return await fe.embed(text, task)

    vmem.embed = _fake  # type: ignore[assignment]


def test_legacy_json_store_not_converted_on_init(tmp_path):
    """RED-FIRST: on origin/master init() converts every row; after the fix none are."""
    db = str(tmp_path / "vm.db")
    _build_legacy_json_store(db, n_rows=20)
    before = _count_json_rows(db)
    assert before == 20

    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())

    assert _count_json_rows(db) == before, "legacy rows were converted on open"
    marker = _get_marker(db)
    assert marker is not None
    assert "dense_format" not in marker
    asyncio.run(vm.close())


def test_legacy_store_search_and_add_stay_json(tmp_path):
    db = str(tmp_path / "vm.db")
    _build_legacy_json_store(db, n_rows=20)

    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())
    _install_fixed_embedder(vm)

    results = asyncio.run(vm.search("legacy document 5 content", limit=5, hybrid=False))
    assert len(results) > 0

    before = _count_json_rows(db)
    row_id = asyncio.run(vm.add("new legacy document"))
    assert row_id > 0
    # The new row is written as JSON (store is legacy); no row is converted.
    assert _count_json_rows(db) == before + 1
    marker = _get_marker(db)
    assert "dense_format" not in marker
    asyncio.run(vm.close())


def _check_store_mode_v040(
    conn, late_interaction: bool = False, binary_quant: bool = False
) -> None:
    """Faithful copy of v0.4.0 _check_store_mode.

    v0.4.0 only knew the format components (late_interaction, binary_quant);
    the embedder of a legacy store is unknowable and not enforced. A legacy
    marker lacking dense_format must be accepted.
    """
    row = conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
    if row is None:
        return
    rec = {}
    for part in row["value"].split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            rec[k] = v
    if rec.get("late_interaction") != str(int(late_interaction)):
        raise StoreModeMismatch("v0.4.0: late_interaction mismatch")
    if rec.get("binary_quant") != str(int(binary_quant)):
        raise StoreModeMismatch("v0.4.0: binary_quant mismatch")


def test_v040_check_store_mode_accepts_legacy_store(tmp_path):
    """The v0.4.0 _check_store_mode logic must accept the legacy store."""
    db = str(tmp_path / "vm.db")
    _build_legacy_json_store(db, n_rows=20)

    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())
    marker = _get_marker(db)
    assert marker is not None
    assert "dense_format" not in marker

    # A copy of the v0.4.0 guard accepts the store (format components match).
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        _check_store_mode_v040(conn, late_interaction=False, binary_quant=False)
    finally:
        conn.close()

    # Reopening in the same dense mode is also accepted.
    vm2 = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm2.init())
    asyncio.run(vm2.close())


def test_fresh_store_writes_f32(tmp_path):
    db = str(tmp_path / "vm.db")
    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())
    _install_fixed_embedder(vm)
    row_id = asyncio.run(vm.add("first document"))
    assert row_id == 1
    emb = vm._conn.execute("SELECT embedding FROM vector_memory WHERE id=1").fetchone()[
        "embedding"
    ]
    assert not emb.startswith("["), f"Expected f32 base64, got JSON: {emb[:50]}"
    marker = _get_marker(db)
    assert "dense_format=f32" in marker
    asyncio.run(vm.close())
