"""Legacy JSON vector stores are served as-is on open (Card A).

A legacy store (marker without ``dense_format``) must be opened read/write
without converting its rows: ``init`` performs zero conversions, ``search``
reads the JSON embeddings, and ``add()`` keeps writing JSON. The store remains
readable by 0.4.x. A fresh store still writes float32.
"""

import asyncio
import json
import sqlite3

import pytest

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


def _v040_signature(late_interaction: bool, binary_quant: bool, embedder: str) -> str:
    """v0.4.0's _store_mode_signature (git show v0.4.0:taosmd/vector_memory.py)."""
    return (
        f"late_interaction={int(late_interaction)};"
        f"binary_quant={int(binary_quant)};"
        f"embedder={embedder}"
    )


def _parse_mode_v040(sig: str) -> dict:
    out = {}
    for part in sig.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def _check_store_mode_v040(
    db_path: str,
    late_interaction: bool = False,
    binary_quant: bool = False,
    embedder: str = "qmd:qmd",
) -> str:
    """Faithful copy of v0.4.0 VectorMemory._check_store_mode, run on a file.

    Same control flow as the original: component-wise comparison over the keys
    the recorded marker carries (so ANY key v0.4.0 does not know, such as
    dense_format, is a conflict), and the no-marker branch. Raises
    StoreModeMismatch where v0.4.0 would. Returns the marker v0.4.0 would leave
    behind.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        current = _v040_signature(late_interaction, binary_quant, embedder)
        cur = _parse_mode_v040(current)
        row = conn.execute("SELECT value FROM store_meta WHERE key = 'mode'").fetchone()
        recorded = row["value"] if row else None

        if recorded is not None:
            rec = _parse_mode_v040(recorded)
            conflict = {k: v for k, v in rec.items() if cur.get(k) != v}
            if conflict:
                raise StoreModeMismatch(
                    f"vector store at {db_path} was built with {recorded} "
                    f"but is being opened with {current} (conflict on {conflict})."
                )
            return current if recorded != current else recorded

        has_rows = conn.execute("SELECT 1 FROM vector_memory LIMIT 1").fetchone() is not None
        legacy_format = {"late_interaction": "0", "binary_quant": "0"}
        if has_rows and any(cur.get(k) != v for k, v in legacy_format.items()):
            raise StoreModeMismatch(f"vector store at {db_path} has rows and no marker")
        return current
    finally:
        conn.close()


def test_v040_guard_copy_refuses_a_converted_marker(tmp_path):
    """Control: the guard copy really refuses what v0.4.0 refuses (dense_format=f32)."""
    db = str(tmp_path / "vm.db")
    _build_legacy_json_store(db, n_rows=1)
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE store_meta SET value = ? WHERE key = 'mode'",
        ("late_interaction=0;binary_quant=0;dense_format=f32;embedder=qmd:qmd",),
    )
    conn.commit()
    conn.close()
    with pytest.raises(StoreModeMismatch):
        _check_store_mode_v040(db)


def test_v040_check_store_mode_accepts_legacy_store(tmp_path):
    """Round trip: after this code opens a legacy store and adds a row, the
    v0.4.0 store-mode check still accepts it."""
    db = str(tmp_path / "vm.db")
    _build_legacy_json_store(db, n_rows=20)

    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())
    _install_fixed_embedder(vm)
    asyncio.run(vm.add("a row added by the new code"))
    asyncio.run(vm.close())
    marker = _get_marker(db)
    assert marker is not None
    assert "dense_format" not in marker
    _check_store_mode_v040(db)

    # Reopening in the same dense mode is also accepted.
    vm2 = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm2.init())
    asyncio.run(vm2.close())
    _check_store_mode_v040(db)


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


# ---------------------------------------------------------------------------
# Markers exactly as origin/master composes them. Master's
# _store_mode_signature (taosmd/vector_memory.py, lines 583-590) is
#   dense_fmt = "f32" if not binary_quant and not late_interaction else "none"
#   late_interaction=<0|1>;binary_quant=<0|1>;dense_format=<dense_fmt>;embedder=...
# binary_quant and late_interaction are mutually exclusive (ValueError in
# __init__), so "both together" is not a reachable store.
# ---------------------------------------------------------------------------

_MASTER_BQ_MARKER = "late_interaction=0;binary_quant=1;dense_format=none;embedder=qmd:qmd"
_MASTER_LI_MARKER = "late_interaction=1;binary_quant=0;dense_format=none;embedder=qmd:qmd"
_MASTER_DENSE_MARKER = "late_interaction=0;binary_quant=0;dense_format=f32;embedder=qmd:qmd"

_MODES = [
    pytest.param({"binary_quant": True}, _MASTER_BQ_MARKER, id="binary_quant"),
    pytest.param({"late_interaction": True}, _MASTER_LI_MARKER, id="late_interaction"),
]


def _install_token_embedder(vmem: VectorMemory) -> None:
    import numpy as np

    async def _tokens(text: str, task: str = "search_document"):
        out = []
        for word in text.split()[:6] or ["x"]:
            rng = np.random.default_rng(sum(map(ord, word)))
            v = rng.normal(0, 1, vmem._token_dim).astype(np.float32)
            out.append((v / (np.linalg.norm(v) + 1e-8)).tolist())
        return out

    vmem.embed_tokens = _tokens  # type: ignore[assignment]


def _prepare(vm: VectorMemory) -> None:
    _install_fixed_embedder(vm)
    _install_token_embedder(vm)


def _force_marker(db: str, marker: str) -> None:
    conn = sqlite3.connect(db)
    conn.execute("INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', ?)", (marker,))
    conn.commit()
    conn.close()


@pytest.mark.parametrize("kwargs,master_marker", _MODES)
def test_store_written_by_master_reopens_and_searches(tmp_path, kwargs, master_marker):
    db = str(tmp_path / "vm.db")
    vm = VectorMemory(db_path=db, embed_mode="qmd", **kwargs)
    asyncio.run(vm.init())
    _prepare(vm)
    for i in range(5):
        asyncio.run(vm.add(f"master written document number {i}"))
    asyncio.run(vm.close())
    _force_marker(db, master_marker)

    vm2 = VectorMemory(db_path=db, embed_mode="qmd", **kwargs)
    asyncio.run(vm2.init())  # must not raise StoreModeMismatch
    _prepare(vm2)
    hits = asyncio.run(vm2.search("master written document number 3", limit=5, hybrid=False))
    assert len(hits) == 5
    assert _get_marker(db) == master_marker
    asyncio.run(vm2.close())


@pytest.mark.parametrize("kwargs,master_marker", _MODES)
def test_store_with_marker_lacking_dense_format_reopens(tmp_path, kwargs, master_marker):
    """A marker without a dense_format key is accepted and left alone, so 0.4.x
    can still open it."""
    db = str(tmp_path / "vm.db")
    vm = VectorMemory(db_path=db, embed_mode="qmd", **kwargs)
    asyncio.run(vm.init())
    _prepare(vm)
    asyncio.run(vm.add("some document"))
    asyncio.run(vm.close())
    bare = master_marker.replace("dense_format=none;", "")
    _force_marker(db, bare)

    vm2 = VectorMemory(db_path=db, embed_mode="qmd", **kwargs)
    asyncio.run(vm2.init())
    asyncio.run(vm2.close())
    assert _get_marker(db) == bare
    _check_store_mode_v040(
        db,
        late_interaction="late_interaction" in kwargs,
        binary_quant="binary_quant" in kwargs,
    )


@pytest.mark.parametrize("kwargs,master_marker", _MODES)
def test_fresh_bq_li_store_writes_master_marker_and_reopens(tmp_path, kwargs, master_marker):
    db = str(tmp_path / "vm.db")
    vm = VectorMemory(db_path=db, embed_mode="qmd", **kwargs)
    asyncio.run(vm.init())
    asyncio.run(vm.close())
    assert _get_marker(db) == master_marker
    vm2 = VectorMemory(db_path=db, embed_mode="qmd", **kwargs)
    asyncio.run(vm2.init())
    asyncio.run(vm2.close())
    assert _get_marker(db) == master_marker


def test_store_converted_by_master_opens_and_searches(tmp_path):
    """A dense store master already converted (f32 rows, dense_format=f32)."""
    db = str(tmp_path / "vm.db")
    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())
    _install_fixed_embedder(vm)
    for i in range(5):
        asyncio.run(vm.add(f"converted document {i}"))
    asyncio.run(vm.close())
    _force_marker(db, _MASTER_DENSE_MARKER)
    assert _count_json_rows(db) == 0

    vm2 = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm2.init())
    _install_fixed_embedder(vm2)
    assert len(asyncio.run(vm2.search("converted document 2", limit=5, hybrid=False))) == 5
    asyncio.run(vm2.add("one more"))
    assert _count_json_rows(db) == 0
    assert _get_marker(db) == _MASTER_DENSE_MARKER
    asyncio.run(vm2.close())


@pytest.mark.parametrize("marker", [_LEGACY_MARKER, _MASTER_DENSE_MARKER], ids=["legacy", "f32"])
def test_mixed_store_opens_and_searches_every_row(tmp_path, marker):
    """5 f32 rows + 20 JSON rows, under both the legacy and the f32 marker."""
    import base64

    import numpy as np

    db = str(tmp_path / "mixed.db")
    _build_legacy_json_store(db, n_rows=20)
    conn = sqlite3.connect(db)
    fe = FixedEmbedder(384)
    for i in range(5):
        text = f"f32 document {i}"
        vec = np.asarray(asyncio.run(fe.embed(text)), dtype="<f4")
        conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', ?)",
            (text, base64.b64encode(vec.tobytes()).decode("ascii"), 100.0 + i),
        )
    conn.commit()
    conn.close()
    _force_marker(db, marker)

    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())
    _install_fixed_embedder(vm)
    hits = asyncio.run(vm.search("anything", limit=100, hybrid=False))
    assert len(hits) == 25
    assert _count_json_rows(db) == 20
    asyncio.run(vm.close())


def test_legacy_search_results_match_reference(tmp_path):
    """Search on the unconverted legacy store returns the ids and scores a
    reference cosine over the stored JSON vectors gives."""
    import numpy as np

    db = str(tmp_path / "vm.db")
    _build_legacy_json_store(db, n_rows=20)
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT id, embedding FROM vector_memory").fetchall()
    conn.close()
    fe = FixedEmbedder(384)
    q = np.asarray(asyncio.run(fe.embed("legacy document 7 with some content")), dtype=np.float32)
    ref = sorted(
        ((float(np.dot(q, np.asarray(json.loads(e), dtype=np.float32))), i) for i, e in rows),
        reverse=True,
    )[:5]

    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())
    _install_fixed_embedder(vm)
    hits = asyncio.run(vm.search("legacy document 7 with some content", limit=5, hybrid=False))
    assert [h["id"] for h in hits] == [i for _, i in ref]
    for h, (score, _) in zip(hits, ref):
        assert abs(h["similarity"] - score) < 2e-4
    asyncio.run(vm.close())


def test_no_marker_legacy_store_marker_written_without_dense_format(tmp_path):
    db = str(tmp_path / "vm.db")
    _build_legacy_json_store(db, n_rows=5, with_marker=False)
    vm = VectorMemory(db_path=db, embed_mode="qmd")
    asyncio.run(vm.init())
    _install_fixed_embedder(vm)
    asyncio.run(vm.add("x"))
    asyncio.run(vm.close())
    assert "dense_format" not in _get_marker(db)
    assert _count_json_rows(db) == 6
