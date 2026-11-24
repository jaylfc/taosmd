"""Tests for the guarded vector store conversion (taosmd migrate --convert-vectors)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
import tracemalloc
import zlib
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from taosmd.vector_memory import SCHEMA, VectorMemory

try:
    from taosmd import store_convert
    from taosmd.store_convert import convert_dense_store
except ImportError:  # keeps the file collectable on a tree without the module
    store_convert = None  # type: ignore[assignment]

    def convert_dense_store(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("taosmd.store_convert is not present in this tree")

LEGACY_MARKER = "late_interaction=0;binary_quant=0;embedder=all-MiniLM-L6-v2"
CURRENT_MARKER = "late_interaction=0;binary_quant=0;dense_format=f32;embedder=all-MiniLM-L6-v2"


def _vec(i: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(i + 1000)
    v = rng.normal(0, 1, dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.asarray(arr, dtype="<f4").tobytes()).decode("ascii")


def make_store(
    path: Path,
    n: int = 30,
    dim: int = 8,
    marker: str | None = LEGACY_MARKER,
    f32_ids: tuple[int, ...] = (),
    overrides: dict[int, str] | None = None,
    float64: bool = False,
    wal: bool = True,
) -> Path:
    """Build a legacy store: JSON embeddings (some optionally f32), raw sqlite."""
    from taosmd import migrations

    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA journal_mode={'WAL' if wal else 'DELETE'}")
    conn.executescript(SCHEMA)
    for i in range(1, n + 1):
        v = _vec(i, dim)
        if i in f32_ids:
            emb = _f32(v)
        elif float64:
            emb = json.dumps((v.astype(np.float64) * 1.0000001234567).tolist())
        else:
            emb = json.dumps(v.tolist())
        if overrides and i in overrides:
            emb = overrides[i]
        conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at, valid_to) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"memory number {i}", emb, json.dumps({"i": i}), 1000.0 + i, 5000.0 if i % 7 == 0 else None),
        )
    if marker is not None:
        conn.execute("INSERT INTO store_meta (key, value) VALUES ('mode', ?)", (marker,))
    conn.execute(f"PRAGMA user_version = {migrations.latest_version('vector_memory')}")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return path


def file_hash(path: Path) -> str:
    """sha256 of the db file after a checkpoint, so WAL content is folded in."""
    c = sqlite3.connect(path)
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path: Path):
    c = sqlite3.connect(path)
    try:
        return c.execute(
            "SELECT id, embedding, text, metadata_json, created_at, valid_to FROM vector_memory ORDER BY id"
        ).fetchall()
    finally:
        c.close()


def marker_of(path: Path):
    c = sqlite3.connect(path)
    try:
        r = c.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
        return r[0] if r else None
    finally:
        c.close()


def n_json(path: Path) -> int:
    c = sqlite3.connect(path)
    try:
        return c.execute("SELECT count(*) FROM vector_memory WHERE embedding LIKE '[%'").fetchone()[0]
    finally:
        c.close()


def backups(path: Path) -> list[Path]:
    d = path.parent / "backups"
    return sorted(d.iterdir()) if d.exists() else []


@pytest.fixture
def store(tmp_path):
    return make_store(tmp_path / "vector-memory.db")


# ----------------------------------------------------------------------
# happy path and the product read path
# ----------------------------------------------------------------------

def test_converts_every_row_exactly(store):
    before = rows(store)
    res = convert_dense_store(store)
    assert res.exit_code == 0 and res.status == "converted", res.message
    assert res.rows_converted == 30 and res.dimension == 8
    after = rows(store)
    assert n_json(store) == 0
    for (i, emb_b, *rest_b), (j, emb_a, *rest_a) in zip(before, after):
        assert i == j and rest_b == rest_a
        assert emb_a == _f32(np.asarray(json.loads(emb_b), dtype="<f4"))
    assert marker_of(store) == CURRENT_MARKER


def test_float64_origin_json_is_cast_to_float32(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", float64=True)
    before = rows(p)
    assert convert_dense_store(p).exit_code == 0
    for (_, eb, *_), (_, ea, *_) in zip(before, rows(p)):
        assert np.array_equal(
            np.frombuffer(base64.b64decode(ea), "<f4"), np.asarray(json.loads(eb), dtype="<f4")
        )


def test_backup_is_named_verified_and_identical(store):
    before = rows(store)
    res = convert_dense_store(store)
    files = backups(store)
    assert len(files) == 1
    b = files[0]
    assert str(b) == res.backup_path
    assert b.name.startswith("vector-memory.") and b.name.endswith(".pre-f32.db")
    assert not any(p.name.endswith((".partial", "-wal", "-shm")) for p in files)
    assert rows(b) == before
    c = sqlite3.connect(b)
    assert c.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    c.close()


def test_custom_backup_dir(store, tmp_path):
    d = tmp_path / "elsewhere"
    res = convert_dense_store(store, backup_dir=d)
    assert res.exit_code == 0
    assert len(list(d.iterdir())) == 1 and not backups(store)


class _Fake:
    """Deterministic embedder shared by the store builder and VectorMemory."""

    dim = 384

    @classmethod
    def vec(cls, text: str) -> list[float]:
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        v = rng.normal(0, 1, cls.dim).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()


def _patch_embed(vm: VectorMemory) -> None:
    async def _embed(text, task="search_document"):
        return _Fake.vec(text)

    vm.embed = _embed  # type: ignore[assignment]


def _search_all(db: Path, queries, monkeypatch):
    # Opening a legacy store no longer converts it, so the "before" read sees JSON rows.
    async def go():
        vm = VectorMemory(db_path=str(db), embed_mode="local", local_model="all-MiniLM-L6-v2")
        await vm.init()
        _patch_embed(vm)
        out = []
        for q in queries:
            res = await vm.search(q, limit=8, hybrid=False)
            out.append([(r["id"], r["similarity"]) for r in res])
        await vm.close()
        return out

    return asyncio.run(go())


def test_product_read_path_matches_before_and_after(tmp_path, monkeypatch):
    # Pin the embedder identity so every store here, fresh or converted, records
    # the same one whether or not sentence-transformers is installed (without
    # it VectorMemory falls back to qmd and would record embedder=qmd:qmd).
    monkeypatch.setattr(VectorMemory, "_embedder_identity", lambda self: "all-MiniLM-L6-v2")
    p = tmp_path / "vector-memory.db"
    c = sqlite3.connect(p)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    for i in range(120):
        t = f"document {i} about subject {i % 9}"
        c.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, '{}', ?)",
            (t, json.dumps(_Fake.vec(t)), float(i)),
        )
    c.execute("INSERT INTO store_meta (key, value) VALUES ('mode', ?)", (LEGACY_MARKER,))
    c.execute("PRAGMA user_version = 2")
    c.commit()
    c.close()

    queries = [f"document {i} about subject {i % 9}" for i in (3, 40, 77, 101)] + ["unrelated words"]
    # "Before": a copy of the untouched JSON store, read through the product.
    legacy_copy = tmp_path / "copy" / "vector-memory.db"
    legacy_copy.parent.mkdir()
    shutil.copy(p, legacy_copy)
    pre = _search_all(legacy_copy, queries, monkeypatch)
    assert n_json(legacy_copy) == 120  # the copy really was still JSON

    res = convert_dense_store(p)
    assert res.exit_code == 0
    post = _search_all(p, queries, monkeypatch)
    for a, b in zip(pre, post):
        assert [i for i, _ in a] == [i for i, _ in b]
        assert max(abs(x - y) for (_, x), (_, y) in zip(a, b)) <= 1e-6
    assert any(a for a in pre)

    # The marker and the row bytes are what a fresh store of this product holds.
    async def fresh():
        vm = VectorMemory(
            db_path=str(tmp_path / "fresh.db"), embed_mode="local", local_model="all-MiniLM-L6-v2"
        )
        await vm.init()
        _patch_embed(vm)
        await vm.add("document 3 about subject 3")
        sig = vm._store_mode_signature()
        emb = vm._conn.execute("SELECT embedding FROM vector_memory").fetchone()[0]
        await vm.close()
        return sig, emb

    sig, emb = asyncio.run(fresh())
    assert marker_of(p) == sig
    conv = sqlite3.connect(p)
    got = conv.execute("SELECT embedding FROM vector_memory WHERE text = ?", ("document 3 about subject 3",)).fetchone()[0]
    conv.close()
    assert got == emb


# ----------------------------------------------------------------------
# exit 0: already current / empty / sizes / mixed / re-run
# ----------------------------------------------------------------------

def test_running_twice_second_is_noop(store):
    assert convert_dense_store(store).exit_code == 0
    h = file_hash(store)
    again = convert_dense_store(store)
    assert again.exit_code == 0 and again.status == "already_current"
    assert file_hash(store) == h
    assert len(backups(store)) == 1  # no second backup for a no-op


def test_empty_store_legacy_marker_writes_marker(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=0)
    res = convert_dense_store(p)
    assert res.exit_code == 0 and res.status == "converted"
    assert marker_of(p) == CURRENT_MARKER and rows(p) == []


def test_empty_store_without_marker_is_current(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=0, marker=None)
    h = file_hash(p)
    res = convert_dense_store(p)
    assert res.exit_code == 0 and res.status == "already_current"
    assert file_hash(p) == h and not backups(p)


def test_clean_768_dimension_store(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=40, dim=768, marker="late_interaction=0;binary_quant=0")
    res = convert_dense_store(p)
    assert res.exit_code == 0 and res.dimension == 768 and res.rows_converted == 40
    r = rows(p)
    assert all(len(base64.b64decode(x[1])) == 768 * 4 for x in r)
    # An embedder-less legacy marker keeps no embedder key (the product adds it on open).
    assert marker_of(p) == "late_interaction=0;binary_quant=0;dense_format=f32"


def test_no_marker_store_with_rows(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", marker=None)
    assert convert_dense_store(p).exit_code == 0
    assert marker_of(p) == "late_interaction=0;binary_quant=0;dense_format=f32"


def test_mixed_rows_some_already_f32(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=20, f32_ids=(2, 3, 11))
    before = {r[0]: r for r in rows(p)}
    res = convert_dense_store(p)
    assert res.exit_code == 0 and res.rows_converted == 17
    for r in rows(p):
        if r[0] in (2, 3, 11):
            assert r[1] == before[r[0]][1]
        else:
            assert r[1] == _f32(np.asarray(json.loads(before[r[0]][1]), dtype="<f4"))


def test_f32_marker_with_leftover_json_rows_is_healed(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=10, marker=CURRENT_MARKER, f32_ids=(1, 2, 3, 4, 5))
    res = convert_dense_store(p)
    assert res.exit_code == 0 and res.rows_converted == 5 and n_json(p) == 0
    assert marker_of(p) == CURRENT_MARKER


def test_dimension_comes_from_the_store_not_an_embedder(tmp_path):
    # 4 odd-length rows among 400: the store dimension is the common length.
    p = make_store(
        tmp_path / "vector-memory.db", n=400, dim=24,
        overrides={5: json.dumps([0.1] * 7), 6: json.dumps([0.1] * 7), 7: json.dumps([0.2] * 5)},
    )
    res = convert_dense_store(p)
    assert res.exit_code == 0 and res.dimension == 24
    assert sorted(i for i, _ in res.malformed) == [5, 6, 7]
    assert "wrong length 7" in dict(res.malformed)[5]


def test_already_f32_row_of_the_wrong_length_is_reported(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=300)
    c = sqlite3.connect(p)
    odd = _f32(np.arange(5, dtype=np.float32))
    c.execute("UPDATE vector_memory SET embedding = ? WHERE id = 9", (odd,))
    c.commit()
    c.close()
    res = convert_dense_store(p)
    assert res.exit_code == 0 and dict(res.malformed)[9].startswith("wrong length 5")
    assert [r[1] for r in rows(p) if r[0] == 9] == [odd]


def test_wal_content_is_in_the_backup(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    p = src / "vector-memory.db"
    make_store(p, n=5)
    live = sqlite3.connect(p)
    live.execute("PRAGMA wal_autocheckpoint=0")
    v = _vec(99, 8)
    live.execute(
        "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES ('in wal', ?, '{}', 9.0)",
        (json.dumps(v.tolist()),),
    )
    live.commit()
    dst = tmp_path / "dst"
    dst.mkdir()
    shutil.copy(p, dst / "vector-memory.db")
    shutil.copy(str(p) + "-wal", str(dst / "vector-memory.db") + "-wal")
    live.close()
    q = dst / "vector-memory.db"
    res = convert_dense_store(q)
    assert res.exit_code == 0 and res.rows_total == 6
    b = Path(res.backup_path)
    assert [r[2] for r in rows(b)][-1] == "in wal"
    assert [r[2] for r in rows(q)][-1] == "in wal"


def test_stale_partial_backup_is_cleared_and_leftover_backups_kept(store):
    d = store.parent / "backups"
    d.mkdir()
    (d / "vector-memory.20200101T000000000000Z.pre-f32.db.partial").write_bytes(b"junk")
    old = d / "vector-memory.20200101T000000000001Z.pre-f32.db"
    old.write_bytes(b"an earlier verified backup")
    assert convert_dense_store(store).exit_code == 0
    names = [p.name for p in backups(store)]
    assert old.name in names and not any(n.endswith(".partial") for n in names)


# ----------------------------------------------------------------------
# exit 2: refused, nothing touched
# ----------------------------------------------------------------------

def test_exit2_when_another_connection_holds_a_write_lock(store):
    h = file_hash(store)
    other = sqlite3.connect(store, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        res = convert_dense_store(store)
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert res.exit_code == 2 and "in use" in res.message
    assert file_hash(store) == h and not backups(store)


def test_exit2_when_another_connection_is_merely_open(store):
    """An idle open connection (a running serve) is also detected."""
    h = file_hash(store)
    other = sqlite3.connect(store)
    other.execute("SELECT count(*) FROM vector_memory").fetchall()
    try:
        res = convert_dense_store(store)
    finally:
        other.close()
    assert res.exit_code == 2
    assert file_hash(store) == h and not backups(store)
    assert n_json(store) == 30


def test_non_wal_store_converts(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", wal=False)
    res = convert_dense_store(p)
    assert res.exit_code == 0 and res.rows_converted == 30
    assert marker_of(p) == CURRENT_MARKER and n_json(p) == 0


def test_exit2_non_wal_store_with_another_writer(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", wal=False)
    h = file_hash(p)
    other = sqlite3.connect(p, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        res = convert_dense_store(p)
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert res.exit_code == 2, res.message
    assert file_hash(p) == h and not backups(p)


def test_converted_store_in_use_answers_already_current(store):
    assert convert_dense_store(store).exit_code == 0
    other = sqlite3.connect(store)
    other.execute("SELECT count(*) FROM vector_memory").fetchall()
    try:
        res = convert_dense_store(store)
    finally:
        other.close()
    assert res.exit_code == 0 and res.status == "already_current"
    assert len(backups(store)) == 1


def test_newer_store_in_use_answers_unknown_format(tmp_path):
    p = make_store(tmp_path / "vector-memory.db")
    c = sqlite3.connect(p)
    c.execute("PRAGMA user_version = 99")
    c.commit()
    c.close()
    other = sqlite3.connect(p)
    other.execute("SELECT count(*) FROM vector_memory").fetchall()
    try:
        res = convert_dense_store(p)
    finally:
        other.close()
    assert res.exit_code == 3 and not backups(p)


_Usage = namedtuple("usage", "total used free")


def test_exit2_when_free_space_is_too_low(store, monkeypatch):
    h = file_hash(store)
    monkeypatch.setattr(store_convert.shutil, "disk_usage", lambda p: _Usage(10**9, 10**9, 10))
    res = convert_dense_store(store)
    assert res.exit_code == 2 and "free" in res.message
    assert file_hash(store) == h and not backups(store)


def test_exit2_when_space_covers_the_backup_but_not_the_conversion(store, monkeypatch):
    h = file_hash(store)
    size = store.stat().st_size
    need_backup = int(size * 1.10)
    monkeypatch.setattr(
        store_convert.shutil, "disk_usage", lambda p: _Usage(10**9, 0, need_backup + size // 2)
    )
    res = convert_dense_store(store)
    assert res.exit_code == 2 and "free disk space" in res.message
    assert file_hash(store) == h and not backups(store)
    # with room for both the backup and the conversion it goes ahead
    monkeypatch.setattr(
        store_convert.shutil, "disk_usage", lambda p: _Usage(10**9, 0, need_backup + size + 4096)
    )
    assert convert_dense_store(store).exit_code == 0


def test_exit2_when_only_the_backup_filesystem_is_short(store, tmp_path, monkeypatch):
    h = file_hash(store)
    real = shutil.disk_usage
    bdir = tmp_path / "bk"
    bdir.mkdir()
    monkeypatch.setattr(store_convert, "_same_filesystem", lambda a, b: False)
    monkeypatch.setattr(
        store_convert.shutil, "disk_usage",
        lambda p: _Usage(1, 1, 5) if Path(p).resolve() == bdir.resolve() else real(p),
    )
    res = convert_dense_store(store, backup_dir=bdir)
    assert res.exit_code == 2 and "backup" in res.message
    assert file_hash(store) == h and list(bdir.iterdir()) == []


def test_exit2_when_only_the_store_filesystem_is_short(store, tmp_path, monkeypatch):
    h = file_hash(store)
    real = shutil.disk_usage
    bdir = tmp_path / "bk"
    bdir.mkdir()
    monkeypatch.setattr(store_convert, "_same_filesystem", lambda a, b: False)
    monkeypatch.setattr(
        store_convert.shutil, "disk_usage",
        lambda p: _Usage(1, 1, 5) if Path(p).resolve() == store.parent.resolve() else real(p),
    )
    res = convert_dense_store(store, backup_dir=bdir)
    assert res.exit_code == 2 and "conversion" in res.message
    assert file_hash(store) == h and list(bdir.iterdir()) == []


def test_exit2_when_backup_hash_mismatches(store, monkeypatch):
    h = file_hash(store)
    real = store_convert._hash_rows
    calls = []

    def fake(conn, cols):
        calls.append(1)
        out = real(conn, cols)
        return out if len(calls) != 2 else "0" * 64  # the backup verification call

    monkeypatch.setattr(store_convert, "_hash_rows", fake)
    res = convert_dense_store(store)
    assert res.exit_code == 2 and "hash" in res.message
    assert file_hash(store) == h
    assert backups(store) == []  # partial backup deleted


def test_exit2_when_backup_row_count_mismatches(store, monkeypatch):
    h = file_hash(store)
    real = store_convert._count
    seen = []

    def fake(conn):
        seen.append(1)
        n = real(conn)
        return n + 1 if len(seen) == 1 else n  # first call after backup is the verify

    # _count is first used by the backup verification (the prescan streams ids).
    monkeypatch.setattr(store_convert, "_count", fake)
    res = convert_dense_store(store)
    assert res.exit_code == 2 and "row count" in res.message
    assert file_hash(store) == h and backups(store) == []


def test_exit2_when_backup_integrity_fails(store, monkeypatch):
    h = file_hash(store)

    class Wrap:
        def __init__(self, c):
            self.c = c

        def execute(self, sql, *a):
            if sql.strip().upper().startswith("PRAGMA INTEGRITY_CHECK"):
                class R:
                    def fetchall(self_):
                        return [("page 3 is corrupt",)]
                return R()
            return self.c.execute(sql, *a)

        def close(self):
            self.c.close()

    real = store_convert._open_ro
    monkeypatch.setattr(store_convert, "_open_ro", lambda p: Wrap(real(p)))
    res = convert_dense_store(store)
    assert res.exit_code == 2 and "integrity" in res.message
    assert file_hash(store) == h and backups(store) == []


def test_exit2_over_one_percent_malformed(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=200, overrides={3: "[0.1, 0.2", 4: "nope", 5: "[]"})
    h = file_hash(p)
    res = convert_dense_store(p)
    assert res.exit_code == 2 and "more than 1%" in res.message
    assert file_hash(p) == h and not backups(p) and n_json(p) == 199


def test_exactly_one_percent_malformed_is_allowed(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=100, overrides={3: "[0.1, 0.2"})
    res = convert_dense_store(p)
    assert res.exit_code == 0 and res.rows_converted == 99


def test_exit2_missing_file(tmp_path):
    res = convert_dense_store(tmp_path / "vector-memory.db")
    assert res.exit_code == 2 and not (tmp_path / "vector-memory.db").exists()


def test_exit2_schema_older_than_valid_to(tmp_path):
    p = tmp_path / "vector-memory.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE vector_memory (id INTEGER PRIMARY KEY, text TEXT, embedding TEXT, "
              "metadata_json TEXT, created_at REAL)")
    c.execute("INSERT INTO vector_memory VALUES (1, 'a', '[0.1, 0.2]', '{}', 1.0)")
    c.commit()
    c.close()
    h = file_hash(p)
    res = convert_dense_store(p)
    assert res.exit_code == 2 and "taosmd migrate" in res.message
    assert file_hash(p) == h and not backups(p)


# ----------------------------------------------------------------------
# exit 3: unknown / newer
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "marker",
    [
        "late_interaction=0;binary_quant=0;dense_format=f64;embedder=x",
        "late_interaction=0;binary_quant=0;shiny_new_key=1;embedder=x",
        "late_interaction=1;binary_quant=0",
        "late_interaction=0;binary_quant=1",
        "garbage without equals",
    ],
)
def test_exit3_unknown_marker(tmp_path, marker):
    p = make_store(tmp_path / "vector-memory.db", marker=marker)
    h = file_hash(p)
    res = convert_dense_store(p)
    assert res.exit_code == 3, res.message
    assert file_hash(p) == h and not backups(p)


def test_exit3_newer_schema_version(tmp_path):
    p = make_store(tmp_path / "vector-memory.db")
    c = sqlite3.connect(p)
    c.execute("PRAGMA user_version = 99")
    c.commit()
    c.close()
    h = file_hash(p)
    res = convert_dense_store(p)
    assert res.exit_code == 3 and "newer" in res.message
    assert file_hash(p) == h and not backups(p)


def test_exit3_not_a_sqlite_file(tmp_path):
    p = tmp_path / "vector-memory.db"
    p.write_bytes(b"this is not a database" * 100)
    before = p.read_bytes()
    res = convert_dense_store(p)
    assert res.exit_code == 3
    assert p.read_bytes() == before


def test_exit3_no_vector_table(tmp_path):
    p = tmp_path / "vector-memory.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE other (x)")
    c.commit()
    c.close()
    assert convert_dense_store(p).exit_code == 3


# ----------------------------------------------------------------------
# malformed rows
# ----------------------------------------------------------------------

BAD = {
    3: ("[0.1, 0.2", "invalid JSON"),
    4: ("[]", "empty list"),
    5: ('[0.1, "x", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]', "non-numeric"),
    6: ("[0.1, 0.2, 0.3]", "wrong length 3"),
    7: ('{"a": 1}', "unrecognised"),
    8: ("[null, 1, 2, 3, 4, 5, 6, 7]", "non-numeric"),
    9: ("[true, 1, 2, 3, 4, 5, 6, 7]", "non-numeric"),
    10: ("[NaN, 1, 2, 3, 4, 5, 6, 7]", "non-finite"),
    11: ("not an embedding!", "unrecognised"),
    12: ("[1e999, 1, 2, 3, 4, 5, 6, 7]", "non-finite"),
}


def test_malformed_rows_left_identical_and_listed(tmp_path):
    n = 2000  # 10 bad rows is 0.5%, under the 1% limit
    p = make_store(tmp_path / "vector-memory.db", n=n, overrides={k: v[0] for k, v in BAD.items()})
    before = {r[0]: r for r in rows(p)}
    res = convert_dense_store(p)
    assert res.exit_code == 0, res.message
    assert dict(res.malformed).keys() == BAD.keys()
    for rid, (_, frag) in BAD.items():
        assert frag in dict(res.malformed)[rid], (rid, dict(res.malformed)[rid])
    assert res.rows_converted == n - len(BAD)
    for r in rows(p):
        if r[0] in BAD:
            assert r == before[r[0]]  # byte-identical row
        else:
            assert not r[1].startswith("[")


# ----------------------------------------------------------------------
# exit 4: post-check failures restore the store
# ----------------------------------------------------------------------

def test_conversion_detects_corrupted_vectors(store, monkeypatch):
    """The red-first test: an encoder that rotates every vector by one."""
    h = file_hash(store)
    before = rows(store)
    monkeypatch.setattr(
        store_convert, "_encode_row", lambda arr: _f32(np.roll(np.asarray(arr, dtype="<f4"), 1))
    )
    res = convert_dense_store(store)
    assert res.exit_code == 4 and res.status == "restored"
    assert file_hash(store) == h
    assert rows(store) == before and n_json(store) == 30
    assert marker_of(store) == LEGACY_MARKER
    assert res.backup_path and Path(res.backup_path).exists()
    # the next, healthy run succeeds
    monkeypatch.undo()
    assert convert_dense_store(store).exit_code == 0


def test_exit4_on_tiny_value_drift(store, monkeypatch):
    h = file_hash(store)
    monkeypatch.setattr(
        store_convert, "_encode_row",
        lambda arr: _f32(np.asarray(arr, dtype="<f4") + np.float32(1e-6)),
    )
    res = convert_dense_store(store)
    assert res.exit_code == 4
    assert file_hash(store) == h


def test_exit4_when_a_non_embedding_column_is_touched(store, monkeypatch):
    h = file_hash(store)
    real = store_convert._convert_rows

    def evil(conn, dim, result):
        real(conn, dim, result)
        conn.execute("UPDATE vector_memory SET text = text || '!' WHERE id = 4")

    monkeypatch.setattr(store_convert, "_convert_rows", evil)
    res = convert_dense_store(store)
    assert res.exit_code == 4 and "non-embedding" in res.message
    assert file_hash(store) == h


def test_exit4_when_a_row_is_lost(store, monkeypatch):
    h = file_hash(store)
    real = store_convert._convert_rows

    def evil(conn, dim, result):
        real(conn, dim, result)
        conn.execute("DELETE FROM vector_memory WHERE id = 9")

    monkeypatch.setattr(store_convert, "_convert_rows", evil)
    res = convert_dense_store(store)
    assert res.exit_code == 4 and "row count" in res.message
    assert file_hash(store) == h


def test_exit4_when_a_row_is_swapped_but_count_matches(store, monkeypatch):
    h = file_hash(store)
    real = store_convert._convert_rows

    def evil(conn, dim, result):
        real(conn, dim, result)
        conn.execute("UPDATE vector_memory SET embedding = (SELECT embedding FROM vector_memory WHERE id = 2) WHERE id = 3")

    monkeypatch.setattr(store_convert, "_convert_rows", evil)
    res = convert_dense_store(store)
    assert res.exit_code == 4
    assert file_hash(store) == h


def test_exit4_when_an_untouched_row_changes(tmp_path, monkeypatch):
    p = make_store(tmp_path / "vector-memory.db", n=300, overrides={3: "[0.1, 0.2"})
    h = file_hash(p)
    real = store_convert._convert_rows

    def evil(conn, dim, result):
        real(conn, dim, result)
        conn.execute("UPDATE vector_memory SET embedding = '[0.1, 0.3' WHERE id = 3")

    monkeypatch.setattr(store_convert, "_convert_rows", evil)
    res = convert_dense_store(p)
    assert res.exit_code == 4 and "untouched" in res.message
    assert file_hash(p) == h


def test_exit4_when_the_marker_is_not_written(store, monkeypatch):
    h = file_hash(store)
    real = store_convert._post_checks

    def after_marker_removed(live, *a):
        live.execute("DELETE FROM store_meta WHERE key = 'mode'")
        return real(live, *a)

    monkeypatch.setattr(store_convert, "_post_checks", after_marker_removed)
    res = convert_dense_store(store)
    assert res.exit_code == 4 and "marker" in res.message
    assert file_hash(store) == h


def _skew_live_side(monkeypatch, vecs_fn=None, ids_fn=None):
    real = store_convert._TopK.add
    state = {"n": 0}

    def skewed(self, ids, vecs):
        state["n"] += 1
        if state["n"] % 2 == 0:  # calls alternate backup side, live side
            if vecs_fn:
                vecs = vecs_fn(np.asarray(vecs).copy())
            if ids_fn:
                ids = ids_fn(np.asarray(ids).copy())
        return real(self, ids, vecs)

    monkeypatch.setattr(store_convert._TopK, "add", skewed)


def _nudge(v):
    v[:, 0] += 1e-3  # same ids, scores off by about 1e-3
    return v


def test_recall_check_rejects_shifted_scores(store, monkeypatch):
    """Per-row checks pass, same top-10 ids, but scores differ by more than 1e-6."""
    h = file_hash(store)
    _skew_live_side(monkeypatch, vecs_fn=_nudge)
    res = convert_dense_store(store)
    assert res.exit_code == 4 and "scores differ" in res.message
    assert file_hash(store) == h


def test_recall_check_rejects_different_top_ids(store, monkeypatch):
    h = file_hash(store)
    _skew_live_side(monkeypatch, ids_fn=lambda ids: ids + 1_000_000)
    res = convert_dense_store(store)
    assert res.exit_code == 4 and "ids differ" in res.message
    assert file_hash(store) == h


def test_per_row_check_is_exact_not_a_tolerance(store, monkeypatch):
    """One float32 ulp in one component is far inside the recall tolerance, so
    only the exact per-row comparison can see it."""
    h = file_hash(store)

    def one_ulp(arr):
        a = np.asarray(arr, dtype="<f4").copy()
        a[0] = np.nextafter(a[0], np.float32(np.inf))
        return _f32(a)

    monkeypatch.setattr(store_convert, "_encode_row", one_ulp)
    res = convert_dense_store(store)
    assert res.exit_code == 4 and "differs from the original" in res.message
    assert file_hash(store) == h


def test_exit4_when_an_id_changes_but_the_count_does_not(store, monkeypatch):
    h = file_hash(store)
    real = store_convert._convert_rows

    def evil(conn, dim, result):
        real(conn, dim, result)
        conn.execute("UPDATE vector_memory SET id = 100005 WHERE id = 5")

    monkeypatch.setattr(store_convert, "_convert_rows", evil)
    res = convert_dense_store(store)
    assert res.exit_code == 4 and "row ids differ" in res.message
    assert file_hash(store) == h


def test_wrong_length_row_never_becomes_a_recall_query(tmp_path):
    # Row 1 is structurally valid but the wrong length; it is always among the
    # sampled candidate queries, so it must be filtered out of the probe.
    p = make_store(tmp_path / "vector-memory.db", n=100, dim=16, overrides={1: json.dumps([0.5] * 9)})
    res = convert_dense_store(p)
    assert res.exit_code == 0, res.message
    assert dict(res.malformed)[1].startswith("wrong length 9")


def test_post_commit_failure_restores_from_backup(store, monkeypatch):
    before = rows(store)
    monkeypatch.setattr(store_convert, "_post_commit_check", lambda db, conn: "simulated damage")
    res = convert_dense_store(store)
    assert res.exit_code == 4 and "restored" in res.message
    assert rows(store) == before and marker_of(store) == LEGACY_MARKER
    c = sqlite3.connect(store)
    assert c.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    c.close()


def test_exit1_unexpected_error_names_the_backup(store, monkeypatch):
    h = file_hash(store)

    def boom(arr):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(store_convert, "_encode_row", boom)
    res = convert_dense_store(store)
    assert res.exit_code == 1 and "disk on fire" in res.message
    assert res.backup_path and res.backup_path in res.message
    assert file_hash(store) == h


# ----------------------------------------------------------------------
# interruption and scale
# ----------------------------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from taosmd import store_convert
    db, flag = sys.argv[1], sys.argv[2]
    real = store_convert._encode_row
    n = [0]
    def slow(arr):
        n[0] += 1
        if n[0] == 2500:
            Path(flag).write_text("mid-conversion")
            time.sleep(120)
        return real(arr)
    store_convert._encode_row = slow
    store_convert.convert_dense_store(db)
    """
)


def test_sigkill_mid_conversion_then_rerun_is_correct(tmp_path):
    p = make_store(tmp_path / "vector-memory.db", n=3000, dim=128)
    before = rows(p)
    flag = tmp_path / "flag"
    child = subprocess.Popen([sys.executable, "-c", _CHILD, str(p), str(flag)])
    try:
        deadline = time.time() + 60
        while not flag.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert flag.exists(), "child never reached mid-conversion"
        # Uncommitted frames really reached the WAL, so the kill is not trivial.
        assert Path(str(p) + "-wal").stat().st_size > 0
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(timeout=30)

    c = sqlite3.connect(p)
    assert c.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    c.close()
    assert rows(p) == before and marker_of(p) == LEGACY_MARKER  # old store intact

    res = convert_dense_store(p)
    assert res.exit_code == 0, res.message
    for (i, eb, *rb), (j, ea, *ra) in zip(before, rows(p)):
        assert i == j and rb == ra
        assert ea == _f32(np.asarray(json.loads(eb), dtype="<f4"))
    assert marker_of(p) == CURRENT_MARKER
    for b in backups(p):  # every backup, including the interrupted run's, is whole
        bc = sqlite3.connect(b)
        assert bc.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        bc.close()
        assert rows(b) == before


def test_50k_rows_convert_with_bounded_memory(tmp_path):
    p = tmp_path / "vector-memory.db"
    dim, n = 32, 50_000
    c = sqlite3.connect(p)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (500, dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    texts = [json.dumps(b.tolist()) for b in base]

    def gen():
        for i in range(n):
            yield (f"row {i}", texts[i % 500], "{}", float(i))

    c.executemany("INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, ?, ?)", gen())
    c.execute("INSERT INTO store_meta (key, value) VALUES ('mode', ?)", (LEGACY_MARKER,))
    c.execute("PRAGMA user_version = 2")
    c.commit()
    c.close()
    json_bytes = sum(len(t) for t in texts) * (n // 500)

    tracemalloc.start()
    try:
        res = convert_dense_store(p)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res.exit_code == 0 and res.rows_converted == n, res.message
    assert n_json(p) == 0
    # Loading every vector would take more than the JSON text itself (>25 MB here).
    assert json_bytes > 25_000_000
    assert peak < 12_000_000, f"peak {peak} bytes is not bounded"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def test_cli_convert_vectors(tmp_path, capsys):
    from taosmd.cli import main

    p = make_store(tmp_path / "vector-memory.db", n=12)
    assert main(["--data-dir", str(tmp_path), "migrate", "--convert-vectors"]) == 0
    out = capsys.readouterr().out
    assert "converted 12 of 12" in out and "backup:" in out and ".pre-f32.db" in out
    assert n_json(p) == 0
    assert main(["--data-dir", str(tmp_path), "migrate", "--convert-vectors", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "already_current"


def test_cli_convert_vectors_refuses_read_only_flags(tmp_path, capsys):
    from taosmd.cli import main

    make_store(tmp_path / "vector-memory.db")
    h = file_hash(tmp_path / "vector-memory.db")
    for flag in ("--probe", "--status", "--check"):
        assert main(["--data-dir", str(tmp_path), "migrate", "--convert-vectors", flag]) == 2
        assert "cannot be combined" in capsys.readouterr().err
    assert file_hash(tmp_path / "vector-memory.db") == h
    assert not backups(tmp_path / "vector-memory.db")


def test_cli_exit_codes_pass_through(tmp_path, capsys):
    from taosmd.cli import main

    assert main(["--data-dir", str(tmp_path), "migrate", "--convert-vectors"]) == 2
    make_store(tmp_path / "vector-memory.db", marker="late_interaction=1;binary_quant=0")
    assert main(["--data-dir", str(tmp_path), "migrate", "--convert-vectors"]) == 3


def test_cli_plain_migrate_does_not_convert(tmp_path, capsys):
    from taosmd.cli import main

    p = make_store(tmp_path / "vector-memory.db", n=5)
    main(["--data-dir", str(tmp_path), "migrate"])
    capsys.readouterr()
    assert n_json(p) == 5 and not backups(p)
