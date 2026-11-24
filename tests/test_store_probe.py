"""Read-only store format probe (`taosmd migrate --probe`, `probe_store`)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import struct
from pathlib import Path

import pytest

from taosmd import cli, migrations

VEC = "vector-memory.db"
LEGACY_MARKER = "late_interaction=0;binary_quant=0;embedder=minilm-onnx"
F32_MARKER = "late_interaction=0;binary_quant=0;dense_format=f32;embedder=minilm-onnx"


def _f32(vec):
    return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode()


def _make_vector_db(path: Path, *, marker, rows, version=None, wal=False):
    from taosmd.vector_memory import SCHEMA

    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    if marker is not None:
        conn.execute("INSERT INTO store_meta (key, value) VALUES ('mode', ?)", (marker,))
    for i, emb in enumerate(rows):
        conn.execute(
            "INSERT INTO vector_memory (text, embedding, created_at) VALUES (?, ?, 1.0)",
            (f"t{i}", emb),
        )
    conn.execute(f"PRAGMA user_version={migrations.latest_version('vector_memory') if version is None else version}")
    conn.commit()
    conn.close()


def _make_stamped(path: Path, db: str, version=None):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS t (x)")
    conn.execute(f"PRAGMA user_version={migrations.latest_version(db) if version is None else version}")
    conn.commit()
    conn.close()


def _stamp_others(d: Path):
    for db, fname in migrations.DB_FILES.items():
        if db != "vector_memory":
            _make_stamped(d / fname, db)


def _snapshot(d: Path):
    out = {}
    for p in sorted(d.rglob("*")):
        st = p.stat()
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
        out[str(p.relative_to(d))] = (st.st_size, st.st_mtime_ns, digest)
    return out


def _probe(d):
    from taosmd.store_probe import probe_store

    return probe_store(d)


def _vec(report):
    return next(x for x in report.databases if x.db == "vector_memory")


# ---------------------------------------------------------------- red-first

def _wal_db(path: Path, *, wal: bool):
    from taosmd.archive import INDEX_SCHEMA

    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(INDEX_SCHEMA)
    conn.commit()
    if wal:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


@pytest.mark.parametrize("wal", [True, False])
def test_status_does_not_create_wal_files(tmp_path, capsys, monkeypatch, wal):
    """`migrate --status` must never create -wal/-shm, even transiently while
    its connection is open, and must leave the file byte-identical. On master
    status_all went through _db.connect, which creates the sidecars for the
    life of the connection and rewrites a rollback-journal header to WAL."""
    path = tmp_path / "archive-index.db"
    _wal_db(path, wal=wal)
    before = _snapshot(tmp_path)

    seen = []
    real_status = migrations.status

    def spy(conn, db, **kw):
        seen.append(sorted(p.name for p in tmp_path.iterdir()))
        return real_status(conn, db, **kw)

    monkeypatch.setattr(migrations, "status", spy)
    rc = cli.main(["--data-dir", str(tmp_path), "migrate", "--status"])
    capsys.readouterr()
    assert rc == 0
    assert seen == [["archive-index.db"]], "sidecars existed while status read the db"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["archive-index.db"]
    assert _snapshot(tmp_path) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_status_works_on_a_read_only_data_dir(tmp_path, capsys):
    _wal_db(tmp_path / "archive-index.db", wal=True)
    os.chmod(tmp_path, 0o555)
    try:
        rc = cli.main(["--data-dir", str(tmp_path), "migrate", "--status"])
    finally:
        os.chmod(tmp_path, 0o755)
    capsys.readouterr()
    assert rc == 0


def test_status_positive_control_reports_current(tmp_path, capsys):
    _make_stamped(tmp_path / "archive-index.db", "archive_index")
    rc = cli.main(["--data-dir", str(tmp_path), "migrate", "--status", "--json"])
    rows = json.loads(capsys.readouterr().out)["databases"]
    assert rc == 0
    assert next(r for r in rows if r["db"] == "archive_index")["current"] is True


# ---------------------------------------------------------------- states

def test_empty_dir_is_current(tmp_path):
    r = _probe(tmp_path)
    assert r.state.value == "current" and r.exit_code == 0
    assert {d.state.value for d in r.databases} == {"absent"}


def test_missing_dir_is_unknown(tmp_path):
    r = _probe(tmp_path / "nope")
    assert r.state.value == "unknown" and r.exit_code == 11


def test_040_shaped_dir_is_old(tmp_path):
    _stamp_others(tmp_path)
    _make_vector_db(tmp_path / VEC, marker=LEGACY_MARKER, rows=["[0.1, 0.2]", "[0.3, 0.4]"])
    r = _probe(tmp_path)
    assert r.state.value == "old" and r.exit_code == 10
    v = _vec(r)
    assert v.state.value == "old" and v.detail["json_rows"] == 2


def test_legacy_markerless_json_store_is_old(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=None, rows=["[0.1, 0.2]"])
    assert _vec(_probe(tmp_path)).state.value == "old"


def test_legacy_marker_without_rows_is_old(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=LEGACY_MARKER, rows=[])
    assert _vec(_probe(tmp_path)).state.value == "old"


def test_converted_store_is_current(tmp_path):
    _stamp_others(tmp_path)
    _make_vector_db(tmp_path / VEC, marker=F32_MARKER, rows=[_f32([0.1, 0.2]), _f32([0.3, 0.4])])
    r = _probe(tmp_path)
    assert r.state.value == "current" and r.exit_code == 0


def test_fresh_empty_vector_store_is_current(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=None, rows=[])
    assert _vec(_probe(tmp_path)).state.value == "current"


def test_schema_version_below_latest_is_old(tmp_path):
    _make_stamped(tmp_path / "claims.db", "claims", version=0)
    r = _probe(tmp_path)
    claims = next(d for d in r.databases if d.db == "claims")
    assert claims.state.value == "old" and claims.user_version == 0
    assert claims.latest == migrations.latest_version("claims")
    assert r.exit_code == 10


def test_user_version_above_latest_is_newer(tmp_path):
    _make_stamped(tmp_path / "claims.db", "claims", version=migrations.latest_version("claims") + 1)
    r = _probe(tmp_path)
    assert r.state.value == "newer" and r.exit_code == 12
    # status() calls this current; the probe must not
    rows = {x["db"]: x for x in migrations.status_all(tmp_path) if x["exists"]}
    assert rows["claims"]["current"] is True


def test_unknown_marker_key_is_newer(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=F32_MARKER + ";quant4=1", rows=[_f32([0.1])])
    v = _vec(_probe(tmp_path))
    assert v.state.value == "newer" and "quant4" in v.reason


def test_unknown_dense_format_value_is_newer(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=F32_MARKER.replace("f32", "f16"), rows=[_f32([0.1])])
    assert _vec(_probe(tmp_path)).state.value == "newer"


def test_f32_marker_with_json_rows_is_unknown(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=F32_MARKER, rows=[_f32([0.1]), "[0.2]"])
    r = _probe(tmp_path)
    assert _vec(r).state.value == "unknown" and r.exit_code == 11


def test_legacy_marker_with_f32_rows_is_unknown(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=LEGACY_MARKER, rows=["[0.1]", _f32([0.2])])
    assert _vec(_probe(tmp_path)).state.value == "unknown"


def test_markerless_rows_that_are_not_json_are_unknown(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=None, rows=[_f32([0.2])])
    assert _vec(_probe(tmp_path)).state.value == "unknown"


def test_unparseable_marker_is_unknown(tmp_path):
    _make_vector_db(tmp_path / VEC, marker="garbage", rows=["[0.1]"])
    assert _vec(_probe(tmp_path)).state.value == "unknown"


def test_non_sqlite_file_is_unknown(tmp_path):
    (tmp_path / "claims.db").write_bytes(b"this is not a database" * 50)
    r = _probe(tmp_path)
    assert r.state.value == "unknown"


def test_worst_state_wins(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=LEGACY_MARKER, rows=["[0.1]"])  # old
    _make_stamped(tmp_path / "claims.db", "claims", version=99)  # newer
    (tmp_path / "collections.db").write_bytes(b"x" * 100)  # unknown
    assert _probe(tmp_path).state.value == "newer"


def test_marker_keys_match_what_vector_memory_writes():
    from taosmd.store_probe import KNOWN_MARKER_KEYS
    from taosmd.vector_memory import VectorMemory

    sig = VectorMemory(":memory:")._store_mode_signature()
    assert set(VectorMemory._parse_mode(sig)) == set(KNOWN_MARKER_KEYS)


def test_unregistered_db_is_listed_unversioned(tmp_path):
    _make_stamped(tmp_path / "crystals.db", "claims")
    r = _probe(tmp_path)
    c = next(d for d in r.databases if d.db == "crystals")
    assert c.state.value == "unversioned"
    assert r.state.value == "current"


# ---------------------------------------------------------------- WAL

def _live_wal(path: Path):
    """Leave a db with a non-empty -wal by keeping a writer connection open."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE IF NOT EXISTS w (x)")
    conn.execute("INSERT INTO w VALUES (1)")
    conn.commit()
    assert os.path.getsize(str(path) + "-wal") > 0
    return conn


def test_nonempty_wal_is_unknown_and_untouched(tmp_path):
    _stamp_others(tmp_path)
    _make_vector_db(tmp_path / VEC, marker=F32_MARKER, rows=[_f32([0.1])])
    writer = _live_wal(tmp_path / "claims.db")
    try:
        before = _snapshot(tmp_path)
        r = _probe(tmp_path)
        after = _snapshot(tmp_path)
    finally:
        writer.close()
    claims = next(d for d in r.databases if d.db == "claims")
    assert claims.state.value == "unknown"
    assert claims.reason == "live writer or uncheckpointed WAL"
    assert r.exit_code == 11
    assert before == after


# ---------------------------------------------------------------- read-only

def test_probe_changes_nothing_and_creates_no_sidecars(tmp_path):
    _stamp_others(tmp_path)
    _make_vector_db(tmp_path / VEC, marker=LEGACY_MARKER, rows=["[0.1, 0.2]"])
    # a WAL-mode db with its sidecars checkpointed away
    p = tmp_path / "claims.db"
    c = sqlite3.connect(p)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()
    for s in ("-wal", "-shm"):
        Path(str(p) + s).unlink(missing_ok=True)
    before = _snapshot(tmp_path)
    r = _probe(tmp_path)
    assert r.state.value == "old"
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob("*-wal")) and not list(tmp_path.glob("*-shm"))


def test_probe_creates_no_sidecars_even_while_the_connection_is_open(tmp_path, monkeypatch):
    """A plain mode=ro open of a WAL database creates -wal/-shm until it closes,
    and a clean close hides that from a before/after listing. Look mid-probe."""
    from taosmd import store_probe

    p = tmp_path / VEC
    _make_vector_db(p, marker=F32_MARKER, rows=[_f32([0.1])])
    c = sqlite3.connect(p)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()
    for sfx in ("-wal", "-shm"):
        Path(str(p) + sfx).unlink(missing_ok=True)

    seen = []
    real = store_probe._table_exists

    def spy(conn, name):
        seen.append(sorted(x.name for x in tmp_path.iterdir()))
        return real(conn, name)

    monkeypatch.setattr(store_probe, "_table_exists", spy)
    _probe(tmp_path)
    assert seen and all(names == [VEC] for names in seen)


def test_probe_does_not_convert_json_vectors(tmp_path):
    _make_vector_db(tmp_path / VEC, marker=LEGACY_MARKER, rows=["[0.1, 0.2]"])
    _probe(tmp_path)
    conn = sqlite3.connect(tmp_path / VEC)
    assert conn.execute("SELECT embedding FROM vector_memory").fetchone()[0] == "[0.1, 0.2]"
    assert "dense_format" not in conn.execute("SELECT value FROM store_meta").fetchone()[0]
    conn.close()


def test_path_with_special_characters(tmp_path):
    d = tmp_path / "my data #1 %20"
    d.mkdir()
    _make_vector_db(d / VEC, marker=LEGACY_MARKER, rows=["[0.1]"])
    assert _vec(_probe(d)).state.value == "old"


# ---------------------------------------------------------------- CLI

def test_cli_exit_codes_and_json(tmp_path, capsys):
    _make_vector_db(tmp_path / VEC, marker=LEGACY_MARKER, rows=["[0.1]"])
    rc = cli.main(["--data-dir", str(tmp_path), "migrate", "--probe", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 10 and payload["exit_code"] == 10 and payload["state"] == "old"
    vec = next(d for d in payload["databases"] if d["db"] == "vector_memory")
    assert set(vec) >= {"path", "state", "user_version", "latest", "reason"}
    assert vec["state"] == "old"


def test_cli_text_output(tmp_path, capsys):
    rc = cli.main(["--data-dir", str(tmp_path), "migrate", "--probe"])
    out = capsys.readouterr().out
    assert rc == 0 and "state: current (exit 0)" in out


def test_cli_probe_does_not_run_status_all(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("probe must not call status_all")

    monkeypatch.setattr(migrations, "status_all", boom)
    assert cli.main(["--data-dir", str(tmp_path), "migrate", "--probe"]) == 0
