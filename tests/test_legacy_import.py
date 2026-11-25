"""``taosmd migrate --import-legacy``: copy-then-verify of a legacy ./data tree.

Real SQLite files in tmp dirs. The source is never modified, moved, renamed,
or deleted: no write, no rename, no chmod.

Tests use ``_fresh()`` so they are idempotent across runs (pytest reuses the
per-test temporary directory between invocations).
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import sqlite3
from pathlib import Path
from unittest import mock

from taosmd import cli
import taosmd.legacy_import as legacy_import


VECTOR_DDL = "CREATE TABLE vector_memory(id INTEGER PRIMARY KEY, content TEXT, embedding BLOB)"
KG_DDL = {
    "entities": "CREATE TABLE entities(id INTEGER PRIMARY KEY, name TEXT)",
    "triples": "CREATE TABLE triples(id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT)",
}


def _fresh(path: Path) -> Path:
    """Return an empty directory at ``path``, removing any prior contents."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    return path


def _make_db(path: Path, tables: dict[str, str], rows: dict[str, list[tuple]]) -> None:
    conn = sqlite3.connect(str(path))
    try:
        for table, ddl in tables.items():
            conn.execute(ddl)
        for table, table_rows in rows.items():
            if not table_rows:
                continue
            placeholders = ", ".join("?" for _ in table_rows[0])
            conn.executemany(
                f"INSERT INTO {table} VALUES ({placeholders})", table_rows
            )
        conn.commit()
    finally:
        conn.close()


def _make_legacy_source(src: Path) -> None:
    """Build a 0.4-era ./data tree with real SQLite stores."""
    src.mkdir(exist_ok=True)
    _make_db(
        src / "vector-memory.db",
        {"vector_memory": VECTOR_DDL},
        {"vector_memory": [(1, "turn one", b"\x01\x02"), (2, "turn two", b"\x03")]},
    )
    _make_db(
        src / "knowledge-graph.db",
        KG_DDL,
        {
            "entities": [(1, "alice"), (2, "bob")],
            "triples": [(1, "alice", "knows", "bob")],
        },
    )
    (src / "agents.json").write_text('{"agents": []}')
    (src / "archive").mkdir()
    (src / "archive" / "turn-1.json").write_text('{"text": "hello"}')


def _snapshot(path: Path) -> dict:
    """Relative path -> (size, mtime_ns, is_dir) for the whole tree."""
    out: dict[str, tuple] = {}
    for entry in sorted(path.rglob("*")):
        st = entry.stat()
        out[str(entry.relative_to(path))] = (st.st_size, st.st_mtime_ns, entry.is_dir())
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path, table: str) -> int:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def _run_import(src: Path, dest: Path, apply: bool = False) -> int:
    argv = ["migrate", "--import-legacy", str(src), "--data-dir", str(dest)]
    if apply:
        argv.append("--apply")
    return cli.main(argv)


# ----------------------------------------------------------------------
# (a) dry run
# ----------------------------------------------------------------------


def test_dry_run_lists_files_and_changes_nothing(tmp_path, capsys):
    src = _fresh(tmp_path / "src")
    dest = _fresh(tmp_path / "dest")
    _make_legacy_source(src)
    before_src = _snapshot(src)
    before_dest = _snapshot(dest)

    rc = _run_import(src, dest, apply=False)

    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
    for store in ("vector-memory.db", "knowledge-graph.db", "archive", "agents.json"):
        assert store in out, f"missing {store} in dry-run listing: {out!r}"
    assert _snapshot(src) == before_src
    assert _snapshot(dest) == before_dest


# ----------------------------------------------------------------------
# (b) --apply copies and verifies
# ----------------------------------------------------------------------


def test_apply_copies_and_verifies(tmp_path):
    src = _fresh(tmp_path / "src")
    dest = _fresh(tmp_path / "dest")
    _make_legacy_source(src)
    before_src = _snapshot(src)

    rc = _run_import(src, dest, apply=True)

    assert rc == 0
    # databases: per-table row counts match
    assert _row_count(dest / "vector-memory.db", "vector_memory") == 2
    assert _row_count(dest / "knowledge-graph.db", "entities") == 2
    assert _row_count(dest / "knowledge-graph.db", "triples") == 1
    # plain files: sha256 matches
    assert _sha256(src / "agents.json") == _sha256(dest / "agents.json")
    for entry in (src / "archive").rglob("*"):
        rel = entry.relative_to(src / "archive")
        assert _sha256(entry) == _sha256(dest / "archive" / rel)
    # source untouched, mtimes included
    assert _snapshot(src) == before_src


# ----------------------------------------------------------------------
# (c) collision
# ----------------------------------------------------------------------


def test_collision_refuses_and_leaves_trees_unchanged(tmp_path):
    src = _fresh(tmp_path / "src")
    dest = _fresh(tmp_path / "dest")
    _make_legacy_source(src)
    (dest / "knowledge-graph.db").write_bytes(b"pre-existing")
    before_src = _snapshot(src)
    before_dest = _snapshot(dest)

    rc = _run_import(src, dest, apply=True)

    assert rc != 0
    assert _snapshot(src) == before_src
    assert _snapshot(dest) == before_dest
    assert (dest / "knowledge-graph.db").read_bytes() == b"pre-existing"


# ----------------------------------------------------------------------
# (d) verify failure
# ----------------------------------------------------------------------


def test_verify_failure_removes_temps_and_leaves_source(tmp_path):
    src = _fresh(tmp_path / "src")
    dest = _fresh(tmp_path / "dest")
    _make_legacy_source(src)
    before_src = _snapshot(src)

    with mock.patch.object(legacy_import, "_verify_database", return_value=False):
        rc = _run_import(src, dest, apply=True)

    assert rc != 0
    leftovers = [p for p in dest.iterdir()]
    assert not leftovers, f"temp or copy left in dest: {leftovers}"
    assert _snapshot(src) == before_src


# ----------------------------------------------------------------------
# (e) SRC == DEST
# ----------------------------------------------------------------------


def test_source_equals_dest_refused(tmp_path):
    src = _fresh(tmp_path / "data")
    _make_legacy_source(src)
    rc = cli.main(["migrate", "--import-legacy", str(src), "--data-dir", str(src)])
    assert rc != 0


# ----------------------------------------------------------------------
# (f) non-empty -wal
# ----------------------------------------------------------------------


def test_non_empty_wal_refused(tmp_path):
    src = _fresh(tmp_path / "src")
    dest = _fresh(tmp_path / "dest")
    _make_legacy_source(src)
    (src / "vector-memory.db-wal").write_bytes(b"\x01" * 100)
    before_src = _snapshot(src)

    rc = _run_import(src, dest, apply=True)

    assert rc != 0
    assert not list(dest.iterdir())
    assert _snapshot(src) == before_src


# ----------------------------------------------------------------------
# (g) the command the warning prints parses and runs (dry run)
# ----------------------------------------------------------------------


def test_warning_command_parses_and_runs_dry_run(tmp_path, capsys):
    """`taosmd migrate --import-legacy ./data` (from the gate D warning)
    parses and runs a dry run successfully. Replaces the message-only test."""
    src = _fresh(tmp_path / "data")
    _make_legacy_source(src)
    dest = _fresh(tmp_path / "dest")

    rc = cli.main(["migrate", "--import-legacy", str(src), "--data-dir", str(dest)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
    for store in ("vector-memory.db", "knowledge-graph.db", "archive", "agents.json"):
        assert store in out, f"missing {store} in dry-run output: {out!r}"


# ----------------------------------------------------------------------
# (5) after import the legacy warning no longer fires for that SRC
# ----------------------------------------------------------------------


def test_import_silences_legacy_warning(monkeypatch, tmp_path, caplog):
    """After a successful import the gate D warning no longer fires for that SRC."""
    import taosmd.agents as agents_mod

    cli._legacy_data_dir_warned = False
    agents_mod._legacy_registry_warned = False

    monkeypatch.chdir(tmp_path)
    legacy = _fresh(tmp_path / "data")
    (legacy / "vector-memory.db").write_text("")
    real = _fresh(tmp_path / "real-data")
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))

    # DEST lacks the store -> warning fires
    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        cli.main(["agent", "list"])
    assert any("import-legacy" in r.getMessage() for r in caplog.records)

    # complete the import into DEST
    cli._legacy_data_dir_warned = False
    rc = cli.main(["migrate", "--import-legacy", str(legacy), "--data-dir", str(real), "--apply"])
    assert rc == 0
    assert (real / "vector-memory.db").exists()

    # DEST now holds the store -> warning no longer fires
    cli._legacy_data_dir_warned = False
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        cli.main(["agent", "list"])
    assert not any("import-legacy" in r.getMessage() for r in caplog.records)
