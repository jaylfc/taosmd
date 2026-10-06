"""Backup, verify, and restore for the taOSmd data dir.

Tests are hermetic: tmp_path only, no network, no models.
"""

from __future__ import annotations

import sqlite3
import tarfile

import pytest

from taosmd._db import connect
from taosmd.backup import (
    backup_create,
    backup_restore,
    backup_verify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_dir(tmp_path, *, include_secrets=False):
    """Create a minimal taOSmd data dir with a SQLite store and a non-sqlite file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "knowledge-graph.db"
    conn = connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'alpha')")
    conn.execute("INSERT INTO t VALUES (2, 'beta')")
    conn.commit()
    conn.close()

    (data_dir / "archive").mkdir()
    (data_dir / "archive" / "turn.jsonl").write_text('{"role":"user","content":"hi"}\n')

    if include_secrets:
        import json
        cfg = {"server_token": "s", "admin_token": "a", "registry_token": "r"}
        (data_dir / "config.json").write_text(json.dumps(cfg))

    return data_dir


# ---------------------------------------------------------------------------
# (a) WAL consistency
# ---------------------------------------------------------------------------

def test_wal_consistency_create_restore_retains_uncommitted_wal(tmp_path, monkeypatch):
    """Rows written but still in WAL must survive create + restore.

    The keep-alive connection holds the WAL open. A naive shutil.copy2 of the
    main db file would NOT see those pages (they live in the -wal file and are
    invisible until checkpoint). sqlite3.Connection.backup() reads the
    consistent database snapshot including committed WAL pages, so the rows are
    preserved.

    This test PASSES with the real backup implementation and would FAIL if
    create() used plain file copy instead of sqlite3 backup.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "store.db"
    conn = connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'wal-row')")
    conn.commit()

    (data_dir / "archive").mkdir()
    (data_dir / "archive" / "turn.jsonl").write_text("x\n")

    monkeypatch.chdir(tmp_path)
    out = backup_create(data_dir, include_secrets=False)

    # Restore to a fresh dir.
    restore_dir = tmp_path / "restored"
    backup_restore(out, restore_dir)

    restored_db = restore_dir / "store.db"
    assert restored_db.exists()
    c2 = sqlite3.connect(str(restored_db))
    rows = c2.execute("SELECT v FROM t WHERE id=1").fetchall()
    c2.close()
    assert rows == [("wal-row",)], (
        "WAL-backed row must survive backup+restore"
    )


# ---------------------------------------------------------------------------
# (b) verify catches corruption
# ---------------------------------------------------------------------------

def test_verify_detects_flipped_byte_and_missing_member(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        tarball = backup_create(data_dir, include_secrets=False)
    finally:
        monkeypatch.undo()

    # Corrupt the tarball: flip one byte in the archive member.
    corrupt = tarball.with_name(tarball.name + ".corrupt")
    with tarball.open("rb") as src, corrupt.open("wb") as dst:
        blob = bytearray(src.read())
        # Flip a byte somewhere in the second half.
        blob[len(blob) // 2] ^= 0xFF
        dst.write(blob)

    rc = backup_verify(corrupt)
    assert rc == 1, "verify must exit non-zero on corruption"


def test_verify_detects_missing_member(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        tarball = backup_create(data_dir, include_secrets=False)
    finally:
        monkeypatch.undo()

    # Build a tarball that is missing one member present in the manifest.
    tmp = tmp_path / "stripped.tar.gz"
    with tarfile.open(tarball, "r:gz") as src, tarfile.open(tmp, "w:gz") as dst:
        for m in src.getmembers():
            if m.name == "archive/turn.jsonl":
                continue
            dst.addfile(m, src.extractfile(m))

    rc = backup_verify(tmp)
    assert rc == 1, "verify must exit non-zero when a manifest member is missing"


# ---------------------------------------------------------------------------
# (c) restore refuses non-empty target, --move-existing preserves
# ---------------------------------------------------------------------------

def test_restore_refuses_non_empty_target(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        tarball = backup_create(data_dir, include_secrets=False)
    finally:
        monkeypatch.undo()

    target = tmp_path / "existing"
    target.mkdir()
    (target / "junk.txt").write_text("junk")

    with pytest.raises(RuntimeError):
        backup_restore(tarball, target)

    assert (target / "junk.txt").exists(), "existing content must not be touched"


def test_restore_move_existing_preserves_old_dir(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        tarball = backup_create(data_dir, include_secrets=False)
    finally:
        monkeypatch.undo()

    target = tmp_path / "existing"
    target.mkdir()
    junk = target / "junk.txt"
    junk.write_text("junk")

    backup_restore(tarball, target, move_existing=True)

    assert not junk.exists(), "old content must be moved away"
    assert (target / "archive" / "turn.jsonl").exists(), "restore must populate target"
    moved = [p.name for p in tmp_path.iterdir() if p.name.startswith("existing.pre-restore-")]
    assert moved, "old dir must be renamed with timestamp"


# ---------------------------------------------------------------------------
# (d) path traversal rejected
# ---------------------------------------------------------------------------

def test_restore_rejects_path_traversal_member(tmp_path):
    """A tarball member with ../evil must be rejected."""
    data_dir = _make_data_dir(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        tarball = backup_create(data_dir, include_secrets=False)
    finally:
        monkeypatch.undo()

    # Craft a tarball with a ../evil member.
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(tarball, "r:gz") as src, tarfile.open(evil, "w:gz") as dst:
        for m in src.getmembers():
            dst.addfile(m, src.extractfile(m))
        info = tarfile.TarInfo(name="../evil")
        info.size = 0
        dst.addfile(info)

    target = tmp_path / "restore"
    with pytest.raises((RuntimeError, ValueError, OSError)):
        backup_restore(evil, target)


# ---------------------------------------------------------------------------
# (e) config.json excluded by default, included with --include-secrets
# ---------------------------------------------------------------------------

def test_config_json_excluded_by_default(tmp_path):
    data_dir = _make_data_dir(tmp_path, include_secrets=True)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        tarball = backup_create(data_dir, include_secrets=False)
    finally:
        monkeypatch.undo()

    with tarfile.open(tarball, "r:gz") as tf:
        names = tf.getnames()
    assert "config.json" not in names, "config.json must be excluded by default"


def test_config_json_included_with_flag(tmp_path):
    data_dir = _make_data_dir(tmp_path, include_secrets=True)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        tarball = backup_create(data_dir, include_secrets=True)
    finally:
        monkeypatch.undo()

    with tarfile.open(tarball, "r:gz") as tf:
        names = tf.getnames()
    assert "config.json" in names, "config.json must be included with --include-secrets"


# ---------------------------------------------------------------------------
# Sanity: end-to-end create + verify + restore
# ---------------------------------------------------------------------------

def test_create_verify_restore_roundtrip(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        tarball = backup_create(data_dir, include_secrets=False)
    finally:
        monkeypatch.undo()

    assert tarball.exists()
    assert tarball.name.startswith("taosmd-backup-")
    assert tarball.suffixes == [".tar", ".gz"]

    rc = backup_verify(tarball)
    assert rc == 0, "freshly created backup must verify cleanly"

    restore_dir = tmp_path / "restored"
    backup_restore(tarball, restore_dir)

    assert (restore_dir / "archive" / "turn.jsonl").read_text() == '{"role":"user","content":"hi"}\n'
    assert (restore_dir / "knowledge-graph.db").exists()
