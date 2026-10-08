"""Tests for ``taosmd.backup``: create, verify, restore.

Every test is hermetic (tmp_path, no network, no models). Where a test is
labelled mutation-killing it MUST fail if the named guard is removed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile

import pytest

from taosmd import _db, backup as bp
from taosmd.backup import _unsafe_member, create, restore, verify
from taosmd.cli import main


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_data_dir(tmp_path, *, include_config=False, extra_files=None):
    d = tmp_path / "data"
    d.mkdir()
    db = d / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (a TEXT)")
    conn.execute("INSERT INTO t VALUES ('hello')")
    conn.commit()
    conn.close()
    if include_config:
        (d / "config.json").write_text(json.dumps({"server_token": "s", "admin_token": "a", "registry_token": "r"}))
    if extra_files:
        for name, content in extra_files.items():
            (d / name).write_text(content)
    return d


def _open_backup(path):
    return tarfile.open(str(path), "r:gz")


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

def test_create_produces_tarball(tmp_path, capsys):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    assert out.exists()
    assert tarfile.is_tarfile(str(out))


def test_create_excludes_config_by_default(tmp_path, capsys):
    data = _make_data_dir(tmp_path, include_config=True)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        names = [m.name for m in tf.getmembers()]
    assert "config.json" not in names


def test_create_includes_config_with_secrets_warning(tmp_path, capsys):
    data = _make_data_dir(tmp_path, include_config=True)
    out = create(data, out=tmp_path / "b.tar.gz", include_secrets=True)
    with _open_backup(out) as tf:
        names = [m.name for m in tf.getmembers()]
    assert "config.json" in names
    captured = capsys.readouterr().err
    assert "unencrypted" in captured


def test_create_wal_sidecars_excluded(tmp_path):
    data = _make_data_dir(tmp_path)
    (data / "test.db-wal").write_text("wal")
    (data / "test.db-shm").write_text("shm")
    (data / "test.db-journal").write_text("journal")
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        names = [m.name for m in tf.getmembers()]
    assert "test.db-wal" not in names
    assert "test.db-shm" not in names
    assert "test.db-journal" not in names


def test_create_out_inside_data_dir_refused(tmp_path):
    data = _make_data_dir(tmp_path)
    with pytest.raises(ValueError, match="must not lie inside"):
        create(data, out=data / "backup.tar.gz")


def test_create_refuses_existing_out(tmp_path):
    data = _make_data_dir(tmp_path)
    out = tmp_path / "exists.tar.gz"
    out.write_text("nope")
    with pytest.raises(FileExistsError):
        create(data, out=out)
    assert out.read_text() == "nope"


def test_create_single_pass_hashes_staged_copy(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    original_copy2 = shutil.copy2

    def patched_copy2(src, dst, *args, **kwargs):
        original_copy2(src, dst, *args, **kwargs)
        with open(src, "a") as f:
            f.write("APPENDED")

    monkeypatch.setattr(shutil, "copy2", patched_copy2)
    out = create(data, out=tmp_path / "b.tar.gz")
    verify(out)


def test_create_appended_file_still_verifies(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    extra = tmp_path / "extra.txt"
    extra.write_text("line1")

    def patched_copy2(src, dst, *args, **kwargs):
        original_copy2(src, dst, *args, **kwargs)
        if Path(src) == extra:
            extra.write_text("line1\nline2")

    from pathlib import Path
    original_copy2 = shutil.copy2
    monkeypatch.setattr(shutil, "copy2", patched_copy2)
    out = create(data, out=tmp_path / "b.tar.gz")
    verify(out)


def test_create_failure_leaves_no_staging_or_partial_out(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)

    def bad_copy(src, dst):
        raise RuntimeError("copy boom")

    monkeypatch.setattr(bp, "_copy_sqlite", bad_copy)
    with pytest.raises(RuntimeError, match="copy boom"):
        create(data, out=tmp_path / "b.tar.gz")
    assert not (tmp_path / "b.tar.gz").exists()
    assert not list(tmp_path.glob("taosmd-backup-*"))


# ---------------------------------------------------------------------------
# WAL consistency: create/restore must capture committed WAL pages
# ---------------------------------------------------------------------------

def test_create_restore_wal_consistency(tmp_path):
    src = _make_data_dir(tmp_path)
    conn = _db.connect(src / "test.db")
    conn.execute("INSERT INTO t VALUES ('from-wal')")
    conn.commit()
    out = create(src, out=tmp_path / "b.tar.gz")
    dest = tmp_path / "restored"
    restore(out, dest=dest)
    conn2 = _db.connect(dest / "test.db")
    row = conn2.execute("SELECT a FROM t WHERE a='from-wal'").fetchone()
    assert row is not None
    assert row[0] == "from-wal"
    conn2.close()


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def test_verify_ok(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    verify(out)


def test_verify_sha256_mismatch(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        members = tf.getmembers()
        manifest = json.loads(tf.extractfile("MANIFEST.json").read().decode("utf-8"))
        member_data = {}
        for m in members:
            if m.name == "MANIFEST.json":
                continue
            fh = tf.extractfile(m)
            member_data[m.name] = fh.read() if fh else b""
    # Rewrite test.db into a new valid gzip tarball, keep old manifest
    new_tf_path = tmp_path / "bad.tar.gz"
    with tarfile.open(str(new_tf_path), "w:gz") as ntf:
        manifest_b = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_b)
        ntf.addfile(info, io.BytesIO(manifest_b))
        for name, data_bytes in member_data.items():
            if name == "MANIFEST.json":
                continue
            if name == "test.db":
                data_bytes = data_bytes[:4] + b"\x00" + data_bytes[5:]
            info = tarfile.TarInfo(name=name)
            info.size = len(data_bytes)
            ntf.addfile(info, io.BytesIO(data_bytes))
    with pytest.raises(ValueError, match="test.db.*sha256 mismatch"):
        verify(new_tf_path)


def test_verify_integrity_check_corrupt_db(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        manifest = json.loads(tf.extractfile("MANIFEST.json").read().decode("utf-8"))
        members = tf.getmembers()
        member_data = {}
        for m in members:
            if m.name == "MANIFEST.json":
                continue
            fh = tf.extractfile(m)
            member_data[m.name] = fh.read() if fh else b""
    corrupt_bytes = member_data.get("test.db", b"")
    if corrupt_bytes and len(corrupt_bytes) > 200:
        corrupt_bytes = corrupt_bytes[:100] + b"\x00" * 100 + corrupt_bytes[200:]
        corrupt_sha = hashlib.sha256(corrupt_bytes).hexdigest()
        manifest["files"][0]["sha256"] = corrupt_sha
    new_tf_path = tmp_path / "corrupt.tar.gz"
    with tarfile.open(str(new_tf_path), "w:gz") as ntf:
        manifest_b = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_b)
        ntf.addfile(info, io.BytesIO(manifest_b))
        for name, data_bytes in member_data.items():
            if name == "MANIFEST.json":
                continue
            if name == "test.db":
                data_bytes = corrupt_bytes
            info = tarfile.TarInfo(name=name)
            info.size = len(data_bytes)
            ntf.addfile(info, io.BytesIO(data_bytes))
    with pytest.raises(ValueError, match="test.db"):
        verify(new_tf_path)


def test_verify_missing_member(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        manifest = json.loads(tf.extractfile("MANIFEST.json").read().decode("utf-8"))
        members = tf.getmembers()
        member_data = {}
        for m in members:
            if m.name == "MANIFEST.json":
                continue
            fh = tf.extractfile(m)
            member_data[m.name] = fh.read() if fh else b""
    manifest["files"].append({"path": "ghost.txt", "size": 0, "sha256": "abcd" * 8})
    new_tf_path = tmp_path / "missing.tar.gz"
    with tarfile.open(str(new_tf_path), "w:gz") as ntf:
        manifest_b = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_b)
        ntf.addfile(info, io.BytesIO(manifest_b))
        for name, data_bytes in member_data.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data_bytes)
            ntf.addfile(info, io.BytesIO(data_bytes))
    with pytest.raises(ValueError, match="ghost.txt.*in manifest but missing"):
        verify(new_tf_path)


def test_verify_manifest_entry_without_path(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        manifest = json.loads(tf.extractfile("MANIFEST.json").read().decode("utf-8"))
        members = tf.getmembers()
        member_data = {}
        for m in members:
            if m.name == "MANIFEST.json":
                continue
            fh = tf.extractfile(m)
            member_data[m.name] = fh.read() if fh else b""
    manifest["files"].append({})
    new_tf_path = tmp_path / "nopath.tar.gz"
    with tarfile.open(str(new_tf_path), "w:gz") as ntf:
        manifest_b = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_b)
        ntf.addfile(info, io.BytesIO(manifest_b))
        for name, data_bytes in member_data.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data_bytes)
            ntf.addfile(info, io.BytesIO(data_bytes))
    with pytest.raises(ValueError, match="no path key"):
        verify(new_tf_path)


def test_verify_rejects_unsafe_manifest_path(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        manifest = json.loads(tf.extractfile("MANIFEST.json").read().decode("utf-8"))
        members = tf.getmembers()
        member_data = {}
        for m in members:
            if m.name == "MANIFEST.json":
                continue
            fh = tf.extractfile(m)
            member_data[m.name] = fh.read() if fh else b""
    manifest["files"].append({"path": "../evil", "size": 0, "sha256": "abcd" * 8})
    new_tf_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(str(new_tf_path), "w:gz") as ntf:
        manifest_b = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_b)
        ntf.addfile(info, io.BytesIO(manifest_b))
        for name, data_bytes in member_data.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data_bytes)
            ntf.addfile(info, io.BytesIO(data_bytes))
    with pytest.raises(ValueError, match="unsafe manifest path"):
        verify(new_tf_path)


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

def test_restore_basic(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    dest = tmp_path / "restored"
    restore(out, dest=dest)
    assert (dest / "test.db").exists()
    conn = sqlite3.connect(str(dest / "test.db"))
    rows = conn.execute("SELECT a FROM t").fetchall()
    assert [r[0] for r in rows] == ["hello"]
    conn.close()


def test_restore_refuses_non_empty_target(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "existing.txt").write_text("x")
    with pytest.raises(ValueError, match="non-empty"):
        restore(out, dest=dest)
    assert (dest / "existing.txt").exists()


def test_restore_move_existing_preserves_old_dir(tmp_path, capsys):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    restore(out, dest=dest, move_existing=True)
    assert (dest / "test.db").exists()
    old_dirs = [p for p in tmp_path.glob("dest.pre-restore-*")]
    assert len(old_dirs) == 1
    assert (old_dirs[0] / "old.txt").exists()


def test_restore_rejects_evil_member(tmp_path):
    out = tmp_path / "evil.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"{}"))
        info2 = tarfile.TarInfo(name="../evil")
        info2.size = 0
        tf.addfile(info2, io.BytesIO(b""))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="unsafe tar member"):
        restore(out, dest=dest)
    assert (dest / "keep.txt").exists()
    written = list(tmp_path.rglob("*"))
    written_names = [p.name for p in written]
    assert "evil" not in written_names


def test_restore_verify_monkeypatched_evil_member_leaves_target_untouched(tmp_path, monkeypatch):
    out = tmp_path / "evil.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"{}"))
        info2 = tarfile.TarInfo(name="../evil")
        info2.size = 0
        tf.addfile(info2, io.BytesIO(b""))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    monkeypatch.setattr("taosmd.backup.verify", lambda path: None)
    with pytest.raises(ValueError, match="unsafe tar member"):
        restore(out, dest=dest)
    assert (dest / "keep.txt").exists()
    written = list(tmp_path.rglob("*"))
    written_names = [p.name for p in written]
    assert "evil" not in written_names


def test_restore_atomicity_failure_leaves_original_intact_no_staging(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    original_extract = bp._extract_members

    def bad_extract(tf, dest_dir):
        original_extract(tf, dest_dir)
        raise RuntimeError("boom")

    monkeypatch.setattr(bp, "_extract_members", bad_extract)
    with pytest.raises(RuntimeError, match="boom"):
        restore(out, dest=dest, move_existing=True)
    assert (dest / "keep.txt").exists()
    assert not (tmp_path / "dest.staging").exists()


def test_restore_filter_data_rejects_symlink(tmp_path):
    out = tmp_path / "symlink.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="unsafe link member"):
        restore(out, dest=dest)


def test_restore_final_rename_failure_moves_original_back(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    original_rename = os.rename

    def bad_rename(src, dst):
        if str(dst) == str(dest) and str(src).endswith(".staging"):
            raise OSError("rename failed")
        original_rename(src, dst)

    monkeypatch.setattr(os, "rename", bad_rename)
    with pytest.raises(OSError, match="rename failed"):
        restore(out, dest=dest, move_existing=True)
    assert (dest / "keep.txt").exists()


# ---------------------------------------------------------------------------
# mutation-killing tests
# ---------------------------------------------------------------------------

def test_mutant_verify_sha256_compare_catches_rewrite(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        members = tf.getmembers()
        manifest = json.loads(tf.extractfile("MANIFEST.json").read().decode("utf-8"))
        member_data = {}
        for m in members:
            if m.name == "MANIFEST.json":
                continue
            fh = tf.extractfile(m)
            member_data[m.name] = fh.read() if fh else b""
    new_tf = tmp_path / "rewritten.tar.gz"
    with tarfile.open(str(new_tf), "w:gz") as ntf:
        manifest_b = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_b)
        ntf.addfile(info, io.BytesIO(manifest_b))
        for name, data_bytes in member_data.items():
            if name == "test.db":
                data_bytes = data_bytes[:4] + b"\x00" + data_bytes[5:]
            info = tarfile.TarInfo(name=name)
            info.size = len(data_bytes)
            ntf.addfile(info, io.BytesIO(data_bytes))
    with pytest.raises(ValueError, match="test.db"):
        verify(new_tf)


def test_mutant_verify_integrity_check_catches_corrupt_db(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        members = tf.getmembers()
        manifest = json.loads(tf.extractfile("MANIFEST.json").read().decode("utf-8"))
        member_data = {}
        for m in members:
            if m.name == "MANIFEST.json":
                continue
            fh = tf.extractfile(m)
            member_data[m.name] = fh.read() if fh else b""
    corrupt_bytes = member_data.get("test.db", b"")
    if corrupt_bytes and len(corrupt_bytes) > 200:
        corrupt_bytes = corrupt_bytes[:100] + b"\x00" * 100 + corrupt_bytes[200:]
        corrupt_sha = hashlib.sha256(corrupt_bytes).hexdigest()
        manifest["files"][0]["sha256"] = corrupt_sha
    new_tf = tmp_path / "corrupt.tar.gz"
    with tarfile.open(str(new_tf), "w:gz") as ntf:
        manifest_b = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_b)
        ntf.addfile(info, io.BytesIO(manifest_b))
        for name, data_bytes in member_data.items():
            if name == "MANIFEST.json":
                continue
            if name == "test.db":
                data_bytes = corrupt_bytes
            info = tarfile.TarInfo(name=name)
            info.size = len(data_bytes)
            ntf.addfile(info, io.BytesIO(data_bytes))
    with pytest.raises(ValueError, match="test.db"):
        verify(new_tf)


def test_mutant_restore_verify_failure_leaves_dest_untouched(tmp_path):
    out = tmp_path / "bad.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"{}"))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    with pytest.raises(ValueError):
        restore(out, dest=dest)
    assert (dest / "keep.txt").exists()
    assert not (dest / "test.db").exists()


def test_mutant_unsafe_member_check_directly():
    assert _unsafe_member("../x")
    assert _unsafe_member("/abs")
    assert _unsafe_member("foo/bar/../baz")
    assert not _unsafe_member("safe/path")


# ---------------------------------------------------------------------------
# U1: extraction traversal guard unpinned
# ---------------------------------------------------------------------------

def test_restore_rejects_evil_member_even_when_verify_patched(tmp_path, monkeypatch):
    out = tmp_path / "evil.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"{}"))
        info2 = tarfile.TarInfo(name="../evil")
        info2.size = 0
        tf.addfile(info2, io.BytesIO(b""))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    monkeypatch.setattr("taosmd.backup.verify", lambda path: None)
    with pytest.raises(ValueError, match="unsafe tar member"):
        restore(out, dest=dest)
    assert (dest / "keep.txt").exists()
    written = list(tmp_path.rglob("*"))
    written_names = [p.name for p in written]
    assert "evil" not in written_names


# ---------------------------------------------------------------------------
# U2: atomicity
# ---------------------------------------------------------------------------

def test_restore_extraction_failure_leaves_no_staging_and_original_intact(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    original = bp._extract_members

    def bad_extract(tf, dest_dir):
        original(tf, dest_dir)
        raise RuntimeError("extract boom")

    monkeypatch.setattr(bp, "_extract_members", bad_extract)
    with pytest.raises(RuntimeError, match="extract boom"):
        restore(out, dest=dest, move_existing=True)
    assert (dest / "keep.txt").exists()
    assert not (tmp_path / "dest.staging").exists()


def test_restore_rejects_symlink_member(tmp_path):
    out = tmp_path / "sym.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="unsafe link member"):
        restore(out, dest=dest)


# ---------------------------------------------------------------------------
# U3: single-pass
# ---------------------------------------------------------------------------

def test_create_hashes_staged_copy_not_live_source(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    original_copy2 = shutil.copy2

    def patched_copy2(src, dst, *args, **kwargs):
        original_copy2(src, dst, *args, **kwargs)
        with open(src, "ab") as f:
            f.write(b"MODIFIED")

    monkeypatch.setattr(shutil, "copy2", patched_copy2)
    out = create(data, out=tmp_path / "b.tar.gz")
    verify(out)


def test_create_wal_sidecar_excluded(tmp_path):
    data = _make_data_dir(tmp_path)
    (data / "test.db-wal").write_text("wal")
    (data / "test.db-shm").write_text("shm")
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        names = [m.name for m in tf.getmembers()]
    assert "test.db-wal" not in names
    assert "test.db-shm" not in names


def test_create_out_inside_data_dir_excluded(tmp_path):
    data = _make_data_dir(tmp_path)
    with pytest.raises(ValueError, match="must not lie inside"):
        create(data, out=data / "backup.tar.gz")


# ---------------------------------------------------------------------------
# U4: CLI error handling
# ---------------------------------------------------------------------------

def test_cli_backup_create_rc1_on_value_error(tmp_path, capsys):
    data = _make_data_dir(tmp_path)
    rc = main(["--data-dir", str(tmp_path), "backup", "create", "--out", str(data / "inside.tar.gz")])
    assert rc == 1
    captured = capsys.readouterr().err
    assert "error:" in captured


def test_cli_backup_verify_missing_manifest_rc1(tmp_path, capsys):
    out = tmp_path / "bad.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="not_manifest")
        info.size = 0
        tf.addfile(info, io.BytesIO(b""))
    rc = main(["--data-dir", str(tmp_path), "backup", "verify", str(out)])
    assert rc == 1


def test_cli_backup_restore_rc1_on_value_error(tmp_path, capsys):
    out = tmp_path / "bad.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"{}"))
        info2 = tarfile.TarInfo(name="extra.txt")
        info2.size = 4
        tf.addfile(info2, io.BytesIO(b"data"))
    rc = main(["--data-dir", str(tmp_path), "backup", "restore", str(out), "--to", str(tmp_path / "dest")])
    assert rc == 1


def test_cli_backup_manifest_entry_without_path_is_verify_failure(tmp_path):
    data = _make_data_dir(tmp_path)
    out = create(data, out=tmp_path / "b.tar.gz")
    with _open_backup(out) as tf:
        manifest = json.loads(tf.extractfile("MANIFEST.json").read().decode("utf-8"))
        members = tf.getmembers()
        member_data = {}
        for m in members:
            if m.name == "MANIFEST.json":
                continue
            fh = tf.extractfile(m)
            member_data[m.name] = fh.read() if fh else b""
    manifest["files"].append({})
    new_tf = tmp_path / "nopath.tar.gz"
    with tarfile.open(str(new_tf), "w:gz") as ntf:
        manifest_b = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = len(manifest_b)
        ntf.addfile(info, io.BytesIO(manifest_b))
        for name, data_bytes in member_data.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data_bytes)
            ntf.addfile(info, io.BytesIO(data_bytes))
    with pytest.raises(ValueError, match="no path key"):
        verify(new_tf)


def test_cli_backup_catches_keyerror_as_error(tmp_path, capsys):
    out = tmp_path / "bad.tar.gz"
    with tarfile.open(str(out), "w:gz") as tf:
        info = tarfile.TarInfo(name="MANIFEST.json")
        info.size = 0
        tf.addfile(info, io.BytesIO(b""))
    rc = main(["--data-dir", str(tmp_path), "backup", "verify", str(out)])
    assert rc == 1
