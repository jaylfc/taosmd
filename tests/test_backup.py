"""Tests for taosmd.backup.

Mutation-killing: each test must FAIL if the named safety check is removed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile

import pytest

from pathlib import Path

from taosmd import _db
from taosmd import backup as backup_mod


def _make_data_dir(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    (data / "archive").mkdir()
    (data / "archive" / "2026-01-01.jsonl").write_text("hello\n")
    db = data / "knowledge-graph.db"
    conn = _db.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t (v) VALUES ('wal-row')")
    conn.commit()
    conn.execute("INSERT INTO t (v) VALUES ('wal-row-2')")
    conn.commit()
    conn.close()
    return data, db


# ---------------------------------------------------------------------------
# WAL consistency
# ---------------------------------------------------------------------------

def test_wal_rows_survive_create_and_restore(tmp_path: Path):
    data, db = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"

    conn = _db.connect(str(db))
    try:
        conn.execute("INSERT INTO t (v) VALUES ('mid-create')")
        conn.commit()
        backup_mod.create(str(data), str(out), include_secrets=False)
    finally:
        conn.close()

    restore_dir = tmp_path / "restored"
    backup_mod.restore(str(out), str(restore_dir))

    restored_db = restore_dir / "knowledge-graph.db"
    rconn = _db.connect(str(restored_db))
    try:
        rows = [r[0] for r in rconn.execute("SELECT v FROM t ORDER BY id").fetchall()]
    finally:
        rconn.close()

    assert rows == ["wal-row", "wal-row-2", "mid-create"]


# ---------------------------------------------------------------------------
# verify catches flipped byte and missing member
# ---------------------------------------------------------------------------

def test_verify_catches_flipped_byte(tmp_path: Path):
    data, _ = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    # Flip a byte in the archive member, keep MANIFEST.json intact
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path_str = tmp.name
    try:
        with tarfile.open(str(out), "r:gz") as src, \
             tarfile.open(tmp_path_str, "w:gz") as dst:
            for member in src.getmembers():
                fh = src.extractfile(member)
                if fh is None:
                    continue
                data_bytes = fh.read()
                if member.name == "MANIFEST.json":
                    dst.addfile(member, io.BytesIO(data_bytes))
                else:
                    flipped = bytearray(data_bytes)
                    if flipped:
                        flipped[0] ^= 0xFF
                    dst.addfile(member, io.BytesIO(bytes(flipped)))
        assert not backup_mod.verify(tmp_path_str), \
            "verify should exit non-zero on a flipped byte"
    finally:
        os.unlink(tmp_path_str)


def test_verify_catches_missing_member(tmp_path: Path):
    data, _ = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path_str = tmp.name
    try:
        with tarfile.open(str(out), "r:gz") as src, \
             tarfile.open(tmp_path_str, "w:gz") as dst:
            manifest_added = False
            for member in src.getmembers():
                fh = src.extractfile(member)
                if fh is None:
                    continue
                data_bytes = fh.read()
                if member.name == "MANIFEST.json":
                    dst.addfile(member, io.BytesIO(data_bytes))
                    manifest_added = True
                else:
                    pass  # skip one non-manifest member
            assert manifest_added
        assert not backup_mod.verify(tmp_path_str), \
            "verify should exit non-zero on a missing member"
    finally:
        os.unlink(tmp_path_str)


# ---------------------------------------------------------------------------
# restore refuses non-empty target and --move-existing preserves old dir
# ---------------------------------------------------------------------------

def test_restore_refuses_non_empty_target(tmp_path: Path):
    data, _ = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    target = tmp_path / "target"
    target.mkdir()
    (target / "junk").write_text("i-am-junk")

    with pytest.raises(RuntimeError, match="non-empty"):
        backup_mod.restore(str(out), str(target))


def test_restore_move_existing_preserves_old_dir(tmp_path: Path):
    data, _ = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    target = tmp_path / "target"
    target.mkdir()
    (target / "junk").write_text("i-am-junk")

    backup_mod.restore(str(out), str(target), move_existing=True)

    assert target.exists()
    assert (target / "archive").exists()
    assert not (target / "junk").exists(), "old content must not be in target"
    backups = list(tmp_path.glob("target.pre-restore-*"))
    assert len(backups) == 1
    assert (backups[0] / "junk").exists(), "old dir must be preserved as backup"


# ---------------------------------------------------------------------------
# crafted tarball with ../evil is rejected
# ---------------------------------------------------------------------------

def test_restore_rejects_path_traversal(tmp_path: Path):
    data, _ = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path_str = tmp.name
    try:
        with tarfile.open(str(out), "r:gz") as src, \
             tarfile.open(tmp_path_str, "w:gz") as dst:
            for member in src.getmembers():
                fh = src.extractfile(member)
                if fh is None:
                    continue
                data_bytes = fh.read()
                if member.name == "MANIFEST.json":
                    dst.addfile(member, io.BytesIO(data_bytes))
                else:
                    evil_name = "../evil"
                    ti = tarfile.TarInfo(name=evil_name)
                    ti.size = len(data_bytes)
                    dst.addfile(ti, io.BytesIO(data_bytes))

        with pytest.raises(RuntimeError, match="verification failed"):
            backup_mod.restore(tmp_path_str, str(tmp_path / "evil-target"))
    finally:
        os.unlink(tmp_path_str)


# ---------------------------------------------------------------------------
# config.json excluded by default, included with --include-secrets
# ---------------------------------------------------------------------------

def test_config_json_excluded_by_default(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.json").write_text(json.dumps({"server_token": "secret"}))
    (data / "other.txt").write_text("hi")

    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    with tarfile.open(str(out), "r:gz") as tar:
        names = tar.getnames()
    assert "config.json" not in names
    assert "other.txt" in names


def test_config_json_included_with_include_secrets(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.json").write_text(json.dumps({"server_token": "secret"}))
    (data / "other.txt").write_text("hi")

    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=True)

    with tarfile.open(str(out), "r:gz") as tar:
        names = tar.getnames()
    assert "config.json" in names
    assert "other.txt" in names


# ---------------------------------------------------------------------------
# mutation-killing verify checks
# ---------------------------------------------------------------------------

def test_verify_sha256_mutation_kill(tmp_path: Path):
    data, _ = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    import json
    with tarfile.open(str(out), "r:gz") as src:
        manifest_fh = src.extractfile("MANIFEST.json")
        manifest = json.loads(manifest_fh.read().decode("utf-8"))
        for f in manifest["files"]:
            if f["path"] == "archive/2026-01-01.jsonl":
                f["sha256"] = "deadbeef" * 8
                break

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as new_tar:
            for member in src.getmembers():
                if member.name == "MANIFEST.json":
                    continue
                fh = src.extractfile(member)
                if fh is None:
                    continue
                new_tar.addfile(member, io.BytesIO(fh.read()))
            new_manifest_bytes = (
                json.dumps(manifest, indent=2).encode("utf-8") + b"\n"
            )
            ti = tarfile.TarInfo(name="MANIFEST.json")
            ti.size = len(new_manifest_bytes)
            new_tar.addfile(ti, io.BytesIO(new_manifest_bytes))

    corrupt = tmp_path / "corrupt.tar.gz"
    corrupt.write_bytes(buf.getvalue())
    assert not backup_mod.verify(str(corrupt)), \
        "verify must catch sha256 mutation"


def test_verify_integrity_check_mutation_kill(tmp_path: Path):
    data, _ = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    import json
    with tarfile.open(str(out), "r:gz") as src:
        manifest_fh = src.extractfile("MANIFEST.json")
        manifest = json.loads(manifest_fh.read().decode("utf-8"))

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as new_tar:
            for member in src.getmembers():
                if member.name == "MANIFEST.json":
                    continue
                fh = src.extractfile(member)
                if fh is None:
                    continue
                if member.name == "knowledge-graph.db":
                    corrupt = b"not sqlite"
                    ti = tarfile.TarInfo(name=member.name)
                    ti.size = len(corrupt)
                    new_tar.addfile(ti, io.BytesIO(corrupt))
                    for f in manifest["files"]:
                        if f["path"] == "knowledge-graph.db":
                            f["size"] = len(corrupt)
                            f["sha256"] = hashlib.sha256(corrupt).hexdigest()
                            f["sqlite_integrity"] = "ok"
                            break
                else:
                    new_tar.addfile(member, io.BytesIO(fh.read()))

            new_manifest_bytes = (
                json.dumps(manifest, indent=2).encode("utf-8") + b"\n"
            )
            ti = tarfile.TarInfo(name="MANIFEST.json")
            ti.size = len(new_manifest_bytes)
            new_tar.addfile(ti, io.BytesIO(new_manifest_bytes))

    corrupt = tmp_path / "corrupt.tar.gz"
    corrupt.write_bytes(buf.getvalue())
    assert not backup_mod.verify(str(corrupt)), \
        "verify must catch integrity_check corruption"


def test_verify_restore_leaves_dir_untouched_on_failure(tmp_path: Path):
    data, _ = _make_data_dir(tmp_path)
    out = tmp_path / "backup.tar.gz"
    backup_mod.create(str(data), str(out), include_secrets=False)

    # Corrupt the tarball by flipping a byte
    corrupt = tmp_path / "corrupt.tar.gz"
    shutil.copy2(out, corrupt)
    with open(corrupt, "r+b") as fh:
        fh.seek(20)
        fh.write(b"\xFF")

    target = tmp_path / "target"
    target.mkdir()
    (target / "junk").write_text("safe")

    with pytest.raises(RuntimeError, match="verification failed"):
        backup_mod.restore(str(corrupt), str(target))

    assert (target / "junk").exists(), "target dir must be untouched on verify failure"


def test_unsafe_member_check_direct():
    with pytest.raises(ValueError, match="path traversal"):
        backup_mod._check_unsafe_member(
            tarfile.TarInfo(name="../x")
        )
    with pytest.raises(ValueError, match="path traversal"):
        backup_mod._check_unsafe_member(
            tarfile.TarInfo(name="/abs")
        )
    sym = tarfile.TarInfo(name="sym")
    sym.type = tarfile.SYMTYPE
    with pytest.raises(ValueError, match="symlink or hardlink"):
        backup_mod._check_unsafe_member(sym)
    lnk = tarfile.TarInfo(name="lnk")
    lnk.type = tarfile.LNKTYPE
    with pytest.raises(ValueError, match="symlink or hardlink"):
        backup_mod._check_unsafe_member(lnk)


# ---------------------------------------------------------------------------
# create refuses existing --out
# ---------------------------------------------------------------------------

def test_create_refuses_existing_out(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "file.txt").write_text("hi")
    out = tmp_path / "out.tar.gz"
    out.write_text("exists")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        backup_mod.create(str(data), str(out))


# ---------------------------------------------------------------------------
# create skips symlinks in data dir
# ---------------------------------------------------------------------------

def test_create_skips_symlinks(tmp_path: Path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "real.txt").write_text("real")
    (data / "link.txt").symlink_to("real.txt")
    out = tmp_path / "out.tar.gz"
    backup_mod.create(str(data), str(out))
    captured = capsys.readouterr()
    assert "skipping symlink" in captured.out


def test_create_handles_disappearing_file(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("alpha")
    (data / "b.txt").write_text("beta")
    out = tmp_path / "out.tar.gz"

    orig_copy2 = shutil.copy2
    call_count = 0

    def flaky_copy2(src, dst):
        nonlocal call_count
        call_count += 1
        if call_count == 2 and Path(dst).name == "b.txt":
            raise FileNotFoundError("vanished mid-backup")
        return orig_copy2(src, dst)

    monkeypatch.setattr(shutil, "copy2", flaky_copy2)
    out_path = backup_mod.create(str(data), str(out))
    assert Path(out_path).exists()
    with tarfile.open(str(out), "r:gz") as tar:
        names = tar.getnames()
    assert "a.txt" in names
    assert "b.txt" not in names


def test_create_appended_file_still_verifies(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("alpha")

    captured_paths = []

    orig_copy2 = shutil.copy2

    def capturing_copy2(src, dst):
        captured_paths.append(Path(dst))
        return orig_copy2(src, dst)

    monkeypatch.setattr(shutil, "copy2", capturing_copy2)

    out = tmp_path / "out.tar.gz"
    backup_mod.create(str(data), str(out))

    (data / "b.txt").write_text("beta")

    out2 = tmp_path / "out2.tar.gz"
    backup_mod.create(str(data), str(out2))

    assert backup_mod.verify(str(out2))
    with tarfile.open(str(out2), "r:gz") as tar:
        names = tar.getnames()
    assert "b.txt" in names

