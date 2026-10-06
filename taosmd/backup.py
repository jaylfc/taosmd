"""Backup, verify, and restore for the taOSmd data dir.

Stdlib only: ``create`` snapshots every file, using ``sqlite3.Connection.backup()``
for SQLite databases so WAL pages are included. ``verify`` checks every member
against the manifest. ``restore`` extracts only after a clean verify and never
overwrites existing content.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# upgrade-path: encryption at rest (gap #11)

_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_HEADER_LEN = len(_SQLITE_HEADER)
_MANIFEST_NAME = "MANIFEST.json"
_CONFIG_NAME = "config.json"
_SECRET_KEYS = {"server_token", "admin_token", "registry_token"}


def _resolve_data_dir(data_dir=None):
    from taosmd.config import _resolve_data_dir as _resolve  # noqa: PLC0415
    return Path(_resolve(data_dir))


def _is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(_SQLITE_HEADER_LEN) == _SQLITE_HEADER
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(io.DEFAULT_BUFFER_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _integrity_check(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return row[0] if row else "fail"
    finally:
        conn.close()


def _backup_sqlite(src: Path, dst: Path) -> None:
    conn = sqlite3.connect(str(src))
    try:
        bkp = sqlite3.connect(str(dst))
        try:
            conn.backup(bkp)
        finally:
            bkp.close()
    finally:
        conn.close()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _walk(data_dir: Path):
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            path = Path(root) / fname
            yield path.relative_to(data_dir), path


def _member_is_safe(member: tarfile.TarInfo) -> bool:
    if member.issym() or member.islnk():
        return False
    name = member.name
    if os.path.isabs(name):
        return False
    if ".." in Path(name).parts:
        return False
    return True


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for member in tf.getmembers():
        if not _member_is_safe(member):
            raise ValueError(f"unsafe tar member: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tf.extract(member, path=dest, filter="data")
        else:
            target = dest / member.name
            if not target.resolve().is_relative_to(dest.resolve()):
                raise ValueError(f"path traversal: {member.name}")
            tf.extract(member, path=dest)


def _has_secrets(data_dir: Path) -> bool:
    cfg = data_dir / _CONFIG_NAME
    if not cfg.exists():
        return False
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    return bool(_SECRET_KEYS & set(data.keys()))


def backup_create(data_dir, *, include_secrets=False, out=None):
    """Create a ``.tar.gz`` backup of *data_dir*.

    Returns the path to the written tarball.
    """
    data_dir = _resolve_data_dir(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data dir not found: {data_dir}")

    if not include_secrets and _has_secrets(data_dir):
        print(
            "WARNING: config.json contains bearer tokens and is EXCLUDED from this backup. "
            "The archive is unencrypted. Pass --include-secrets to store it.",
            file=__import__("sys").stderr,
        )

    if out is None:
        out = Path.cwd() / f"taosmd-backup-{_utc_now_ts()}.tar.gz"
    else:
        out = Path(out)

    tmp_dir = Path(tempfile.mkdtemp(prefix="taosmd-backup-"))
    manifest_files = []
    sqlite_tmp = tmp_dir / "sql"
    sqlite_tmp.mkdir()

    for rel, path in _walk(data_dir):
        rel_str = str(rel).replace("\\", "/")
        if rel_str == _CONFIG_NAME and not include_secrets:
            continue
        if path.name.endswith("-wal") or path.name.endswith("-shm") or path.name.endswith("-journal"):
            continue

        if _is_sqlite(path):
            tmp_db = sqlite_tmp / f"{hashlib.sha256(rel_str.encode()).hexdigest()}.db"
            _backup_sqlite(path, tmp_db)
            integrity = _integrity_check(tmp_db)
            if integrity != "ok":
                raise RuntimeError(
                    f"SQLite integrity_check failed for {rel_str}: {integrity}"
                )
            size = tmp_db.stat().st_size
            sha = _sha256_file(tmp_db)
            manifest_files.append({
                "path": rel_str,
                "size": size,
                "sha256": sha,
                "integrity_check": integrity,
            })
        else:
            size = path.stat().st_size
            sha = _sha256_file(path)
            manifest_files.append({
                "path": rel_str,
                "size": size,
                "sha256": sha,
            })

    manifest = {
        "taosmd_version": __import__("taosmd", fromlist=["__version__"]).__version__,
        "created_at": _utc_now_iso(),
        "source_data_dir": str(data_dir),
        "files": manifest_files,
    }

    with tarfile.open(out, "w:gz") as tf:
        # Write manifest first.
        info = tarfile.TarInfo(name=_MANIFEST_NAME)
        blob = json.dumps(manifest, indent=2).encode("utf-8")
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))

        for rel, path in _walk(data_dir):
            rel_str = str(rel).replace("\\", "/")
            if rel_str == _CONFIG_NAME and not include_secrets:
                continue
            if path.name.endswith("-wal") or path.name.endswith("-shm") or path.name.endswith("-journal"):
                continue
            if _is_sqlite(path):
                tmp_db = sqlite_tmp / f"{hashlib.sha256(rel_str.encode()).hexdigest()}.db"
                tf.add(tmp_db, arcname=rel_str)
            else:
                tf.add(path, arcname=rel_str)

    import shutil  # noqa: PLC0415
    shutil.rmtree(tmp_dir)
    return out


def backup_verify(path):
    """Verify a backup tarball. Returns 0 on success, 1 on any mismatch."""
    path = Path(path)
    if not path.is_file():
        print(f"error: backup file not found: {path}", file=__import__("sys").stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="taosmd-verify-"))
    try:
        try:
            with tarfile.open(path, "r:gz") as tf:
                try:
                    manifest_member = tf.getmember(_MANIFEST_NAME)
                except KeyError:
                    print("error: MANIFEST.json missing from tarball", file=__import__("sys").stderr)
                    return 1
                manifest_blob = tf.extractfile(manifest_member).read()
                manifest = json.loads(manifest_blob)

                members = tf.getmembers()
                manifest_paths = {f["path"] for f in manifest.get("files", [])}
                tarball_paths = set()
                bad = []

                for member in members:
                    if member.name == _MANIFEST_NAME:
                        continue
                    tarball_paths.add(member.name)

                extra_in_tarball = tarball_paths - manifest_paths
                missing_from_tarball = manifest_paths - tarball_paths
                if extra_in_tarball or missing_from_tarball:
                    if extra_in_tarball:
                        bad.append(f"extra in tarball: {sorted(extra_in_tarball)}")
                    if missing_from_tarball:
                        bad.append(f"missing from tarball: {sorted(missing_from_tarball)}")

                manifest_index = {f["path"]: f for f in manifest.get("files", [])}
                for member in members:
                    if member.name == _MANIFEST_NAME:
                        continue
                    entry = manifest_index.get(member.name)
                    if entry is None:
                        continue
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        bad.append(f"cannot extract: {member.name}")
                        continue
                    blob = extracted.read()
                    actual_sha = hashlib.sha256(blob).hexdigest()
                    if actual_sha != entry["sha256"]:
                        bad.append(
                            f"sha256 mismatch: {member.name} "
                            f"(expected {entry['sha256']}, got {actual_sha})"
                        )

                    if "integrity_check" in entry:
                        candidate = tmp / member.name
                        candidate.parent.mkdir(parents=True, exist_ok=True)
                        candidate.write_bytes(blob)
                        ic = _integrity_check(candidate)
                        if ic != "ok":
                            bad.append(f"integrity_check failed: {member.name}: {ic}")

                if bad:
                    for line in bad:
                        print(f"error: {line}", file=__import__("sys").stderr)
                    return 1
                return 0
        except Exception as exc:
            print(f"error: cannot read tarball: {exc}", file=__import__("sys").stderr)
            return 1
    finally:
        import shutil  # noqa: PLC0415
        shutil.rmtree(tmp, ignore_errors=True)


def backup_restore(path, dest, *, move_existing=False):
    """Restore a backup tarball to *dest*.

    Verifies first. Refuses to overwrite an existing non-empty directory
    unless *move_existing* is True.
    """
    path = Path(path)
    dest = Path(dest)
    if not path.is_file():
        raise FileNotFoundError(f"backup file not found: {path}")

    rc = backup_verify(path)
    if rc != 0:
        raise RuntimeError("backup verification failed; refusing to restore")

    if dest.exists():
        if any(dest.iterdir()):
            if not move_existing:
                raise RuntimeError(
                    f"destination {dest} exists and is non-empty; "
                    "use --move-existing to rename it first"
                )
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            moved = dest.with_name(f"{dest.name}.pre-restore-{ts}")
            dest.rename(moved)
            print(f"moved existing dir to {moved}")

    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as tf:
        _safe_extract(tf, dest)
