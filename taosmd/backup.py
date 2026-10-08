"""Backup, verify, and restore the taOSmd data dir.

Stdlib only. Uses ``sqlite3.Connection.backup()`` for every SQLite file so
WAL-mode stores are captured safely (committed pages in ``-wal`` are included
without copying the sidecar). Non-SQLite files are copied byte-for-byte.

The tarball contains a ``MANIFEST.json`` with per-file sha256 and SQLite
``PRAGMA integrity_check`` results.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from taosmd.config import _resolve_data_dir
from taosmd import __version__

SQLITE_HEADER = b"SQLite format 3\x00"
WAL_SIDECARS = ("-wal", "-shm", "-journal")
SECRET_KEYS = {"server_token", "admin_token", "registry_token"}


def _is_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(16) == SQLITE_HEADER
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _integrity_check(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return row[0] if row else "fail"
    except sqlite3.DatabaseError:
        return "fail"
    finally:
        conn.close()


def _copy_sqlite(src: Path, dst: Path) -> None:
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dst))
    try:
        src_conn.backup(dst_conn)
        dst_conn.commit()
    finally:
        src_conn.close()
        dst_conn.close()


def _safe_member_name(name: str) -> bool:
    if name.startswith("/"):
        return False
    parts = Path(name).parts
    if any(p == ".." for p in parts):
        return False
    return True


def _check_unsafe_member(member: tarfile.TarInfo) -> None:
    if member.issym() or member.islnk():
        raise ValueError(f"unsafe tar member: {member.name} (symlink or hardlink)")
    if not _safe_member_name(member.name):
        raise ValueError(f"unsafe tar member: {member.name} (path traversal)")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create(
    data_dir: str | None = None,
    out_path: str | None = None,
    include_secrets: bool = False,
) -> str:
    resolved = _resolve_data_dir(data_dir)
    base = Path(resolved).resolve()

    if out_path is None:
        out_path = str(Path.cwd() / f"taosmd-backup-{_utc_now_iso()}.tar.gz")
    out = Path(out_path).resolve()

    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {out}")

    out_rel = None
    try:
        out_rel = out.relative_to(base)
    except ValueError:
        pass

    tmp_dir = tempfile.mkdtemp(prefix="taosmd-backup-")
    staging = Path(tmp_dir)
    manifest: list[dict] = []
    try:
        for root, dirs, files in os.walk(resolved):
            dirs.sort()
            for name in sorted(files):
                src = Path(root) / name
                try:
                    rel = src.relative_to(base)
                except ValueError:
                    continue

                if out_rel is not None and rel == out_rel:
                    continue

                if any(rel.name.endswith(s) for s in WAL_SIDECARS):
                    continue

                if src.is_symlink():
                    print(f"backup: skipping symlink {rel}")
                    continue

                if rel.name == "config.json" and not include_secrets:
                    continue

                dst = staging / rel
                dst.parent.mkdir(parents=True, exist_ok=True)

                if _is_sqlite(src):
                    try:
                        _copy_sqlite(src, dst)
                    except OSError as exc:
                        print(f"backup: skipping {rel}: {exc}", file=sys.stderr)
                        continue
                    ic = _integrity_check(dst)
                    if ic != "ok":
                        raise RuntimeError(
                            f"integrity_check failed for {rel}: {ic}"
                        )
                    entry = {
                        "path": str(rel),
                        "size": dst.stat().st_size,
                        "sha256": _sha256_file(dst),
                        "sqlite_integrity": ic,
                    }
                else:
                    try:
                        shutil.copy2(src, dst)
                    except OSError as exc:
                        print(f"backup: skipping {rel}: {exc}", file=sys.stderr)
                        continue
                    entry = {
                        "path": str(rel),
                        "size": dst.stat().st_size,
                        "sha256": _sha256_file(dst),
                    }
                manifest.append(entry)

        manifest_path = staging / "MANIFEST.json"
        manifest_doc = {
            "taosmd_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_data_dir": str(base),
            "files": manifest,
        }
        manifest_path.write_text(json.dumps(manifest_doc, indent=2) + "\n")

        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(out), "w:gz") as tar:
            for entry in manifest:
                tar.add(str(staging / entry["path"]), arcname=entry["path"])
            tar.add(str(manifest_path), arcname="MANIFEST.json")

        if include_secrets:
            print(
                "warning: backup includes config.json which holds bearer tokens "
                "(server_token, admin_token, registry_token). The archive is "
                "unencrypted; store the tarball securely.",
                file=sys.stderr,
            )

        return str(out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def verify(tar_path: str) -> bool:
    try:
        tar = tarfile.open(tar_path, "r:gz")
    except tarfile.ReadError as exc:
        print(f"error: cannot open tarball: {exc}", file=sys.stderr)
        return False

    try:
        try:
            manifest_member = tar.getmember("MANIFEST.json")
            manifest_fh = tar.extractfile(manifest_member)
            if manifest_fh is None:
                print("error: MANIFEST.json missing from tarball", file=sys.stderr)
                return False
            manifest = json.loads(manifest_fh.read().decode("utf-8"))
        except (KeyError, json.JSONDecodeError, tarfile.ReadError) as exc:
            print(f"error: cannot read MANIFEST.json: {exc}", file=sys.stderr)
            return False

        manifest_files = {f["path"]: f for f in manifest.get("files", [])}
        tar_names = set()

        tmp_dir = tempfile.mkdtemp(prefix="taosmd-verify-")
        try:
            for member in tar.getmembers():
                if member.name == "MANIFEST.json":
                    continue
                try:
                    _check_unsafe_member(member)
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return False

                tar_names.add(member.name)
                fh = tar.extractfile(member)
                if fh is None:
                    print(
                        f"error: cannot read {member.name} from tarball",
                        file=sys.stderr,
                    )
                    return False

                data = fh.read()
                expected = manifest_files.get(member.name)
                if expected is None:
                    print(
                        f"error: {member.name} in tarball but not in manifest",
                        file=sys.stderr,
                    )
                    return False

                actual_sha = hashlib.sha256(data).hexdigest()
                if actual_sha != expected["sha256"]:
                    print(
                        f"error: sha256 mismatch for {member.name}: "
                        f"expected {expected['sha256']}, got {actual_sha}",
                        file=sys.stderr,
                    )
                    return False

                if "sqlite_integrity" in expected:
                    db_path = Path(tmp_dir) / member.name
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    db_path.write_bytes(data)
                    ic = _integrity_check(db_path)
                    if ic != "ok":
                        print(
                            f"error: integrity_check failed for {member.name}: {ic}",
                            file=sys.stderr,
                        )
                        return False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        for path in manifest_files:
            if path not in tar_names:
                print(
                    f"error: {path} in manifest but missing from tarball",
                    file=sys.stderr,
                )
                return False

        return True
    finally:
        tar.close()


def restore(
    tar_path: str,
    to_dir: str,
    move_existing: bool = False,
) -> None:
    if not verify(tar_path):
        raise RuntimeError("backup verification failed; refusing to restore")

    target = Path(to_dir).resolve()

    if target.exists():
        entries = list(target.iterdir())
        if entries:
            if not move_existing:
                raise RuntimeError(
                    f"target {target} is non-empty; pass --move-existing to "
                    f"rename it to {target}.pre-restore-<ts> first"
                )

    tmp = target.parent / f"{target.name}.restore-{_utc_now_iso()}"
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        tar = tarfile.open(tar_path, "r:gz")
        try:
            for member in tar.getmembers():
                if member.name == "MANIFEST.json":
                    continue
                _check_unsafe_member(member)
            tar.extractall(str(tmp), filter="data")
        finally:
            tar.close()

        if target.exists():
            entries = list(target.iterdir())
            if entries:
                backup_name = f"{target.name}.pre-restore-{_utc_now_iso()}"
                backup_path = target.parent / backup_name
                target.rename(backup_path)
                print(f"moved existing {target} to {backup_path}")

        tmp.rename(target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
