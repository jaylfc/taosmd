"""Copy-then-verify import of a legacy 0.4-era ``./data`` tree.

``taosmd migrate --import-legacy SRC [--data-dir DEST] [--apply]`` copies a
legacy data directory into the current data dir and verifies every copy
before anything is renamed into place.

The source is never modified, moved, renamed, or deleted: no write, no
rename, no chmod. Everything is copied into DEST under a temp name, verified
(byte-identical sha256 for files, ``PRAGMA integrity_check`` plus per-table
row counts for SQLite databases), and only then renamed into place. On any
failure only the temp copies this run made are removed and the command exits
non-zero. Without ``--apply`` it is a dry run: it lists what would be copied
and exits 0.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import sys
from pathlib import Path

from .migrations import DB_FILES

#: Store names a legacy ./data tree may hold, in a stable copy order.
DB_STORES: tuple[str, ...] = tuple(sorted(DB_FILES.values()))
DIR_STORES: tuple[str, ...] = ("agent-memory", "archive")
FILE_STORES: tuple[str, ...] = ("agents.json",)
ALL_STORES: tuple[str, ...] = (*DB_STORES, *DIR_STORES, *FILE_STORES)

_TEMP_SUFFIX = ".taosmd-import-tmp"


class LegacyImportError(Exception):
    """Fatal import problem reported to the user with a non-zero exit."""


# ----------------------------------------------------------------------
# read-only probe
# ----------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _quoted_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name",
    ).fetchall()
    return [row[0] for row in rows]


def _table_row_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT count(*) FROM {_quoted_ident(table)}").fetchone()[0])


def _probe_database(path: Path) -> dict:
    """Read-only probe of one SQLite database. Writes nothing to ``path``."""
    info: dict = {"kind": "db", "size": path.stat().st_size}
    wal = path.parent / (path.name + "-wal")
    journal = path.parent / (path.name + "-journal")
    info["wal_bytes"] = wal.stat().st_size if wal.exists() else 0
    info["hot_journal"] = bool(
        journal.exists() and journal.stat().st_size > 0
    )
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        info["error"] = f"cannot open read-only: {exc}"
        return info
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        info["integrity"] = rows[0][0] if rows else "unknown"
        info["integrity_ok"] = len(rows) == 1 and rows[0][0] == "ok"
        info["user_version"] = int(
            conn.execute("PRAGMA user_version").fetchone()[0]
        )
        info["tables"] = {}
        for table in _table_names(conn):
            info["tables"][table] = _table_row_count(conn, table)
    except sqlite3.Error as exc:
        info["error"] = f"probe failed: {exc}"
    finally:
        conn.close()
    return info


def _probe_directory(path: Path) -> dict:
    files = [entry for entry in path.rglob("*") if entry.is_file()]
    return {
        "kind": "directory",
        "size": _directory_size(path),
        "files": len(files),
    }


def store_probe(src: Path) -> dict:
    """Read-only probe of a legacy data directory. Writes nothing.

    Returns a report with one entry per store present in ``src``, each with
    its size, plus ``total_bytes``. For databases the report also carries
    ``user_version``, ``integrity_ok``, per-table row counts, ``wal_bytes``
    and ``hot_journal``.
    """
    report: dict = {"src": str(src), "stores": {}, "total_bytes": 0}
    total = 0
    for name in DB_STORES:
        path = src / name
        if not path.exists():
            continue
        report["stores"][name] = _probe_database(path)
        total += report["stores"][name]["size"]
    for name in DIR_STORES:
        path = src / name
        if not path.exists():
            continue
        report["stores"][name] = _probe_directory(path)
        total += report["stores"][name]["size"]
    path = src / "agents.json"
    if path.exists():
        size = path.stat().st_size
        report["stores"]["agents.json"] = {"kind": "file", "size": size}
        total += size
    report["total_bytes"] = total
    return report


# ----------------------------------------------------------------------
# copy + verify
# ----------------------------------------------------------------------


def _copy_database(src_path: Path, temp_path: Path) -> None:
    """Online backup of ``src_path`` into ``temp_path`` (read-only source)."""
    src_conn = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    try:
        dst_conn = sqlite3.connect(str(temp_path))
        try:
            with dst_conn:
                src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _verify_database(src_path: Path, temp_path: Path) -> bool:
    """True when the temp copy passes integrity_check and row-count compare."""
    try:
        src_conn = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        dst_conn = sqlite3.connect(f"file:{temp_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        rows = dst_conn.execute("PRAGMA integrity_check").fetchall()
        if not (len(rows) == 1 and rows[0][0] == "ok"):
            return False
        if _table_names(src_conn) != _table_names(dst_conn):
            return False
        for table in _table_names(src_conn):
            if _table_row_count(src_conn, table) != _table_row_count(
                dst_conn, table
            ):
                return False
        return True
    except sqlite3.Error:
        return False
    finally:
        src_conn.close()
        dst_conn.close()


def _copy_tree_or_file(src_path: Path, temp_path: Path) -> None:
    if src_path.is_dir():
        shutil.copytree(src_path, temp_path, copy_function=shutil.copy2)
    else:
        shutil.copy2(src_path, temp_path)


def _verify_tree(src_path: Path, temp_path: Path) -> bool:
    """True when every file under src_path is sha256-identical in temp_path."""
    try:
        src_files = {
            entry.relative_to(src_path): entry
            for entry in src_path.rglob("*")
            if entry.is_file()
        }
    except OSError:
        return False
    for rel, src_file in src_files.items():
        temp_file = temp_path / rel
        if not temp_file.is_file():
            return False
        try:
            if _file_sha256(src_file) != _file_sha256(temp_file):
                return False
        except OSError:
            return False
    return True


# ----------------------------------------------------------------------
# command
# ----------------------------------------------------------------------


def _temp_name(dest: Path, store: str) -> Path:
    return dest / f".{store}{_TEMP_SUFFIX}-{os.getpid()}"


def legacy_import(src: Path, dest: Path, apply: bool = False) -> int:
    """Run the import-legacy command. Returns a process exit code."""
    src = src.resolve()
    dest = dest.resolve()

    if src == dest:
        print(
            f"error: --import-legacy source {src} is the same as the data "
            f"dir {dest}; refusing to copy a directory onto itself",
            file=sys.stderr,
        )
        return 2

    present = [store for store in ALL_STORES if (src / store).exists()]
    if not present:
        print(
            f"error: no legacy stores found in {src}; nothing to import "
            f"(expected one of: {', '.join(ALL_STORES)})",
            file=sys.stderr,
        )
        return 2

    collisions = [store for store in present if (dest / store).exists()]
    if collisions:
        print(
            f"error: {dest} already holds {', '.join(collisions)}; "
            f"refusing to overwrite or merge",
            file=sys.stderr,
        )
        return 2

    total = _directory_size(src)
    free = shutil.disk_usage(dest).free
    if free < 2 * total:
        print(
            f"error: only {free} bytes free at {dest}; importing {src} "
            f"needs at least {2 * total} bytes (2x source size {total})",
            file=sys.stderr,
        )
        return 2

    for name in DB_STORES:
        path = src / name
        if not path.exists():
            continue
        wal = path.parent / (path.name + "-wal")
        if wal.exists() and wal.stat().st_size > 0:
            print(
                f"error: {name} has a non-empty -wal; stop the writer, or "
                f"run `taosmd migrate --checkpoint` if that exists, then "
                f"retry",
                file=sys.stderr,
            )
            return 2
        journal = path.parent / (path.name + "-journal")
        if journal.exists() and journal.stat().st_size > 0:
            print(
                f"error: {name} has a hot journal; stop the writer and let "
                f"it checkpoint, or run `taosmd migrate --checkpoint` if "
                f"that exists, then retry",
                file=sys.stderr,
            )
            return 2

    report = store_probe(src)

    if not apply:
        print(f"taosmd migrate --import-legacy {src} (dry run, nothing will be copied)")
        print(f"  source: {src}")
        print(f"  destination: {dest}")
        if not present:
            print("  no stores to copy")
        else:
            print("  stores to copy:")
            for name in present:
                info = report["stores"].get(name, {})
                size = info.get("size", "-")
                line = f"    {name:<32} {size} bytes"
                if info.get("kind") == "db":
                    extra = []
                    if info.get("integrity_ok") is not None:
                        extra.append(
                            f"integrity={'ok' if info['integrity_ok'] else info.get('integrity')}"
                        )
                    if info.get("wal_bytes"):
                        extra.append(f"wal={info['wal_bytes']}B")
                    if info.get("hot_journal"):
                        extra.append("hot-journal")
                    if extra:
                        line += "  (" + ", ".join(extra) + ")"
                print(line)
        print(f"  total: {report['total_bytes']} bytes")
        print(
            "  use --apply to copy each store into the data dir and verify it "
            "before anything is renamed into place"
        )
        return 0

    # --apply: copy everything under temp names, verify, then rename.
    temps: list[tuple[Path, Path]] = []
    try:
        for name in present:
            src_path = src / name
            temp_path = _temp_name(dest, name)
            final_path = dest / name
            # Track the temp before we touch anything so a failure removes
            # exactly the copies this run made, and nothing else.
            temps.append((temp_path, final_path))
            if name in DB_STORES:
                _copy_database(src_path, temp_path)
                if not _verify_database(src_path, temp_path):
                    raise LegacyImportError(
                        f"verification failed for {name} (integrity or row counts)"
                    )
            else:
                _copy_tree_or_file(src_path, temp_path)
                if not _verify_tree(src_path, temp_path):
                    raise LegacyImportError(
                        f"verification failed for {name} (sha256 mismatch)"
                    )
    except LegacyImportError:
        _remove_temps(temps)
        print(
            f"error: import aborted; removed temp copies from {dest}; "
            f"source {src} is unchanged",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        _remove_temps(temps)
        print(
            f"error: import failed: {exc}; removed temp copies from {dest}; "
            f"source {src} is unchanged",
            file=sys.stderr,
        )
        return 1

    for temp_path, final_path in temps:
        os.replace(temp_path, final_path)

    print(f"imported {len(temps)} store(s) from {src} into {dest}")
    for _, final_path in temps:
        print(f"  {final_path}")
    print(
        f"note: {src} was not modified; remove it by hand once you are "
        f"satisfied the import is good"
    )
    return 0


def _remove_temps(temps: list[tuple[Path, Path]]) -> None:
    for temp_path, _ in temps:
        try:
            if temp_path.is_dir():
                shutil.rmtree(temp_path, ignore_errors=True)
            elif temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
