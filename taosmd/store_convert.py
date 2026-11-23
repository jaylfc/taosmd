"""Guarded float32 conversion for dense vector stores (Card B).

``convert_dense_store`` converts a legacy JSON-stored dense vector memory to
base64-encoded little-endian float32 in a single verified transaction:

1. read-only format probe (Card C shape) -- old / current / unknown / newer
2. exclusive lock (``BEGIN IMMEDIATE``); refuse if a live writer holds it
3. free-space checks on the backup and store filesystems
4. backup via ``sqlite3.Connection.backup()`` (never a file copy), fsync'd
5. backup verification: integrity_check, row count, content hash
6. batched conversion inside the same transaction (bounded peak RSS)
7. post-checks before COMMIT: row count, per-row equality against the backup,
   non-embedding column hash, recall equivalence
8. COMMIT, marker write, optional VACUUM; backup kept (zero-loss)

On any failure the transaction rolls back (live store untouched) and the
result carries exit code 4 ("failed, restored"); a post-commit integrity
failure restores from the backup in the reverse direction.

This module intentionally contains a small read-only format probe shaped like
the Card C probe (``taosmd/store_probe.py`` is tracked separately and has not
merged). When Card C lands, this probe should delegate to it rather than be
duplicated.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np

from . import migrations

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000
RECALL_QUERIES = 20
RECALL_TOP_K = 10
# Busy timeout (ms) for acquiring the exclusive conversion lock. A live writer
# (e.g. ``taosmd serve``) that holds the store past this window causes a
# clean refusal rather than an indefinite block.
BUSY_LOCK_MS = 5000

_PROBE_STATUS_CURRENT = "current"
_PROBE_STATUS_OLD = "old"
_PROBE_STATUS_UNKNOWN = "unknown"
_PROBE_STATUS_NEWER = "newer"

_EXIT_CODES = {
    _PROBE_STATUS_CURRENT: 0,
    "already_current": 0,
    "converted": 0,
    "refused": 2,
    _PROBE_STATUS_NEWER: 3,
    "failed_restored": 4,
    "unexpected": 1,
}

# Mode-signature keys this build understands. A marker carrying any other key
# is from a newer build.
_KNOWN_MODE_KEYS = {"late_interaction", "binary_quant", "dense_format", "embedder"}

_HASH_COLUMNS_WITH_EMBEDDING = (
    "id, embedding, text, metadata_json, created_at, valid_to"
)
_HASH_COLUMNS_WITHOUT_EMBEDDING = (
    "id, text, metadata_json, created_at, valid_to"
)


class ConvertError(Exception):
    """Raised internally when a conversion post-check fails.

    The caller rolls the transaction back and reports exit code 4 so the live
    store stays byte-identical to its pre-run state.
    """


class BackupVerifyError(Exception):
    """Raised internally when the pre-conversion backup fails verification."""


@dataclass(frozen=True)
class ProbeResult:
    """Read-only format probe result (Card C shape, single vector store)."""

    status: str  # "current" | "old" | "unknown" | "newer"
    detail: str


@dataclass
class ConvertResult:
    """Outcome of :func:`convert_dense_store`."""

    status: str  # "converted" | "already_current" | "refused" | "newer" | "failed_restored" | "unexpected"
    exit_code: int
    message: str
    backup_path: str | None = None
    rows_converted: int = 0


# ----------------------------------------------------------------------
# mode-signature helpers
# ----------------------------------------------------------------------


def _parse_mode(sig: str | None) -> dict[str, str]:
    """Parse a ``store_meta.mode`` signature string into a key->value dict."""
    out: dict[str, str] = {}
    if not sig:
        return out
    for part in sig.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ----------------------------------------------------------------------
# read-only format probe (Card C shape)
# ----------------------------------------------------------------------


def _probe_from_conn(conn: sqlite3.Connection) -> ProbeResult:
    """Probe the vector store format using an already-open read-only
    connection. Never writes PRAGMAs or creates sidecar files."""
    try:
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error:
        return ProbeResult(_PROBE_STATUS_UNKNOWN, "unreadable database")

    latest = migrations.latest_version("vector_memory")
    if user_version > latest:
        return ProbeResult(
            _PROBE_STATUS_NEWER,
            f"user_version {user_version} > latest {latest}",
        )

    row = conn.execute(
        "SELECT value FROM store_meta WHERE key = 'mode'"
    ).fetchone()
    recorded = row[0] if row else None

    try:
        total = int(
            conn.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]
        )
        json_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM vector_memory WHERE embedding LIKE '[%'"
            ).fetchone()[0]
        )
    except sqlite3.Error:
        return ProbeResult(_PROBE_STATUS_UNKNOWN, "vector_memory table unreadable")

    rec = _parse_mode(recorded) if recorded else {}

    if recorded is not None:
        unknown_keys = set(rec) - _KNOWN_MODE_KEYS
        if unknown_keys:
            return ProbeResult(
                _PROBE_STATUS_NEWER,
                f"marker carries unknown keys {sorted(unknown_keys)}",
            )
        dense = rec.get("dense_format")
        if dense is not None:
            if dense != "f32":
                return ProbeResult(
                    _PROBE_STATUS_UNKNOWN,
                    f"unsupported dense_format {dense!r}",
                )
            if json_count > 0:
                return ProbeResult(
                    _PROBE_STATUS_UNKNOWN,
                    "mixed rows: f32 marker with JSON embeddings",
                )
            return ProbeResult(
                _PROBE_STATUS_CURRENT,
                "dense_format=f32 marker, all rows float32",
            )
        # Legacy marker lacking dense_format: format needs upgrading (even on
        # an empty store, the marker itself is legacy and must be stamped).
        return ProbeResult(
            _PROBE_STATUS_OLD,
            "legacy marker without dense_format; conversion needed",
        )

    # No marker at all.
    if json_count > 0:
        return ProbeResult(
            _PROBE_STATUS_OLD, "no mode marker; JSON rows need conversion"
        )
    return ProbeResult(_PROBE_STATUS_CURRENT, "empty or fresh store")


def _probe_vector_store(db_path: Path) -> ProbeResult:
    """Sidecar-free read-only probe of the vector store (Card C shape).

    Opens with ``immutable=1`` so no ``-wal``/`-shm`` sidecars are created and
    un-checkpointed WAL content is reported as ``unknown`` rather than guessed.
    """
    if not db_path.exists():
        return ProbeResult(_PROBE_STATUS_CURRENT, "no vector store yet")
    wal_path = Path(str(db_path) + "-wal")
    if wal_path.exists() and wal_path.stat().st_size > 0:
        return ProbeResult(
            _PROBE_STATUS_UNKNOWN, "live writer or uncheckpointed WAL present"
        )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        return _probe_from_conn(conn)
    finally:
        conn.close()


def _probe_recovered(db_path: Path) -> ProbeResult | None:
    """Open the store normally (WAL recovery runs) and re-probe.

    Used to distinguish a benign leftover -wal from an interrupted conversion
    (atomic, so the store is still intact) from a genuinely unreadable store.
    Returns None when the store cannot be opened at all.
    """
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return None
    try:
        return _probe_from_conn(conn)
    finally:
        conn.close()


# ----------------------------------------------------------------------
# embedding codecs
# ----------------------------------------------------------------------


def _json_row_to_f32(emb_json: str) -> str:
    """Convert one JSON embedding to base64 little-endian float32.

    This is the per-row encoder used by the conversion loop. It is a module
    global so tests can patch it narrowly (e.g. to rotate vectors and prove
    the post-check detects corruption and rolls back).
    """
    vec = json.loads(emb_json)
    arr = np.asarray(vec, dtype="<f4")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _decode_embedding(raw) -> np.ndarray:
    """Decode a stored embedding (JSON legacy or base64 f32) to a float32
    vector. Raises on unreadable/malformed content; callers skip such rows."""
    if raw is None:
        raise ValueError("null embedding")
    if isinstance(raw, str) and raw.startswith("["):
        return np.asarray(json.loads(raw), dtype="<f4")
    return np.frombuffer(base64.b64decode(raw), dtype="<f4")


# ----------------------------------------------------------------------
# hashing
# ----------------------------------------------------------------------


def _hash_rows(conn: sqlite3.Connection, include_embedding: bool) -> bytes:
    """SHA-256 over the selected columns of every row, in id order."""
    cols = (
        _HASH_COLUMNS_WITH_EMBEDDING
        if include_embedding
        else _HASH_COLUMNS_WITHOUT_EMBEDDING
    )
    h = hashlib.sha256()
    for row in conn.execute(
        f"SELECT {cols} FROM vector_memory ORDER BY id"
    ).fetchall():
        h.update(repr(tuple(row)).encode("utf-8"))
    return h.digest()


def _hash_row_sequence(rows, include_embedding: bool) -> bytes:
    """SHA-256 over the selected columns of an iterable of rows, in id order.

    ``rows`` may be sqlite3 ``Row`` objects or tuples. Used to hash the
    already-loaded backup rows after the backup connection is closed.
    """
    keep = (
        slice(0, 6)
        if include_embedding
        else (0, 2, 3, 4, 5)
    )
    h = hashlib.sha256()
    for row in rows:
        t = tuple(row)
        if include_embedding:
            selected = t
        else:
            selected = (t[0], t[2], t[3], t[4], t[5])
        h.update(repr(selected).encode("utf-8"))
    return h.digest()


# ----------------------------------------------------------------------
# recall equivalence
# ----------------------------------------------------------------------


def _topk_pairs(pairs: list[tuple[int, np.ndarray]], query_vec, k: int):
    """Top-k (id, score) by cosine similarity over ``(id, vector)`` pairs."""
    valid = [(i, v) for i, v in pairs if v is not None and v.size]
    if not valid:
        return []
    ids = [i for i, _ in valid]
    mat = np.stack([v for _, v in valid]).astype(np.float32)
    q = np.asarray(query_vec, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-8)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    sims = mn @ qn
    order = np.argsort(-sims, kind="stable")[: min(k, len(ids))]
    return [(int(ids[i]), float(sims[i])) for i in order]


def _topk_conn(conn: sqlite3.Connection, query_vec, k: int):
    rows = conn.execute(
        "SELECT id, embedding FROM vector_memory ORDER BY id"
    ).fetchall()
    pairs = []
    for r in rows:
        try:
            pairs.append((r["id"], _decode_embedding(r["embedding"])))
        except Exception:
            continue
    return _topk_pairs(pairs, query_vec, k)


def _recall_equivalent(conn: sqlite3.Connection, backup_rows: dict) -> bool:
    """Compare top-k recall using 20 stored vectors as queries, pre (backup)
    vs post (live). Identical ids and float32 scores required."""
    backup_pairs = []
    for brow in backup_rows.values():
        try:
            backup_pairs.append(
                (brow["id"], _decode_embedding(brow["embedding"]))
            )
        except Exception:
            continue
    if not backup_pairs:
        return True

    step = max(1, len(backup_pairs) // RECALL_QUERIES)
    queries = backup_pairs[::step][:RECALL_QUERIES]

    pre_mat = np.stack([v for _, v in backup_pairs]).astype(np.float32)
    pre_ids = [i for i, _ in backup_pairs]

    live_pairs = []
    for r in conn.execute(
        "SELECT id, embedding FROM vector_memory ORDER BY id"
    ).fetchall():
        try:
            live_pairs.append((r[0], _decode_embedding(r[1])))
        except Exception:
            continue
    if not live_pairs:
        return len(backup_pairs) == 0
    live_mat = np.stack([v for _, v in live_pairs]).astype(np.float32)
    live_ids = [i for i, _ in live_pairs]

    for _qid, qvec in queries:
        pre = _topk_pairs(
            list(zip(pre_ids, [row for row in pre_mat])), qvec, RECALL_TOP_K
        )
        post = _topk_pairs(
            list(zip(live_ids, [row for row in live_mat])), qvec, RECALL_TOP_K
        )
        if pre != post:
            return False
    return True


# ----------------------------------------------------------------------
# filesystem helpers
# ----------------------------------------------------------------------


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _delete_file(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        logger.warning("could not delete partial backup %s", path)


# ----------------------------------------------------------------------
# marker
# ----------------------------------------------------------------------


def _build_marker(conn: sqlite3.Connection) -> str:
    """Build the post-conversion mode marker, preserving the recorded
    embedder identity and setting dense_format=f32."""
    row = conn.execute(
        "SELECT value FROM store_meta WHERE key = 'mode'"
    ).fetchone()
    recorded = row[0] if row else None
    rec = _parse_mode(recorded) if recorded else {}
    embedder = rec.get("embedder", "qmd:qmd")
    return (
        f"late_interaction=0;binary_quant=0;"
        f"dense_format=f32;embedder={embedder}"
    )


# ----------------------------------------------------------------------
# restore
# ----------------------------------------------------------------------


def _restore_live_from_backup(db_path: Path, backup_path: Path) -> None:
    """Overwrite the live store with the backup (backup() in reverse)."""
    live = sqlite3.connect(str(db_path))
    try:
        live.execute("BEGIN IMMEDIATE")
        backup = sqlite3.connect(str(backup_path))
        try:
            backup.backup(live)
        finally:
            backup.close()
        live.execute("COMMIT")
    except Exception:
        try:
            live.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        live.close()


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------


def convert_dense_store(
    db_path: Union[str, Path],
    *,
    backup_dir: Union[str, Path, None] = None,
) -> ConvertResult:
    """Guard-convert a dense vector store from JSON to base64 float32.

    Returns a :class:`ConvertResult` whose ``exit_code`` follows the Card B
    contract: 0 converted/already-current, 2 refused (nothing touched), 3
    newer/unknown format, 4 failed-and-restored, 1 unexpected error.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return ConvertResult(
            "already_current", 0, "no vector store to convert"
        )

    # Step 1: read-only probe.
    probe = _probe_vector_store(db_path)
    if probe.status == _PROBE_STATUS_CURRENT:
        return ConvertResult(
            "already_current", 0, f"already current: {probe.detail}"
        )
    if probe.status == _PROBE_STATUS_NEWER:
        return ConvertResult(
            "newer", 3, f"store newer than this build: {probe.detail}"
        )
    if probe.status == _PROBE_STATUS_UNKNOWN:
        # A non-empty -wal is often a leftover from an interrupted conversion:
        # the transaction is atomic, so the store itself is still the old
        # intact store. Open the store normally (WAL recovery runs) and re-probe
        # before refusing.
        recovered = _probe_recovered(db_path)
        if recovered is None:
            return ConvertResult(
                "refused", 2, "unknown store format (unreadable)"
            )
        if recovered.status == _PROBE_STATUS_CURRENT:
            return ConvertResult(
                "already_current", 0, f"already current: {recovered.detail}"
            )
        if recovered.status == _PROBE_STATUS_NEWER:
            return ConvertResult(
                "newer", 3, f"store newer than this build: {recovered.detail}"
            )
        if recovered.status != _PROBE_STATUS_OLD:
            return ConvertResult(
                "refused", 2, f"unknown store format: {recovered.detail}"
            )
        probe = recovered
    # probe.status == _PROBE_STATUS_OLD -> proceed

    data_dir = db_path.parent
    backup_root = (
        Path(backup_dir) if backup_dir is not None else (data_dir / "backups")
    )

    # Step 2: exclusive lock so no live writer (e.g. taosmd serve) holds it.
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"PRAGMA busy_timeout={BUSY_LOCK_MS}")
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        conn.close()
        return ConvertResult(
            "refused",
            2,
            "store is locked by a live writer (taosmd serve holds it)",
        )

    try:
        # Re-probe under the lock (the store may have changed since step 1).
        probe = _probe_from_conn(conn)
        if probe.status != _PROBE_STATUS_OLD:
            conn.execute("ROLLBACK")
            if probe.status == _PROBE_STATUS_CURRENT:
                return ConvertResult(
                    "already_current", 0, f"already current: {probe.detail}"
                )
            if probe.status == _PROBE_STATUS_NEWER:
                return ConvertResult(
                    "newer", 3, f"store newer than this build: {probe.detail}"
                )
            return ConvertResult(
                "refused", 2, f"unknown store format: {probe.detail}"
            )

        # Step 3: free-space checks, nothing touched yet.
        db_size = db_path.stat().st_size
        wal_path = Path(str(db_path) + "-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        backup_free = shutil.disk_usage(
            backup_root if backup_root.exists() else data_dir
        ).free
        store_free = shutil.disk_usage(data_dir).free
        need_backup = db_size + wal_size + (db_size + wal_size) // 10
        if backup_free < need_backup:
            conn.execute("ROLLBACK")
            return ConvertResult(
                "refused", 2, "insufficient free space on backup filesystem"
            )
        if store_free < 2 * db_size:
            conn.execute("ROLLBACK")
            return ConvertResult(
                "refused", 2, "insufficient free space on store filesystem"
            )

        # Step 4: backup via sqlite3 online backup (never a file copy), fsync.
        # The source is a *separate* read-only connection: ``backup()`` does not
        # run on the same connection that holds the ``BEGIN IMMEDIATE`` lock.
        backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_root / f"vector-memory.{stamp}.pre-f32.db"
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            target = sqlite3.connect(str(backup_path))
            try:
                src.backup(target)
            finally:
                target.close()
        finally:
            src.close()
        _fsync_file(backup_path)
        _fsync_dir(backup_root)

        # Step 5: verify the backup.
        backup_conn = sqlite3.connect(str(backup_path))
        backup_conn.row_factory = sqlite3.Row
        try:
            integrity = backup_conn.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if integrity != "ok":
                raise BackupVerifyError(f"integrity_check={integrity}")
            backup_count = int(
                backup_conn.execute(
                    "SELECT COUNT(*) FROM vector_memory"
                ).fetchone()[0]
            )
            live_count = int(
                conn.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]
            )
            if backup_count != live_count:
                raise BackupVerifyError(
                    f"row count backup={backup_count} live={live_count}"
                )
            if _hash_rows(backup_conn, True) != _hash_rows(conn, True):
                raise BackupVerifyError("content hash mismatch")
        except BackupVerifyError:
            backup_conn.close()
            _delete_file(backup_path)
            conn.execute("ROLLBACK")
            return ConvertResult(
                "refused", 2, "backup verification failed; backup deleted"
            )
        backup_rows = {
            r["id"]: r
            for r in backup_conn.execute(
                "SELECT id, embedding, text, metadata_json, created_at, valid_to "
                "FROM vector_memory ORDER BY id"
            ).fetchall()
        }
        backup_hash_no_emb = _hash_row_sequence(
            backup_rows.values(), include_embedding=False
        )
        backup_conn.close()

        count_before = live_count

        # Step 6: convert in one transaction, streaming in bounded batches.
        rows_converted = 0
        converted_ids: set[int] = set()
        last_id = 0
        while True:
            batch = conn.execute(
                "SELECT id, embedding FROM vector_memory "
                "WHERE embedding LIKE '[%' AND id > ? ORDER BY id LIMIT ?",
                (last_id, BATCH_SIZE),
            ).fetchall()
            if not batch:
                break
            updates = []
            for row in batch:
                try:
                    new_emb = _json_row_to_f32(row["embedding"])
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    # Card E: a malformed row is left as-is, logged, counted;
                    # the conversion still converts every other row.
                    logger.warning(
                        "skipping malformed embedding row %s: %s",
                        row["id"],
                        exc,
                    )
                    continue
                updates.append((new_emb, row["id"]))
                converted_ids.add(row["id"])
            if updates:
                conn.executemany(
                    "UPDATE vector_memory SET embedding = ? WHERE id = ?",
                    updates,
                )
            rows_converted += len(updates)
            last_id = batch[-1]["id"]

        # Step 7: post-checks before COMMIT.
        count_after = int(
            conn.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]
        )
        if count_after != count_before:
            raise ConvertError(
                f"row count changed {count_before} -> {count_after}"
            )

        for row in conn.execute(
            "SELECT id, embedding FROM vector_memory ORDER BY id"
        ):
            rid = row["id"]
            if rid not in converted_ids:
                continue  # pre-existing f32 or malformed; not converted
            bk = backup_rows.get(rid)
            if bk is None:
                raise ConvertError(f"row {rid} missing from backup")
            new_arr = np.frombuffer(
                base64.b64decode(row["embedding"]), dtype="<f4"
            )
            try:
                old_arr = np.asarray(
                    json.loads(bk["embedding"]), dtype="<f4"
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not np.array_equal(new_arr, old_arr):
                raise ConvertError(f"per-row embedding mismatch at id {rid}")

        if _hash_rows(conn, False) != backup_hash_no_emb:
            raise ConvertError("non-embedding column hash mismatch")

        if not _recall_equivalent(conn, backup_rows):
            raise ConvertError("recall equivalence check failed")

        # Step 8: write marker and COMMIT.
        marker = _build_marker(conn)
        conn.execute(
            "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', ?)",
            (marker,),
        )
        conn.execute("COMMIT")
    except ConvertError:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        conn.close()
        return ConvertResult(
            "failed_restored",
            4,
            "conversion failed; live store rolled back, byte-identical to pre-run",
            rows_converted=rows_converted,
        )
    except Exception as exc:  # unexpected error
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        conn.close()
        return ConvertResult(
            "unexpected",
            1,
            f"unexpected conversion error: {exc} (backup at {backup_path})",
            backup_path=str(backup_path),
        )

    # Post-commit integrity check; on failure restore from the backup.
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    except sqlite3.Error:
        integrity = "error"
    if integrity != "ok":
        _restore_live_from_backup(db_path, backup_path)
        conn.close()
        return ConvertResult(
            "failed_restored",
            4,
            "post-commit integrity check failed; restored from backup",
            backup_path=str(backup_path),
            rows_converted=rows_converted,
        )

    # Optional VACUUM if space allows.
    try:
        if shutil.disk_usage(data_dir).free >= 2 * db_size:
            conn.execute("VACUUM")
    except sqlite3.Error as exc:
        logger.warning("VACUUM after conversion skipped: %s", exc)

    conn.close()
    return ConvertResult(
        "converted",
        0,
        f"converted {rows_converted} dense embeddings to base64 float32",
        backup_path=str(backup_path),
        rows_converted=rows_converted,
    )
