"""Guarded conversion of a vector store's dense embeddings, JSON -> base64 float32.

``taosmd migrate --convert-vectors`` is the only supported way to convert a
store. After it runs, a release that only knows the JSON format can no longer
read the store, so the conversion is built to be unable to lose or corrupt a
memory:

1. detect the store format (already converted: nothing to do; unknown or newer:
   refuse),
2. take an exclusive lock so no other process is using the store,
3. check free space for the backup and for the conversion before writing,
4. take a backup with the sqlite backup API and fsync it,
5. verify the backup (integrity, row count, content hash),
6. convert inside ONE transaction, streaming in batches,
7. before COMMIT, re-read every row from the backup and check the new float32
   values, the untouched columns and a recall probe; any failure rolls back,
8. COMMIT, run an integrity check, and keep the backup.

The module works on the sqlite file directly and never opens the store through
``VectorMemory``. The format detection in ``_detect`` is deliberately small and
private; it can delegate to ``taosmd.store_probe`` once that is available.

Exit codes (``ConvertResult.exit_code``): 0 converted or already current;
2 refused, store untouched; 3 unknown or newer format, store untouched;
4 failed and restored (store identical to before); 1 unexpected error (the
message names the backup path when one exists).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import random
import shutil
import sqlite3
import stat
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_UNKNOWN_FORMAT = 3
EXIT_RESTORED = 4

#: Rows read and written per step. Peak memory scales with this, not the store.
_BATCH = 1000
#: A store with more than this share of unreadable rows is refused outright.
_MAX_BAD_FRACTION = 0.01
#: Stored vectors used as queries in the recall check, and how many hits compared.
_RECALL_QUERIES = 20
_RECALL_K = 10
_RECALL_TOL = 1e-6
#: Extra space asked for on top of the backup size.
_BACKUP_MARGIN = 1.10

_TABLE = "vector_memory"
_HASH_COLUMNS = ("id", "embedding", "text", "metadata_json", "created_at", "valid_to")
_NON_EMBEDDING_COLUMNS = ("id", "text", "metadata_json", "created_at", "valid_to")
_MARKER_KEYS = {"late_interaction", "binary_quant", "dense_format", "embedder"}


@dataclass
class ConvertResult:
    status: str  # converted | already_current | refused | unknown_format | restored | error
    exit_code: int
    message: str
    backup_path: str | None = None
    rows_total: int = 0
    rows_converted: int = 0
    dimension: int | None = None
    #: (row id, reason) for rows left byte-identical because they are unreadable.
    malformed: list[tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "message": self.message,
            "backup_path": self.backup_path,
            "rows_total": self.rows_total,
            "rows_converted": self.rows_converted,
            "dimension": self.dimension,
            "malformed": [{"id": i, "reason": r} for i, r in self.malformed],
        }


class _Stop(Exception):
    """Internal: stop the run with a finished result."""

    def __init__(self, result: ConvertResult):
        super().__init__(result.message)
        self.result = result


# ----------------------------------------------------------------------
# row helpers
# ----------------------------------------------------------------------

def _encode_row(arr: np.ndarray) -> str:
    """Encode one vector exactly as ``VectorMemory.add`` does for a dense store."""
    return base64.b64encode(np.asarray(arr, dtype="<f4").tobytes()).decode("ascii")


def _reference_encode(arr: np.ndarray) -> str:
    """Independent copy of the encoding, used only by the post-checks."""
    return base64.b64encode(np.asarray(arr, dtype="<f4").tobytes()).decode("ascii")


def _parse_json_vector(text: str) -> tuple[np.ndarray | None, str]:
    """Parse a JSON embedding into float32. Returns (array, "") or (None, reason)."""
    try:
        vec = json.loads(text)
    except (ValueError, RecursionError):
        return None, "invalid JSON"
    if not vec:
        return None, "empty list"
    for x in vec:
        if type(x) not in (int, float):
            return None, "non-numeric element"
    try:
        arr = np.asarray(vec, dtype="<f4")
    except (OverflowError, ValueError, TypeError):
        return None, "value does not fit float32"
    if not np.isfinite(arr).all():
        return None, "non-finite value"
    return arr, ""


def _classify(emb: object) -> tuple[str, np.ndarray | int | str]:
    """Classify a stored embedding.

    ("json", float32 array) for a readable JSON vector, ("f32", length) for an
    already converted row, ("bad", reason) for anything else.
    """
    if not isinstance(emb, str):
        return "bad", "embedding is not text"
    if emb.lstrip().startswith("["):
        arr, reason = _parse_json_vector(emb)
        if arr is None:
            return "bad", reason
        return "json", arr
    try:
        raw = base64.b64decode(emb, validate=True)
    except (binascii.Error, ValueError):
        return "bad", "unrecognised encoding"
    if not raw or len(raw) % 4:
        return "bad", "unrecognised encoding"
    return "f32", len(raw) // 4


def _decode_f32(emb: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(emb, validate=True), dtype="<f4")


def _hash_rows(conn: sqlite3.Connection, columns: tuple[str, ...]) -> str:
    """sha256 over the given columns of every row, in id order, streamed."""
    h = hashlib.sha256()
    cols = ", ".join(columns)
    last = None
    while True:
        if last is None:
            cur = conn.execute(f"SELECT {cols} FROM {_TABLE} ORDER BY id LIMIT ?", (_BATCH,))
        else:
            cur = conn.execute(
                f"SELECT {cols} FROM {_TABLE} WHERE id > ? ORDER BY id LIMIT ?", (last, _BATCH)
            )
        rows = cur.fetchall()
        if not rows:
            break
        for row in rows:
            h.update(repr(tuple(row)).encode("utf-8", "surrogatepass"))
            h.update(b"\n")
        last = rows[-1][0]
    return h.hexdigest()


def _count(conn: sqlite3.Connection) -> int:
    return int(conn.execute(f"SELECT count(*) FROM {_TABLE}").fetchone()[0])


def _batches(conn: sqlite3.Connection, columns: str = "id, embedding"):
    """Yield lists of rows in id order, each at most _BATCH long."""
    last = None
    while True:
        if last is None:
            rows = conn.execute(
                f"SELECT {columns} FROM {_TABLE} ORDER BY id LIMIT ?", (_BATCH,)
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {columns} FROM {_TABLE} WHERE id > ? ORDER BY id LIMIT ?",
                (last, _BATCH),
            ).fetchall()
        if not rows:
            return
        yield rows
        last = rows[-1][0]


# ----------------------------------------------------------------------
# detection (private; can delegate to taosmd.store_probe later)
# ----------------------------------------------------------------------

@dataclass
class _Detected:
    marker: str | None
    marker_has_f32: bool
    new_marker: str | None  # marker to write; None when the marker is already right


def _parse_marker(value: str) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for part in value.split(";"):
        if not part:
            continue
        if "=" not in part:
            return None
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _detect(conn: sqlite3.Connection) -> _Detected:
    """Read the schema, version and marker. Raises _Stop for refuse/unknown."""
    from . import migrations  # noqa: PLC0415

    def stop(status: str, code: int, msg: str) -> _Stop:
        return _Stop(ConvertResult(status=status, exit_code=code, message=msg))

    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    except sqlite3.DatabaseError as e:
        raise stop("unknown_format", EXIT_UNKNOWN_FORMAT, f"not a readable sqlite store: {e}") from e
    if _TABLE not in tables:
        raise stop("unknown_format", EXIT_UNKNOWN_FORMAT, f"no {_TABLE} table: not a vector store")

    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({_TABLE})")}
    needed = set(_HASH_COLUMNS) | {"metadata_json"}
    missing = needed - cols
    if missing == {"valid_to"}:
        raise stop(
            "refused", EXIT_REFUSED,
            "the store schema is older than this release (no valid_to column); "
            "run `taosmd migrate` first, then retry",
        )
    if missing:
        raise stop(
            "unknown_format", EXIT_UNKNOWN_FORMAT,
            f"vector_memory is missing columns this release expects: {sorted(missing)}",
        )

    user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    latest = migrations.latest_version("vector_memory")
    if user_version > latest:
        raise stop(
            "unknown_format", EXIT_UNKNOWN_FORMAT,
            f"the store schema version {user_version} is newer than this build ({latest})",
        )

    marker = None
    if "store_meta" in tables:
        row = conn.execute("SELECT value FROM store_meta WHERE key = 'mode'").fetchone()
        marker = row[0] if row else None

    if marker is None:
        return _Detected(None, False, "late_interaction=0;binary_quant=0;dense_format=f32")

    rec = _parse_marker(marker) if isinstance(marker, str) else None
    if rec is None:
        raise stop("unknown_format", EXIT_UNKNOWN_FORMAT, f"unparseable store marker: {marker!r}")
    unknown = set(rec) - _MARKER_KEYS
    if unknown:
        raise stop(
            "unknown_format", EXIT_UNKNOWN_FORMAT,
            f"the store marker has keys this build does not know: {sorted(unknown)}",
        )
    if rec.get("late_interaction", "0") != "0" or rec.get("binary_quant", "0") != "0":
        raise stop(
            "unknown_format", EXIT_UNKNOWN_FORMAT,
            "this store is not a dense float store (late-interaction or binary-quant); "
            "there is nothing to convert",
        )
    fmt = rec.get("dense_format")
    if fmt is not None and fmt != "f32":
        raise stop(
            "unknown_format", EXIT_UNKNOWN_FORMAT,
            f"the store records dense_format={fmt!r}, which this build does not know",
        )
    if fmt == "f32":
        return _Detected(marker, True, None)
    new = "late_interaction=0;binary_quant=0;dense_format=f32"
    if "embedder" in rec:
        new += f";embedder={rec['embedder']}"
    return _Detected(marker, False, new)


# ----------------------------------------------------------------------
# prescan
# ----------------------------------------------------------------------

@dataclass
class _Scan:
    total: int = 0
    lengths: Counter = field(default_factory=Counter)  # structurally valid rows
    json_lengths: Counter = field(default_factory=Counter)
    reservoir: list[tuple[int, np.ndarray]] = field(default_factory=list)

    @property
    def dimension(self) -> int | None:
        if not self.lengths:
            return None
        # Most common length; ties go to the smaller dimension for determinism.
        return sorted(self.lengths.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _prescan(conn: sqlite3.Connection) -> _Scan:
    scan = _Scan()
    rng = random.Random(0)
    seen = 0
    cap = _RECALL_QUERIES * 3
    for rows in _batches(conn):
        for rid, emb in rows:
            scan.total += 1
            kind, val = _classify(emb)
            if kind == "bad":
                continue
            vec = None
            if kind == "json":
                n = len(val)
                scan.json_lengths[n] += 1
                vec = val
            else:
                n = val
            scan.lengths[n] += 1
            seen += 1
            # Reservoir sample of vectors (any length; filtered to the store
            # dimension afterwards) used as recall queries.
            if len(scan.reservoir) < cap:
                scan.reservoir.append((rid, vec if vec is not None else _decode_f32(emb)))
            else:
                j = rng.randrange(seen)
                if j < cap:
                    scan.reservoir[j] = (rid, vec if vec is not None else _decode_f32(emb))
    return scan


# ----------------------------------------------------------------------
# space, locking, backup
# ----------------------------------------------------------------------

def _existing_parent(p: Path) -> Path:
    p = p.resolve()
    while not p.exists():
        p = p.parent
    return p


def _same_filesystem(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def _free_space_check(db: Path, backup_dir: Path) -> None:
    db_size = db.stat().st_size
    wal = Path(str(db) + "-wal")
    wal_size = wal.stat().st_size if wal.exists() else 0
    need_backup = int((db_size + wal_size) * _BACKUP_MARGIN)
    need_store = db_size  # WAL growth while converting
    store_loc = _existing_parent(db.parent)
    backup_loc = _existing_parent(backup_dir)
    if _same_filesystem(store_loc, backup_loc):
        free = shutil.disk_usage(store_loc).free
        need = need_backup + need_store
        if free < need:
            raise _Stop(ConvertResult(
                status="refused", exit_code=EXIT_REFUSED,
                message=f"not enough free disk space: need about {need} bytes for the "
                        f"backup and the conversion, {free} free. Nothing was changed.",
            ))
        return
    free_b = shutil.disk_usage(backup_loc).free
    if free_b < need_backup:
        raise _Stop(ConvertResult(
            status="refused", exit_code=EXIT_REFUSED,
            message=f"not enough free space for the backup: need about {need_backup} "
                    f"bytes in {backup_loc}, {free_b} free. Nothing was changed.",
        ))
    free_s = shutil.disk_usage(store_loc).free
    if free_s < need_store:
        raise _Stop(ConvertResult(
            status="refused", exit_code=EXIT_REFUSED,
            message=f"not enough free space for the conversion: need about {need_store} "
                    f"bytes in {store_loc}, {free_s} free. Nothing was changed.",
        ))


def _fsync_path(path: Path, directory: bool = False) -> None:
    fd = os.open(str(path), os.O_RDONLY if directory else os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)


def _take_backup(
    conn: sqlite3.Connection, db: Path, backup_dir: Path, live_all_hash: str, live_count: int,
) -> Path:
    """Write and verify a backup. Returns its final path; raises _Stop on failure."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    # Partial files can only come from an earlier interrupted run of this
    # command (we hold the store lock), so they are safe to clear.
    for stale in backup_dir.glob("*.pre-f32.db.partial"):
        try:
            stale.unlink()
        except OSError:
            pass
        for ext in ("-wal", "-shm", "-journal"):
            try:
                Path(str(stale) + ext).unlink()
            except OSError:
                pass

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final = backup_dir / f"{db.stem}.{ts}.pre-f32.db"
    partial = Path(str(final) + ".partial")

    def drop_partial() -> None:
        for p in (partial, Path(str(partial) + "-wal"), Path(str(partial) + "-shm"),
                  Path(str(partial) + "-journal")):
            try:
                p.unlink()
            except OSError:
                pass

    bconn = None
    try:
        bconn = sqlite3.connect(str(partial), isolation_level=None)
        conn.backup(bconn)
        bconn.execute("PRAGMA journal_mode=DELETE")
        bconn.close()
        bconn = None
        try:
            os.chmod(partial, stat.S_IMODE(db.stat().st_mode))
        except OSError:
            pass
        _fsync_path(partial)

        v = _open_ro(partial)
        try:
            ok = v.execute("PRAGMA integrity_check").fetchall()
            if ok != [("ok",)]:
                raise _Stop(ConvertResult(
                    status="refused", exit_code=EXIT_REFUSED,
                    message="the backup failed its integrity check; it was removed and "
                            "the store was not changed",
                ))
            if _count(v) != live_count:
                raise _Stop(ConvertResult(
                    status="refused", exit_code=EXIT_REFUSED,
                    message="the backup row count does not match the store; it was "
                            "removed and the store was not changed",
                ))
            if _hash_rows(v, _HASH_COLUMNS) != live_all_hash:
                raise _Stop(ConvertResult(
                    status="refused", exit_code=EXIT_REFUSED,
                    message="the backup content hash does not match the store; it was "
                            "removed and the store was not changed",
                ))
        finally:
            v.close()
        os.replace(partial, final)
        try:
            _fsync_path(backup_dir, directory=True)
        except OSError:  # directories cannot be fsynced on every platform
            pass
    except BaseException:
        if bconn is not None:
            try:
                bconn.close()
            except Exception:
                pass
        drop_partial()
        raise
    return final


# ----------------------------------------------------------------------
# conversion and post-checks
# ----------------------------------------------------------------------

def _convert_rows(conn: sqlite3.Connection, dim: int, result: ConvertResult) -> None:
    for rows in _batches(conn):
        updates = []
        for rid, emb in rows:
            kind, val = _classify(emb)
            if kind == "bad":
                result.malformed.append((rid, val))
            elif kind == "json":
                if len(val) != dim:
                    result.malformed.append((rid, f"wrong length {len(val)} (store dimension {dim})"))
                else:
                    updates.append((_encode_row(val), rid))
            else:  # already f32
                if val != dim:
                    result.malformed.append((rid, f"wrong length {val} (store dimension {dim})"))
        if updates:
            conn.executemany(f"UPDATE {_TABLE} SET embedding = ? WHERE id = ?", updates)
            result.rows_converted += len(updates)


def _unit(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, np.float32(1e-12))


class _TopK:
    def __init__(self, queries: np.ndarray, k: int):
        self.q = _unit(queries)
        self.k = k
        nq = self.q.shape[0]
        self.scores = np.full((nq, 0), -np.inf, dtype=np.float32)
        self.ids = np.zeros((nq, 0), dtype=np.int64)

    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        if len(ids) == 0:
            return
        s = (_unit(vecs) @ self.q.T).T.astype(np.float32)  # (Q, B)
        scores = np.concatenate([self.scores, s], axis=1)
        idm = np.concatenate([self.ids, np.broadcast_to(ids, s.shape)], axis=1)
        ns = np.empty((scores.shape[0], min(self.k, scores.shape[1])), dtype=np.float32)
        ni = np.empty(ns.shape, dtype=np.int64)
        for r in range(scores.shape[0]):
            order = np.lexsort((idm[r], -scores[r]))[: ns.shape[1]]
            ns[r] = scores[r][order]
            ni[r] = idm[r][order]
        self.scores, self.ids = ns, ni


class _PostCheckFailed(Exception):
    pass


def _post_checks(
    live: sqlite3.Connection,
    backup: sqlite3.Connection,
    dim: int | None,
    queries: np.ndarray | None,
    expected_count: int,
    backup_plain_hash: str,
    expected_marker: str | None,
    marker_should_change: bool,
) -> None:
    """Raise _PostCheckFailed unless the live store (inside the open transaction)
    holds exactly the backup's content with every JSON vector now float32."""
    if _count(live) != expected_count:
        raise _PostCheckFailed("row count changed")

    topk_b = topk_l = None
    if queries is not None and len(queries):
        topk_b = _TopK(queries, _RECALL_K)
        topk_l = _TopK(queries, _RECALL_K)

    for b_rows in _batches(backup):
        first, last = b_rows[0][0], b_rows[-1][0]
        l_rows = live.execute(
            f"SELECT id, embedding FROM {_TABLE} WHERE id >= ? AND id <= ? ORDER BY id",
            (first, last),
        ).fetchall()
        if [r[0] for r in l_rows] != [r[0] for r in b_rows]:
            raise _PostCheckFailed("row ids differ from the backup")
        ids, bvecs, lvecs = [], [], []
        for (rid, b_emb), (_, l_emb) in zip(b_rows, l_rows):
            kind, val = _classify(b_emb)
            if kind == "json" and len(val) == dim:
                want = _reference_encode(val)
                if l_emb != want:
                    raise _PostCheckFailed(f"row {rid}: float32 value differs from the original")
                ids.append(rid)
                bvecs.append(val)
                lvecs.append(_decode_f32(l_emb))
            else:
                if l_emb != b_emb:
                    raise _PostCheckFailed(f"row {rid}: an untouched row changed")
                if kind == "f32" and val == dim:
                    ids.append(rid)
                    bvecs.append(_decode_f32(b_emb))
                    lvecs.append(_decode_f32(l_emb))
        if topk_b is not None and ids:
            ida = np.asarray(ids, dtype=np.int64)
            topk_b.add(ida, np.vstack(bvecs))
            topk_l.add(ida, np.vstack(lvecs))

    if _hash_rows(live, _NON_EMBEDDING_COLUMNS) != backup_plain_hash:
        raise _PostCheckFailed("non-embedding columns differ from the backup")

    if topk_b is not None:
        if not np.array_equal(topk_b.ids, topk_l.ids):
            raise _PostCheckFailed("recall check: top-10 ids differ")
        if topk_b.scores.size and float(np.max(np.abs(topk_b.scores - topk_l.scores))) > _RECALL_TOL:
            raise _PostCheckFailed("recall check: scores differ by more than 1e-6")

    if marker_should_change:
        row = live.execute("SELECT value FROM store_meta WHERE key = 'mode'").fetchone()
        if not row or row[0] != expected_marker:
            raise _PostCheckFailed("store marker was not written")


def _post_commit_check(db: Path, conn: sqlite3.Connection) -> str | None:
    """Return a failure description, or None when the committed store is sound."""
    res = conn.execute("PRAGMA integrity_check").fetchall()
    if res != [("ok",)]:
        return f"integrity_check reported {res[:3]}"
    return None


def _maybe_vacuum(conn: sqlite3.Connection, db: Path) -> None:
    try:
        size = db.stat().st_size
        if shutil.disk_usage(_existing_parent(db.parent)).free >= 2 * size:
            conn.execute("VACUUM")
    except Exception as e:  # reclaiming space is optional
        logger.warning("VACUUM after conversion failed (non-fatal): %s", e)


def _peek_without_lock(db: Path) -> ConvertResult | None:
    """Best-effort read-only look at a store somebody else is using.

    Lets a re-run on an already converted store (or one in a format this build
    does not know) answer 0 or 3 even while `taosmd serve` has it open, instead
    of a misleading "in use". Returns None when it cannot tell, and the caller
    then refuses with exit 2. Never writes.
    """
    c = None
    try:
        c = _open_ro(db)
        c.execute("PRAGMA busy_timeout=500")
        try:
            det = _detect(c)
        except _Stop as stop:
            return stop.result if stop.result.exit_code == EXIT_UNKNOWN_FORMAT else None
        if not det.marker_has_f32:
            return None
        json_left = c.execute(
            f"SELECT 1 FROM {_TABLE} WHERE embedding LIKE '[%' OR embedding LIKE ' %' LIMIT 1"
        ).fetchone()
        if json_left is None:
            return ConvertResult(
                status="already_current", exit_code=EXIT_OK,
                message="already current: nothing to convert (the store is in use by another process)",
            )
        return None
    except Exception:  # noqa: BLE001
        return None
    finally:
        if c is not None:
            try:
                c.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------

def convert_dense_store(db_path, *, backup_dir=None) -> ConvertResult:
    """Convert one vector store from JSON text to base64 float32, with a backup.

    Never raises for expected conditions; inspect ``exit_code``. See the module
    docstring for the contract.
    """
    db = Path(db_path)
    bdir = Path(backup_dir) if backup_dir is not None else db.parent / "backups"
    if not db.is_file():
        return ConvertResult(
            status="refused", exit_code=EXIT_REFUSED,
            message=f"no vector store at {db}; nothing was changed",
        )

    conn: sqlite3.Connection | None = None
    bconn: sqlite3.Connection | None = None
    backup_path: Path | None = None
    in_txn = False
    committed = False
    try:
        # 2. exclusive access. EXCLUSIVE locking mode also fails while another
        # connection (for example an idle `taosmd serve`) has the store open.
        conn = sqlite3.connect(str(db), timeout=0.5, isolation_level=None)
        try:
            conn.execute("PRAGMA locking_mode=EXCLUSIVE")
            # The first read takes the exclusive lock and, in this mode, keeps
            # it until the connection closes, so the detection, the backup and
            # the conversion all see one unchanging store.
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            # Rollback-journal stores only block writers once the write
            # reservation is taken, so take and release it here, inside the
            # lock error mapping, rather than failing later with a backup made.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
        except sqlite3.DatabaseError as e:
            low = str(e).lower()
            if "locked" in low or "busy" in low:
                peeked = _peek_without_lock(db)
                if peeked is not None:
                    return peeked
                raise _Stop(ConvertResult(
                    status="refused", exit_code=EXIT_REFUSED,
                    message="the store is in use by another process (for example "
                            "`taosmd serve`); stop it and retry. Nothing was changed.",
                )) from e
            raise _Stop(ConvertResult(
                status="unknown_format", exit_code=EXIT_UNKNOWN_FORMAT,
                message=f"not a readable sqlite store: {e}",
            )) from e

        # 1. format detection
        det = _detect(conn)
        scan = _prescan(conn)
        dim = scan.dimension
        convertible = scan.json_lengths.get(dim, 0) if dim is not None else 0
        marker_ok = det.marker_has_f32 or (det.marker is None and scan.total == 0)
        if marker_ok and convertible == 0:
            bad = scan.total - (scan.lengths.get(dim, 0) if dim is not None else 0)
            extra = f" ({bad} unreadable row(s) left as they are)" if bad else ""
            raise _Stop(ConvertResult(
                status="already_current", exit_code=EXIT_OK,
                message=f"already current: nothing to convert{extra}",
                rows_total=scan.total, dimension=dim,
            ))
        bad_total = scan.total - (scan.lengths.get(dim, 0) if dim is not None else 0)
        if scan.total and bad_total > _MAX_BAD_FRACTION * scan.total:
            raise _Stop(ConvertResult(
                status="refused", exit_code=EXIT_REFUSED,
                message=f"{bad_total} of {scan.total} rows are unreadable (more than 1%); "
                        "refusing to convert. Nothing was changed.",
                rows_total=scan.total, dimension=dim,
            ))

        # 3. free space
        _free_space_check(db, bdir)

        # 4 + 5. backup and verification
        live_count = scan.total
        live_all = _hash_rows(conn, _HASH_COLUMNS)
        backup_path = _take_backup(conn, db, bdir, live_all, live_count)
        bconn = _open_ro(backup_path)
        backup_plain = _hash_rows(bconn, _NON_EMBEDDING_COLUMNS)

        # 6. convert, one transaction
        conn.execute("BEGIN IMMEDIATE")
        in_txn = True
        result = ConvertResult(
            status="converted", exit_code=EXIT_OK, message="", backup_path=str(backup_path),
            rows_total=scan.total, dimension=dim,
        )
        if dim is not None:
            _convert_rows(conn, dim, result)
        if det.new_marker is not None:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', ?)",
                (det.new_marker,),
            )

        # 7. post-checks before COMMIT
        queries = None
        if dim is not None:
            qv = [v for _, v in scan.reservoir if len(v) == dim][:_RECALL_QUERIES]
            if qv:
                queries = np.vstack(qv)
        try:
            _post_checks(
                conn, bconn, dim, queries, live_count, backup_plain,
                det.new_marker, det.new_marker is not None,
            )
        except _PostCheckFailed as e:
            conn.execute("ROLLBACK")
            in_txn = False
            raise _Stop(ConvertResult(
                status="restored", exit_code=EXIT_RESTORED,
                message=f"post-conversion check failed ({e}); the store was rolled back "
                        f"and is unchanged. Backup kept at {backup_path}.",
                backup_path=str(backup_path), rows_total=scan.total, dimension=dim,
            )) from e

        # 8. commit, then verify the committed file
        conn.execute("COMMIT")
        in_txn = False
        committed = True
        problem = _post_commit_check(db, conn)
        if problem is not None:
            bconn.close()
            bconn = None
            restore = sqlite3.connect(str(backup_path), isolation_level=None)
            try:
                restore.backup(conn)
            finally:
                restore.close()
            raise _Stop(ConvertResult(
                status="restored", exit_code=EXIT_RESTORED,
                message=f"the committed store failed verification ({problem}); it was "
                        f"restored from the backup at {backup_path}.",
                backup_path=str(backup_path), rows_total=scan.total, dimension=dim,
            ))
        _maybe_vacuum(conn, db)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            pass

        n_bad = len(result.malformed)
        result.message = (
            f"converted {result.rows_converted} of {scan.total} rows to float32 "
            f"(dimension {dim}); backup kept at {backup_path}"
        )
        if n_bad:
            result.message += f"; {n_bad} unreadable row(s) left as they were"
        return result

    except _Stop as stop:
        if in_txn and conn is not None:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        r = stop.result
        if r.backup_path is None and backup_path is not None and r.status == "restored":
            r.backup_path = str(backup_path)
        return r
    except Exception as e:  # noqa: BLE001
        logger.exception("vector store conversion failed unexpectedly")
        if in_txn and conn is not None:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        where = f" The backup is at {backup_path}." if backup_path else " No backup had been taken."
        state = (
            " The conversion had been committed."
            if committed
            else " The store was rolled back and is unchanged."
        )
        return ConvertResult(
            status="error", exit_code=EXIT_ERROR,
            message=f"unexpected error: {type(e).__name__}: {e}.{state}{where}",
            backup_path=str(backup_path) if backup_path else None,
        )
    finally:
        for c in (bconn, conn):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
