"""Read-only probe of the on-disk store format.

``probe_store(data_dir)`` answers one question without touching anything: would
this build open the stores in ``data_dir`` as they are (``current``), after a
pending schema or vector-format step (``old``), not at all because they were
written by a newer build (``newer``), or can it not tell (``unknown``).

The probe never opens a store class, never calls ``_db.connect`` and never runs
a PRAGMA that writes. Each database is opened with
``file:<path>?mode=ro&immutable=1``, which creates no ``-wal``/``-shm``
sidecars. ``immutable=1`` also ignores any write-ahead log, so a database with a
non-empty ``-wal`` (or a hot ``-journal``) is reported ``unknown`` instead of
being read as if its newest content did not exist.

Exit codes, as used by ``taosmd migrate --probe``:

    0   current   every database is at the format this build writes
    10  old       a schema step is pending, or the vector store still holds
                  JSON vectors (a supported state, served as is; converting
                  is optional and never happens on open)
    11  unknown   unreadable, live writer or uncheckpointed WAL, unparseable
                  marker, or rows that contradict the marker
    12  newer     a ``user_version`` or marker key this build does not know

A missing or empty data dir is ``current`` (a first install: each database is
"not created yet"). A database whose only pending step is the version stamp
(schema already complete) is ``current`` and says "stamp only".

When databases disagree the worst wins: newer, then unknown, then old.
"""

from __future__ import annotations

import sqlite3
import urllib.parse
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from . import migrations

__all__ = [
    "ProbeState",
    "DbProbe",
    "ProbeReport",
    "WalPresent",
    "EXIT_CODES",
    "KNOWN_MARKER_KEYS",
    "open_readonly",
    "probe_store",
]


class ProbeState(str, Enum):
    CURRENT = "current"
    OLD = "old"
    UNKNOWN = "unknown"
    NEWER = "newer"
    # Per-database only; never the overall state.
    ABSENT = "absent"
    UNVERSIONED = "unversioned"


EXIT_CODES: dict[ProbeState, int] = {
    ProbeState.CURRENT: 0,
    ProbeState.OLD: 10,
    ProbeState.UNKNOWN: 11,
    ProbeState.NEWER: 12,
}

# Worst first. A newer store must not be touched by this build at all, so it
# outranks an unreadable one.
_SEVERITY = (ProbeState.NEWER, ProbeState.UNKNOWN, ProbeState.OLD, ProbeState.CURRENT)

#: Keys of the vector store's ``store_meta.mode`` marker that this build writes
#: (``VectorMemory._store_mode_signature``). A test pins this to the signature.
KNOWN_MARKER_KEYS = frozenset(
    {"late_interaction", "binary_quant", "dense_format", "embedder"}
)
_KNOWN_DENSE_FORMATS = frozenset({"f32", "none"})

_VECTOR_DB = "vector_memory"
_WAL_REASON = "live writer or uncheckpointed WAL"
_JSON_NOTE = (
    "{what}; supported as is, not an error (an explicit conversion is optional "
    "and is not run on open)"
)
_NOT_CREATED = "not created yet"


class WalPresent(Exception):
    """A non-empty ``-wal`` or ``-journal`` sits next to the database."""


@dataclass
class DbProbe:
    db: str
    path: str
    state: ProbeState
    user_version: Optional[int] = None
    latest: Optional[int] = None
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


@dataclass
class ProbeReport:
    data_dir: str
    state: ProbeState
    databases: list[DbProbe]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.state]

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "state": self.state.value,
            "exit_code": self.exit_code,
            "databases": [d.to_dict() for d in self.databases],
        }


def _nonempty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _live_sidecar(path: Path) -> Optional[str]:
    for suffix in ("-wal", "-journal"):
        if _nonempty(path.with_name(path.name + suffix)):
            return suffix
    return None


def open_readonly(path: Union[str, Path]) -> sqlite3.Connection:
    """Open ``path`` read-only and immutable, creating no sidecar files.

    Raises :class:`WalPresent` when a non-empty ``-wal`` or ``-journal`` exists,
    because an immutable open would silently ignore it.
    """
    p = Path(path).resolve()
    if _live_sidecar(p):
        raise WalPresent(_WAL_REASON)
    uri = "file:" + urllib.parse.quote(str(p)) + "?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _probe_vector_rows(conn: sqlite3.Connection, user_version: int) -> tuple[ProbeState, str, dict]:
    """Classify the vector store by its marker and its actual row encoding."""
    detail: dict[str, Any] = {}
    marker = None
    if _table_exists(conn, "store_meta"):
        row = conn.execute("SELECT value FROM store_meta WHERE key='mode'").fetchone()
        marker = row[0] if row else None
    total = json_rows = 0
    if _table_exists(conn, "vector_memory"):
        total, json_rows = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(embedding LIKE '[%'), 0) FROM vector_memory"
        ).fetchone()
    f32_rows = total - json_rows
    detail.update(marker=marker, rows=total, json_rows=json_rows, f32_rows=f32_rows)

    if marker is None:
        if total == 0:
            return ProbeState.CURRENT, "fresh store, no marker and no rows", detail
        if f32_rows:
            return ProbeState.UNKNOWN, "rows without a marker include non-JSON embeddings", detail
        return ProbeState.OLD, _JSON_NOTE.format(what="legacy JSON vectors with no marker"), detail

    parsed: dict[str, str] = {}
    for part in marker.split(";"):
        if not part:
            continue
        if "=" not in part:
            return ProbeState.UNKNOWN, f"unparseable marker {marker!r}", detail
        k, v = part.split("=", 1)
        parsed[k] = v
    if not parsed:
        return ProbeState.UNKNOWN, f"unparseable marker {marker!r}", detail

    unknown_keys = sorted(set(parsed) - KNOWN_MARKER_KEYS)
    if unknown_keys:
        return ProbeState.NEWER, f"marker keys this build does not know: {', '.join(unknown_keys)}", detail

    fmt = parsed.get("dense_format")
    if fmt is None:
        if parsed.get("late_interaction") == "1" or parsed.get("binary_quant") == "1":
            # A 0.4.x opt-in binary or late-interaction store: master opens it
            # as is and leaves the marker alone; nothing converts it.
            return ProbeState.CURRENT, (
                "legacy marker on a late-interaction or binary store; opens as is"
            ), detail
        if f32_rows:
            return ProbeState.UNKNOWN, "legacy marker but non-JSON embeddings present", detail
        return ProbeState.OLD, _JSON_NOTE.format(
            what="legacy marker without dense_format (JSON vectors)"
        ), detail
    if fmt not in _KNOWN_DENSE_FORMATS:
        return ProbeState.NEWER, f"dense_format={fmt!r} is not known to this build", detail
    if fmt == "f32" and json_rows:
        return ProbeState.UNKNOWN, (
            f"marker says dense_format=f32 but {json_rows} of {total} rows are JSON"
        ), detail
    return ProbeState.CURRENT, "", detail


def _probe_one(db: str, path: Path) -> DbProbe:
    latest = migrations.latest_version(db)
    out = DbProbe(db=db, path=str(path), state=ProbeState.UNKNOWN, latest=latest)
    if not path.exists():
        out.state = ProbeState.ABSENT
        out.reason = _NOT_CREATED
        return out
    try:
        conn = open_readonly(path)
    except WalPresent as exc:
        out.reason = str(exc)
        return out
    except sqlite3.Error as exc:
        out.reason = f"cannot open: {exc}"
        return out
    try:
        try:
            uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
            out.user_version = uv
            if uv > latest:
                out.state = ProbeState.NEWER
                out.reason = f"user_version {uv} is above the {latest} this build knows"
                return out
            state, reason = ProbeState.CURRENT, ""
            if uv < latest:
                st = migrations.status(conn, db)
                if st["pending"]:
                    state, reason = ProbeState.OLD, f"user_version {uv} is below {latest}"
                else:
                    # Same wording as `migrate --status`: only the stamp is
                    # missing, nothing is converted (a2a-receipts.db is never
                    # stamped by its own store).
                    state = ProbeState.CURRENT
                    reason = f"stamp only, schema already at {st['detected_baseline']}"
                    out.detail["stamp_only"] = True
            if db == _VECTOR_DB:
                vstate, vreason, detail = _probe_vector_rows(conn, uv)
                out.detail.update(detail)
                if vstate in (ProbeState.NEWER, ProbeState.UNKNOWN):
                    state, reason = vstate, vreason
                elif vstate is ProbeState.OLD:
                    state = ProbeState.OLD
                    reason = "; ".join(r for r in (reason, vreason) if r)
        except sqlite3.Error as exc:
            out.reason = f"unreadable: {exc}"
            return out
    finally:
        conn.close()
    out.state, out.reason = state, reason
    return out


def probe_store(data_dir: Union[str, Path]) -> ProbeReport:
    """Probe every database in ``data_dir`` without modifying anything."""
    root = Path(data_dir)
    if not root.exists():
        dbs = [
            DbProbe(db=db, path=str(root / migrations.DB_FILES[db]),
                    state=ProbeState.ABSENT, latest=migrations.latest_version(db),
                    reason=_NOT_CREATED)
            for db in sorted(migrations.DB_FILES)
        ]
        return ProbeReport(data_dir=str(root), state=ProbeState.CURRENT, databases=dbs)
    if not root.is_dir():
        return ProbeReport(
            data_dir=str(root),
            state=ProbeState.UNKNOWN,
            databases=[DbProbe(
                db="(data dir)", path=str(root), state=ProbeState.UNKNOWN,
                reason="data dir is not a directory",
            )],
        )
    dbs = [_probe_one(db, root / migrations.DB_FILES[db]) for db in sorted(migrations.DB_FILES)]
    known_files = set(migrations.DB_FILES.values())
    for p in sorted(root.glob("*.db")):
        if p.name in known_files or not p.is_file():
            continue
        dbs.append(DbProbe(
            db=p.stem, path=str(p), state=ProbeState.UNVERSIONED,
            reason="not tracked by the migration registry",
        ))
    present = {d.state for d in dbs}
    overall = next((s for s in _SEVERITY if s in present), ProbeState.CURRENT)
    return ProbeReport(data_dir=str(root), state=overall, databases=dbs)
