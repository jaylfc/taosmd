"""Schema migrations for every SQLite database the package owns.

Why this exists
---------------
Before this module there was no migration mechanism at all: no version stamp,
no upgrade path. Schemas were created with ``CREATE TABLE IF NOT EXISTS`` and
evolved by three hand-rolled ``try/except ALTER TABLE`` guards (the archive
``project`` column, the session-catalog taxonomy columns, and the vector-memory
``valid_to`` column). That worked only because every change so far was purely
additive at the table level. The moment a column is added to an *existing*
table, ``CREATE TABLE IF NOT EXISTS`` does nothing on a database that already
exists and every query naming the new column fails at runtime.

The contract
------------
Each logical database has an ordered list of :class:`Migration` steps, numbered
from 1. The database's own ``PRAGMA user_version`` records how far it has been
taken. :func:`migrate` applies only the steps above that number, each inside
its own transaction, stamping the new version atomically with the change.
Running it on an already-current database is a cheap no-op (one PRAGMA read),
so it is safe to call on every store open.

Two rules bind every migration step, and the tests enforce the second:

1. **Transaction rule.** A step must not commit, roll back, or call
   ``executescript`` (which implicitly commits). The runner owns the
   transaction so that a failing step leaves neither a half-applied schema nor
   a bumped version. Use :func:`exec_script` for multi-statement SQL.
2. **Idempotency rule.** Applying a step to a database that already has the
   change must be harmless. Use the ``IF NOT EXISTS`` forms and the
   :func:`add_column` helper.

The zero-version problem
------------------------
Databases already in the field report ``user_version = 0`` while carrying a
fully modern schema, because the old ``CREATE TABLE IF NOT EXISTS`` plus
hand-rolled ALTER path built them without ever stamping a version. Replaying
every step against such a store is the single most likely way to break a live
install.

So each migration carries a ``detect`` probe: a cheap read (usually
``PRAGMA table_info``) answering "is this change already present?". On an
unstamped database the runner walks the list in order, and the longest leading
run of steps that report themselves already present becomes the *baseline*: the
version the database is stamped to without any step being executed. Only the
remainder is applied. A brand-new empty file detects nothing and is built from
step 1; a modern legacy store detects everything and is stamped straight to the
latest version with zero writes to its schema.

Relationship to the ``SCHEMA`` constants
----------------------------------------
Each store module keeps its ``CREATE TABLE IF NOT EXISTS`` schema constant and
still runs it on open, and the step-1 "baseline" migration of each database
runs the same constant. That redundancy is deliberate: adding a whole new
table can stay in the schema constant (a no-op for stores that have it, created
for those that do not), while **any change to an existing table must be a
migration**, because the schema constant cannot deliver it.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, Union

__all__ = [
    "Migration",
    "MigrationResult",
    "REGISTRY",
    "DB_FILES",
    "migrate",
    "migrate_all",
    "status",
    "status_all",
    "latest_version",
    "table_exists",
    "index_exists",
    "has_column",
    "columns",
    "add_column",
    "exec_script",
]

ApplyFn = Callable[[sqlite3.Connection], None]
DetectFn = Callable[[sqlite3.Connection], bool]


@dataclass(frozen=True)
class Migration:
    """One ordered schema step for one database.

    ``version`` is the ``user_version`` the database carries once the step has
    been applied. ``detect`` reports whether the change is already present, and
    is what lets an unstamped legacy database be stamped rather than replayed.
    """

    version: int
    name: str
    apply: ApplyFn
    detect: DetectFn


@dataclass(frozen=True)
class MigrationResult:
    db: str
    from_version: int
    to_version: int
    applied: tuple[str, ...] = ()
    #: True when an unstamped database was recognised as already carrying some
    #: or all of the schema and stamped to that baseline without replaying it.
    stamped_baseline: bool = False


# ----------------------------------------------------------------------
# introspection helpers (usable from detect probes and apply steps)
# ----------------------------------------------------------------------

def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def index_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Column names of ``table``; empty set when the table does not exist."""
    # PRAGMA does not accept bound parameters, and a bad identifier here would
    # be a programming error in a migration, not user input.
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in columns(conn, table)


def add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> bool:
    """``ALTER TABLE ... ADD COLUMN``, skipped when the column is present.

    The idempotent form every column-adding migration should use. Returns True
    when the column was actually added. ``ADD COLUMN`` only appends a column to
    the table definition, it never rewrites or drops rows, so existing data
    survives with the new column at its default (NULL unless ``decl`` says
    otherwise).
    """
    if has_column(conn, table, column):
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    return True


def exec_script(conn: sqlite3.Connection, sql: str) -> None:
    """Run a multi-statement script without breaking the caller's transaction.

    ``sqlite3.Connection.executescript`` issues an implicit COMMIT before it
    runs, which would silently end the transaction the runner opened around a
    migration and defeat the rollback guarantee. This splits the script on
    statement boundaries (``sqlite3.complete_statement`` is literal-aware, so a
    semicolon inside a quoted string does not split) and executes the pieces on
    the current transaction.
    """
    buf = ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if buf.strip() and sqlite3.complete_statement(buf):
            conn.execute(buf)
            buf = ""
    if buf.strip():
        conn.execute(buf)


# ----------------------------------------------------------------------
# per-database migration lists
# ----------------------------------------------------------------------
#
# Adding a migration: append a Migration with the next version number, an
# idempotent ``apply``, and a ``detect`` that reads the schema. Never renumber
# or edit a released step: stores in the field are already stamped past it.

# --- archive-index.db --------------------------------------------------

def _archive_index_baseline(conn: sqlite3.Connection) -> None:
    from taosmd.archive import INDEX_SCHEMA  # noqa: PLC0415  (avoids a cycle)

    exec_script(conn, INDEX_SCHEMA)


def _archive_index_project(conn: sqlite3.Connection) -> None:
    # Subsumes the old back-fill in ArchiveStore.init, which probed with
    # `SELECT project FROM archive_index LIMIT 1` inside a bare try/except.
    add_column(conn, "archive_index", "project", "TEXT")


_ARCHIVE_INDEX: tuple[Migration, ...] = (
    Migration(
        1, "archive_index_baseline", _archive_index_baseline,
        lambda c: table_exists(c, "archive_index") and table_exists(c, "archive_settings"),
    ),
    Migration(
        2, "archive_index_project", _archive_index_project,
        lambda c: has_column(c, "archive_index", "project"),
    ),
)


# --- session-catalog.db ------------------------------------------------

def _session_catalog_baseline(conn: sqlite3.Connection) -> None:
    from taosmd.session_catalog import SCHEMA  # noqa: PLC0415

    exec_script(conn, SCHEMA)


_TAXONOMY_COLUMNS = (
    ("primary_project", "TEXT DEFAULT ''"),
    ("primary_topic", "TEXT DEFAULT ''"),
    ("primary_subtopic", "TEXT DEFAULT ''"),
    ("labels_json", "TEXT DEFAULT '[]'"),
    ("classified_at", "REAL DEFAULT 0"),
)


def _session_catalog_taxonomy(conn: sqlite3.Connection) -> None:
    # Subsumes the loop of five try/except ALTERs plus the deferred index in
    # SessionCatalog.init. The index must come after the columns exist.
    for name, decl in _TAXONOMY_COLUMNS:
        add_column(conn, "sessions", name, decl)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_path "
        "ON sessions(primary_project, primary_topic, primary_subtopic)"
    )


def _session_catalog_agent_name(conn: sqlite3.Connection) -> None:
    add_column(conn, "sessions", "agent_name", "TEXT DEFAULT ''")


_SESSION_CATALOG: tuple[Migration, ...] = (
    Migration(
        1, "session_catalog_baseline", _session_catalog_baseline,
        lambda c: table_exists(c, "sessions"),
    ),
    Migration(
        2, "session_catalog_taxonomy", _session_catalog_taxonomy,
        lambda c: (
            all(has_column(c, "sessions", n) for n, _ in _TAXONOMY_COLUMNS)
            and index_exists(c, "idx_sessions_path")
        ),
    ),
    Migration(
        3, "session_catalog_agent_name", _session_catalog_agent_name,
        lambda c: has_column(c, "sessions", "agent_name"),
    ),
)


# --- vector-memory.db --------------------------------------------------

def _vector_memory_baseline(conn: sqlite3.Connection) -> None:
    from taosmd.vector_memory import SCHEMA  # noqa: PLC0415

    exec_script(conn, SCHEMA)


def _vector_memory_valid_to(conn: sqlite3.Connection) -> None:
    """Subsumes ``VectorMemory._migrate``.

    ``valid_to`` backs the correction-supersede feature. ADD COLUMN appends a
    NULL-defaulted column without rewriting rows, so every existing vector
    survives and stays active (``valid_to IS NULL``). The index is created
    here, after the column, rather than in the schema constant: on a legacy
    table the column only appears via the ALTER above, and a CREATE INDEX in
    the schema constant would fire first, against a column that does not yet
    exist.
    """
    add_column(conn, "vector_memory", "valid_to", "REAL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_vm_valid ON vector_memory(valid_to)"
    )


_VECTOR_MEMORY: tuple[Migration, ...] = (
    Migration(
        1, "vector_memory_baseline", _vector_memory_baseline,
        lambda c: table_exists(c, "vector_memory"),
    ),
    Migration(
        2, "vector_memory_valid_to", _vector_memory_valid_to,
        lambda c: (
            has_column(c, "vector_memory", "valid_to")
            and index_exists(c, "idx_vm_valid")
        ),
    ),
)


# --- knowledge-graph.db ------------------------------------------------

def _knowledge_graph_baseline(conn: sqlite3.Connection) -> None:
    from taosmd.knowledge_graph import SCHEMA  # noqa: PLC0415

    exec_script(conn, SCHEMA)


_KNOWLEDGE_GRAPH: tuple[Migration, ...] = (
    Migration(
        1, "knowledge_graph_baseline", _knowledge_graph_baseline,
        lambda c: table_exists(c, "entities") and table_exists(c, "triples"),
    ),
)


# --- claims.db ---------------------------------------------------------

def _claims_baseline(conn: sqlite3.Connection) -> None:
    from taosmd.claims.store import _SCHEMA  # noqa: PLC0415

    exec_script(conn, _SCHEMA)


_CLAIMS: tuple[Migration, ...] = (
    Migration(
        1, "claims_baseline", _claims_baseline,
        lambda c: table_exists(c, "claims"),
    ),
)


# --- collections.db ----------------------------------------------------

def _collections_baseline(conn: sqlite3.Connection) -> None:
    from taosmd.collections import SCHEMA  # noqa: PLC0415

    exec_script(conn, SCHEMA)


_COLLECTIONS: tuple[Migration, ...] = (
    Migration(
        1, "collections_baseline", _collections_baseline,
        lambda c: (
            table_exists(c, "collections")
            and table_exists(c, "collection_links")
            and table_exists(c, "collection_grants")
            and table_exists(c, "collection_files")
        ),
    ),
)


#: Logical database name -> ordered migration steps.
REGISTRY: dict[str, tuple[Migration, ...]] = {
    "archive_index": _ARCHIVE_INDEX,
    "claims": _CLAIMS,
    "collections": _COLLECTIONS,
    "knowledge_graph": _KNOWLEDGE_GRAPH,
    "session_catalog": _SESSION_CATALOG,
    "vector_memory": _VECTOR_MEMORY,
}

#: Logical database name -> filename inside the data directory.
DB_FILES: dict[str, str] = {
    "archive_index": "archive-index.db",
    "claims": "claims.db",
    "collections": "collections.db",
    "knowledge_graph": "knowledge-graph.db",
    "session_catalog": "session-catalog.db",
    "vector_memory": "vector-memory.db",
}


# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------

def _resolve(db: str, migrations_list: Sequence[Migration] | None) -> tuple[Migration, ...]:
    if migrations_list is not None:
        return tuple(migrations_list)
    try:
        return REGISTRY[db]
    except KeyError:
        raise KeyError(
            f"unknown database {db!r}; known: {', '.join(sorted(REGISTRY))}"
        ) from None


def latest_version(db: str, migrations_list: Sequence[Migration] | None = None) -> int:
    migs = _resolve(db, migrations_list)
    return migs[-1].version if migs else 0


def _read_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _detect_baseline(conn: sqlite3.Connection, migs: Sequence[Migration]) -> int:
    """Highest version whose whole leading run is already present in the schema.

    Stops at the first step that does not detect itself. A later step that is
    coincidentally present is still re-applied, which is safe because every
    step is required to be idempotent, and is the conservative choice: it never
    skips a step that might not have run.
    """
    baseline = 0
    for m in migs:
        try:
            present = bool(m.detect(conn))
        except sqlite3.Error:
            present = False
        if not present:
            break
        baseline = m.version
    return baseline


def migrate(
    conn: sqlite3.Connection,
    db: str,
    *,
    migrations_list: Sequence[Migration] | None = None,
) -> MigrationResult:
    """Bring ``conn`` up to the latest schema version for ``db``.

    Cheap enough to call on every store open: when the database is current this
    is a single ``PRAGMA user_version`` read and nothing else.

    Raises whatever a failing step raises, after rolling that step back. Steps
    that already succeeded keep their committed versions, so a re-run resumes
    from the last good one.
    """
    migs = _resolve(db, migrations_list)
    if not migs:
        return MigrationResult(db=db, from_version=0, to_version=0)

    # A caller-owned open transaction would collide with our BEGIN, and would
    # also be swept into our COMMIT. Flush it before taking control.
    if conn.in_transaction:
        conn.commit()

    current = _read_version(conn)
    target = migs[-1].version
    stamped = False

    if current == 0:
        baseline = _detect_baseline(conn, migs)
        if baseline > 0:
            # The database predates versioning but already carries this much
            # schema. Record that rather than replaying it over live data.
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(f"PRAGMA user_version = {int(baseline)}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            current = baseline
            stamped = True

    if current >= target:
        return MigrationResult(db=db, from_version=current, to_version=current,
                               stamped_baseline=stamped)

    applied: list[str] = []
    for m in migs:
        if m.version <= current:
            continue
        # One transaction per step. `PRAGMA user_version` writes the database
        # header inside the transaction, so a rollback un-stamps it too: a
        # failed step can never leave a bumped version or a partial schema.
        conn.execute("BEGIN IMMEDIATE")
        try:
            m.apply(conn)
            conn.execute(f"PRAGMA user_version = {int(m.version)}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        applied.append(m.name)
        current = m.version

    return MigrationResult(db=db, from_version=_start_version(applied, migs, current),
                           to_version=current, applied=tuple(applied),
                           stamped_baseline=stamped)


def _start_version(applied: list[str], migs: Sequence[Migration], current: int) -> int:
    """The version the database was on before the applied steps ran."""
    if not applied:
        return current
    first = next(m for m in migs if m.name == applied[0])
    return first.version - 1


def status(
    conn: sqlite3.Connection,
    db: str,
    *,
    migrations_list: Sequence[Migration] | None = None,
) -> dict:
    """Report ``db``'s version and what is still pending. Read-only."""
    migs = _resolve(db, migrations_list)
    version = _read_version(conn)
    target = latest_version(db, migs)
    pending = [m.name for m in migs if m.version > version]
    return {
        "db": db,
        "user_version": version,
        "latest": target,
        "current": version >= target,
        "pending": pending,
    }


def _db_path(data_dir: Union[str, Path], db: str) -> Path:
    return Path(data_dir) / DB_FILES[db]


def status_all(data_dir: Union[str, Path]) -> list[dict]:
    """Status for every known database in ``data_dir``. Creates nothing.

    Databases that do not exist yet are reported with ``exists: False`` and are
    not counted as current, so a deploy pre-flight sees them as work to do.
    """
    rows: list[dict] = []
    for db in sorted(REGISTRY):
        path = _db_path(data_dir, db)
        if not path.exists():
            rows.append({
                "db": db,
                "path": str(path),
                "exists": False,
                "user_version": None,
                "latest": latest_version(db),
                "current": False,
                "pending": [m.name for m in REGISTRY[db]],
            })
            continue
        conn = sqlite3.connect(str(path))
        try:
            row = status(conn, db)
        finally:
            conn.close()
        row["path"] = str(path)
        row["exists"] = True
        rows.append(row)
    return rows


def migrate_all(data_dir: Union[str, Path]) -> list[MigrationResult]:
    """Migrate every database that already exists in ``data_dir``.

    Absent databases are skipped rather than created: an empty data directory
    is not a store, and each store module builds its own database on first
    open anyway.
    """
    from taosmd import _db  # noqa: PLC0415

    results: list[MigrationResult] = []
    for db in sorted(REGISTRY):
        path = _db_path(data_dir, db)
        if not path.exists():
            continue
        conn = _db.connect(path)
        try:
            results.append(migrate(conn, db))
        finally:
            conn.close()
    return results
