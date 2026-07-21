"""Tests for the schema migration framework.

The valuable tests here are the legacy ones: a database created by an older
build has ``user_version = 0`` but already carries a modern schema, because
every schema change so far was applied by ``CREATE TABLE IF NOT EXISTS`` plus
hand-rolled ``try/except ALTER TABLE`` guards. A naive runner would re-apply
every step against such a store. The stamping tests below are the ones that
would have caught a real field break.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from taosmd import _db, migrations
from taosmd.migrations import Migration


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _spy(mig: Migration, calls: list[str]) -> Migration:
    """Wrap a migration so applying it records its name."""
    def _apply(conn, _inner=mig.apply, _name=mig.name):
        calls.append(_name)
        _inner(conn)
    return Migration(version=mig.version, name=mig.name, apply=_apply, detect=mig.detect)


def _spied(db: str, calls: list[str]) -> tuple[Migration, ...]:
    return tuple(_spy(m, calls) for m in migrations.REGISTRY[db])


# ----------------------------------------------------------------------
# registry sanity
# ----------------------------------------------------------------------

def test_registry_versions_are_contiguous_and_ordered():
    for db, migs in migrations.REGISTRY.items():
        versions = [m.version for m in migs]
        assert versions == list(range(1, len(migs) + 1)), (
            f"{db} migration versions must be 1..N in order, got {versions}"
        )


def test_every_registered_db_has_a_filename():
    for db in migrations.REGISTRY:
        assert db in migrations.DB_FILES


def test_every_migration_has_a_detect_probe():
    # detect() is what makes legacy stamping possible; a migration without
    # one would force a re-apply on every unstamped store.
    for db, migs in migrations.REGISTRY.items():
        for m in migs:
            assert callable(m.detect), f"{db} v{m.version} ({m.name}) has no detect probe"


# ----------------------------------------------------------------------
# fresh databases
# ----------------------------------------------------------------------

@pytest.mark.parametrize("db", sorted(migrations.REGISTRY))
def test_fresh_empty_database_is_built_at_latest_version(tmp_path, db):
    """An empty file gets the whole schema and lands on the latest version."""
    conn = _db.connect(tmp_path / f"{db}.db")
    result = migrations.migrate(conn, db)

    assert _user_version(conn) == migrations.latest_version(db)
    assert result.to_version == migrations.latest_version(db)
    assert result.from_version == 0
    # Everything ran; nothing was stamped as pre-existing.
    assert result.applied == tuple(m.name for m in migrations.REGISTRY[db])
    assert result.stamped_baseline is False
    conn.close()


@pytest.mark.parametrize("db", sorted(migrations.REGISTRY))
def test_migrate_is_idempotent(tmp_path, db):
    """Second and third runs are no-ops that touch nothing."""
    conn = _db.connect(tmp_path / f"{db}.db")
    migrations.migrate(conn, db)

    calls: list[str] = []
    second = migrations.migrate(conn, db, migrations_list=_spied(db, calls))
    third = migrations.migrate(conn, db, migrations_list=_spied(db, calls))

    assert calls == []
    assert second.applied == () and third.applied == ()
    assert second.to_version == third.to_version == migrations.latest_version(db)
    conn.close()


# ----------------------------------------------------------------------
# the zero-version problem: legacy stores stamped, never re-run
# ----------------------------------------------------------------------

def test_legacy_modern_archive_index_is_stamped_without_reapplying(tmp_path):
    """A store built by the pre-migration code path: modern schema, version 0.

    This is the field case. ``archive-index.db`` already has the ``project``
    column (the old hand-rolled ALTER guard added it), so the runner must
    recognise that and stamp straight to the latest version WITHOUT touching
    the schema or the data.
    """
    from taosmd.archive import INDEX_SCHEMA

    path = tmp_path / "archive-index.db"
    conn = _db.connect(path)
    conn.executescript(INDEX_SCHEMA)  # modern shape, includes `project`
    conn.execute(
        "INSERT INTO archive_index "
        "(timestamp, event_type, agent_name, app_id, project, summary,"
        " file_path, line_number, data_json) "
        "VALUES (1.0, 'turn', 'a', 'app', 'proj', 's', 'f.jsonl', 1, '{}')"
    )
    conn.commit()
    assert _user_version(conn) == 0

    calls: list[str] = []
    result = migrations.migrate(conn, "archive_index",
                                migrations_list=_spied("archive_index", calls))

    assert calls == [], "a modern legacy store must not re-run any migration"
    assert result.stamped_baseline is True
    assert result.applied == ()
    assert _user_version(conn) == migrations.latest_version("archive_index")
    # Data survived untouched.
    row = conn.execute("SELECT project, summary FROM archive_index").fetchone()
    assert tuple(row) == ("proj", "s")
    conn.close()


def test_genuinely_old_archive_index_gains_project_column(tmp_path):
    """A store older than the `project` back-fill: schema advanced, data kept."""
    path = tmp_path / "archive-index.db"
    conn = _db.connect(path)
    # The archive_index shape as it existed BEFORE the project column.
    conn.executescript(
        """
        CREATE TABLE archive_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            agent_name TEXT,
            app_id TEXT,
            summary TEXT NOT NULL DEFAULT '',
            file_path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            data_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE archive_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    conn.execute(
        "INSERT INTO archive_index "
        "(timestamp, event_type, agent_name, app_id, summary, file_path,"
        " line_number, data_json) "
        "VALUES (2.0, 'turn', 'old', 'app', 'kept', 'f.jsonl', 7, '{\"k\": 1}')"
    )
    conn.commit()
    assert "project" not in _columns(conn, "archive_index")

    result = migrations.migrate(conn, "archive_index")

    assert "project" in _columns(conn, "archive_index")
    assert _user_version(conn) == migrations.latest_version("archive_index")
    assert result.applied  # at least the project step ran
    row = conn.execute(
        "SELECT summary, line_number, data_json, project FROM archive_index"
    ).fetchone()
    assert tuple(row) == ("kept", 7, '{"k": 1}', None)
    conn.close()


def test_legacy_session_catalog_without_taxonomy_columns(tmp_path):
    """Pre-taxonomy catalog: the two old ALTER guards run as ordered steps."""
    conn = _db.connect(tmp_path / "session-catalog.db")
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT ''
        );
        """
    )
    conn.execute("INSERT INTO sessions (id, title) VALUES ('s1', 'kept')")
    conn.commit()

    migrations.migrate(conn, "session_catalog")

    cols = _columns(conn, "sessions")
    for expected in ("primary_project", "primary_topic", "primary_subtopic",
                     "labels_json", "classified_at", "agent_name"):
        assert expected in cols
    assert _user_version(conn) == migrations.latest_version("session_catalog")
    assert conn.execute("SELECT title FROM sessions").fetchone()[0] == "kept"
    conn.close()


def test_partially_migrated_database_advances_only_the_missing_steps(tmp_path):
    """Taxonomy columns present, agent_name missing: only the tail runs."""
    conn = _db.connect(tmp_path / "session-catalog.db")
    migrations.migrate(conn, "session_catalog")
    # Rewind to the state of a store that stopped one version short.
    conn.execute("PRAGMA user_version = 0")
    conn.execute("ALTER TABLE sessions DROP COLUMN agent_name")
    conn.commit()

    calls: list[str] = []
    result = migrations.migrate(conn, "session_catalog",
                               migrations_list=_spied("session_catalog", calls))

    assert "agent_name" in _columns(conn, "sessions")
    assert _user_version(conn) == migrations.latest_version("session_catalog")
    # The baseline and taxonomy steps were detected as present, not re-run.
    assert len(calls) < len(migrations.REGISTRY["session_catalog"])
    assert result.from_version > 0, "the present prefix should have been stamped"
    conn.close()


def test_legacy_vector_memory_gains_valid_to_and_its_index(tmp_path):
    """The third hand-rolled guard: valid_to plus the deferred index.

    The index must be created after the column exists; that ordering hazard is
    exactly what the old ``_migrate`` comment warned about.
    """
    conn = _db.connect(tmp_path / "vector-memory.db")
    conn.executescript(
        """
        CREATE TABLE vector_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB,
            metadata TEXT NOT NULL DEFAULT '{}',
            timestamp REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO vector_memory (content, metadata, timestamp)"
        " VALUES ('hello', '{}', 1.0)"
    )
    conn.commit()

    migrations.migrate(conn, "vector_memory")

    assert "valid_to" in _columns(conn, "vector_memory")
    idx = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_vm_valid" in idx
    row = conn.execute("SELECT content, valid_to FROM vector_memory").fetchone()
    assert tuple(row) == ("hello", None), "existing rows stay active"
    assert _user_version(conn) == migrations.latest_version("vector_memory")
    conn.close()


def test_legacy_collections_db_is_stamped_to_phase1_baseline(tmp_path):
    """collections.db shipped with no migration path at all; stamp it."""
    from taosmd.collections import SCHEMA as COLLECTIONS_SCHEMA

    conn = sqlite3.connect(tmp_path / "collections.db")
    conn.executescript(COLLECTIONS_SCHEMA)
    conn.execute(
        "INSERT INTO collections (id, name, kind, source_path, status, created_at)"
        " VALUES ('col-1', 'docs', 'folder', '/tmp/docs', 'ready', 1.0)"
    )
    conn.commit()

    calls: list[str] = []
    result = migrations.migrate(conn, "collections",
                               migrations_list=_spied("collections", calls))

    assert calls == []
    assert result.stamped_baseline is True
    assert _user_version(conn) == migrations.latest_version("collections")
    assert conn.execute("SELECT name FROM collections").fetchone()[0] == "docs"
    conn.close()


# ----------------------------------------------------------------------
# failure atomicity
# ----------------------------------------------------------------------

def test_failing_migration_rolls_back_and_leaves_version_unchanged(tmp_path):
    conn = _db.connect(tmp_path / "t.db")

    def _ok(c):
        c.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, a TEXT)")

    def _boom(c):
        c.execute("ALTER TABLE t ADD COLUMN b TEXT")
        raise RuntimeError("migration exploded")

    migs = (
        Migration(1, "base", _ok, lambda c: migrations.table_exists(c, "t")),
        Migration(2, "boom", _boom, lambda c: migrations.has_column(c, "t", "b")),
    )

    with pytest.raises(RuntimeError, match="exploded"):
        migrations.migrate(conn, "t", migrations_list=migs)

    # v1 committed on its own; v2 rolled back entirely.
    assert _user_version(conn) == 1
    assert "b" not in _columns(conn, "t")
    assert not conn.in_transaction
    conn.close()


def test_failing_migration_does_not_leave_a_half_applied_schema(tmp_path):
    """Two statements, second fails: neither survives."""
    conn = _db.connect(tmp_path / "t.db")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    conn.commit()

    def _half(c):
        c.execute("ALTER TABLE t ADD COLUMN one TEXT")
        c.execute("ALTER TABLE nope ADD COLUMN two TEXT")  # no such table

    migs = (Migration(1, "half", _half, lambda c: migrations.has_column(c, "t", "two")),)

    with pytest.raises(sqlite3.OperationalError):
        migrations.migrate(conn, "t", migrations_list=migs)

    assert _columns(conn, "t") == {"id"}
    assert _user_version(conn) == 0
    conn.close()


# ----------------------------------------------------------------------
# the scenario that is broken today: a new column on an existing table
# ----------------------------------------------------------------------

def test_new_column_on_existing_table_reaches_an_existing_store(tmp_path):
    """The demonstration.

    Today, adding a column to an existing table by editing the ``CREATE TABLE
    IF NOT EXISTS`` schema does nothing to a database that already exists, and
    every query naming the column fails at runtime. This proves the framework
    fixes exactly that: an already-current store, stamped at the latest
    version, gains a newly appended column on the next open.
    """
    from taosmd.archive import INDEX_SCHEMA

    path = tmp_path / "archive-index.db"
    conn = _db.connect(path)
    conn.executescript(INDEX_SCHEMA)
    conn.execute(
        "INSERT INTO archive_index "
        "(timestamp, event_type, summary, file_path, line_number)"
        " VALUES (1.0, 'turn', 'before', 'f.jsonl', 1)"
    )
    conn.commit()
    migrations.migrate(conn, "archive_index")
    current = migrations.latest_version("archive_index")

    # Prove the failure mode first: re-running the CREATE TABLE script with a
    # new column in it changes nothing on an existing database.
    conn.executescript(INDEX_SCHEMA.replace(
        "data_json TEXT NOT NULL DEFAULT '{}'",
        "data_json TEXT NOT NULL DEFAULT '{}',\n    retention_class TEXT",
    ))
    assert "retention_class" not in _columns(conn, "archive_index")

    # Now the same change as a migration appended to the list.
    def _add_retention_class(c):
        migrations.add_column(c, "archive_index", "retention_class", "TEXT")

    extended = migrations.REGISTRY["archive_index"] + (
        Migration(current + 1, "archive_index_retention_class",
                  _add_retention_class,
                  lambda c: migrations.has_column(c, "archive_index", "retention_class")),
    )

    result = migrations.migrate(conn, "archive_index", migrations_list=extended)

    assert "retention_class" in _columns(conn, "archive_index")
    assert result.applied == ("archive_index_retention_class",)
    assert _user_version(conn) == current + 1
    # Round trip: the pre-existing row is intact and readable through the
    # new column list.
    row = conn.execute(
        "SELECT summary, retention_class FROM archive_index").fetchone()
    assert tuple(row) == ("before", None)

    # And it is idempotent on the next open.
    again = migrations.migrate(conn, "archive_index", migrations_list=extended)
    assert again.applied == ()
    conn.close()


# ----------------------------------------------------------------------
# status / check
# ----------------------------------------------------------------------

def test_status_reports_pending_for_an_unstamped_legacy_store(tmp_path):
    conn = _db.connect(tmp_path / "archive-index.db")
    conn.executescript(
        "CREATE TABLE archive_index (id INTEGER PRIMARY KEY, timestamp REAL,"
        " event_type TEXT, agent_name TEXT, app_id TEXT, summary TEXT,"
        " file_path TEXT, line_number INTEGER, data_json TEXT);"
        "CREATE TABLE archive_settings (key TEXT PRIMARY KEY, value TEXT);"
    )
    conn.commit()

    st = migrations.status(conn, "archive_index")
    assert st["user_version"] == 0
    assert st["latest"] == migrations.latest_version("archive_index")
    assert st["current"] is False
    # The baseline is already present, so it is reported as detected rather
    # than as a step that would run.
    assert st["detected_baseline"] == 1
    assert st["pending"] == ["archive_index_project"]

    migrations.migrate(conn, "archive_index")
    st = migrations.status(conn, "archive_index")
    assert st["current"] is True
    assert st["pending"] == []
    conn.close()


def test_status_of_a_complete_legacy_store_reports_stamp_only_work(tmp_path):
    """Schema already modern, version 0: nothing to apply, stamp still needed."""
    from taosmd.archive import INDEX_SCHEMA

    conn = _db.connect(tmp_path / "archive-index.db")
    conn.executescript(INDEX_SCHEMA)
    conn.commit()

    st = migrations.status(conn, "archive_index")
    assert st["user_version"] == 0
    assert st["detected_baseline"] == migrations.latest_version("archive_index")
    assert st["pending"] == [], "a fully present schema has no step to execute"
    assert st["current"] is False, "the stamp itself is still outstanding"
    conn.close()


def test_status_all_skips_absent_databases(tmp_path):
    rows = migrations.status_all(tmp_path)
    assert {r["db"] for r in rows} == set(migrations.REGISTRY)
    assert all(r["exists"] is False for r in rows)
    assert all(r["current"] is False for r in rows)


def test_migrate_all_brings_every_present_database_current(tmp_path):
    from taosmd.archive import INDEX_SCHEMA

    conn = _db.connect(tmp_path / "archive-index.db")
    conn.executescript(INDEX_SCHEMA)
    conn.commit()
    conn.close()

    results = migrations.migrate_all(tmp_path)
    assert [r.db for r in results] == ["archive_index"]

    rows = [r for r in migrations.status_all(tmp_path) if r["exists"]]
    assert len(rows) == 1
    assert rows[0]["current"] is True


# ----------------------------------------------------------------------
# integration: the real store open paths
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_archive_store_open_stamps_the_database(tmp_path):
    from taosmd.archive import ArchiveStore

    store = ArchiveStore(archive_dir=tmp_path / "archive",
                         index_path=tmp_path / "archive-index.db")
    await store.init()
    await store.record("turn", {"text": "hi"}, agent_name="a", summary="s")
    await store.close()

    conn = sqlite3.connect(tmp_path / "archive-index.db")
    assert _user_version(conn) == migrations.latest_version("archive_index")
    assert "project" in _columns(conn, "archive_index")
    conn.close()


@pytest.mark.asyncio
async def test_archive_store_upgrades_a_legacy_index_in_place(tmp_path):
    """End to end: a real ArchiveStore opened over a pre-project database."""
    from taosmd.archive import ArchiveStore

    path = tmp_path / "archive-index.db"
    conn = _db.connect(path)
    conn.executescript(
        """
        CREATE TABLE archive_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            agent_name TEXT,
            app_id TEXT,
            summary TEXT NOT NULL DEFAULT '',
            file_path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            data_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    conn.execute(
        "INSERT INTO archive_index (timestamp, event_type, summary, file_path,"
        " line_number) VALUES (1.0, 'turn', 'legacy', 'f.jsonl', 1)"
    )
    conn.commit()
    conn.close()

    store = ArchiveStore(archive_dir=tmp_path / "archive", index_path=path)
    await store.init()
    await store.record("turn", {"text": "new"}, agent_name="a", summary="fresh")
    events = await store.search_fts("legacy")
    await store.close()

    conn = sqlite3.connect(path)
    assert _user_version(conn) == migrations.latest_version("archive_index")
    assert "project" in _columns(conn, "archive_index")
    assert conn.execute("SELECT COUNT(*) FROM archive_index").fetchone()[0] == 2
    conn.close()
    assert events is not None


@pytest.mark.asyncio
async def test_session_catalog_open_stamps_the_database(tmp_path):
    from taosmd.session_catalog import SessionCatalog

    cat = SessionCatalog(db_path=tmp_path / "session-catalog.db",
                         archive_dir=tmp_path / "archive",
                         sessions_dir=tmp_path / "sessions")
    await cat.init()
    await cat.close()

    conn = sqlite3.connect(tmp_path / "session-catalog.db")
    assert _user_version(conn) == migrations.latest_version("session_catalog")
    cols = _columns(conn, "sessions")
    assert "primary_project" in cols and "agent_name" in cols
    conn.close()


@pytest.mark.asyncio
async def test_claim_store_open_stamps_the_database(tmp_path):
    from taosmd.claims.store import ClaimStore

    store = ClaimStore(db_path=tmp_path / "claims.db")
    await store.init()
    await store.add_claim("x", [1], "test")
    await store.close()

    conn = sqlite3.connect(tmp_path / "claims.db")
    assert _user_version(conn) == migrations.latest_version("claims")
    conn.close()


@pytest.mark.asyncio
async def test_knowledge_graph_open_stamps_the_database(tmp_path):
    from taosmd.knowledge_graph import TemporalKnowledgeGraph

    kg = TemporalKnowledgeGraph(db_path=tmp_path / "knowledge-graph.db")
    await kg.init()
    await kg.close()

    conn = sqlite3.connect(tmp_path / "knowledge-graph.db")
    assert _user_version(conn) == migrations.latest_version("knowledge_graph")
    conn.close()


def test_collection_store_open_stamps_the_database(tmp_path):
    from taosmd.collections import CollectionStore

    store = CollectionStore(tmp_path)
    store.close()

    conn = sqlite3.connect(tmp_path / "collections.db")
    assert _user_version(conn) == migrations.latest_version("collections")
    conn.close()


@pytest.mark.asyncio
async def test_vector_memory_open_stamps_the_database(tmp_path):
    from taosmd.vector_memory import VectorMemory

    vm = VectorMemory(db_path=tmp_path / "vector-memory.db")
    await vm.init()
    await vm.close()

    conn = sqlite3.connect(tmp_path / "vector-memory.db")
    assert _user_version(conn) == migrations.latest_version("vector_memory")
    assert "valid_to" in _columns(conn, "vector_memory")
    conn.close()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _seed_legacy_archive(tmp_path):
    conn = _db.connect(tmp_path / "archive-index.db")
    conn.executescript(
        """
        CREATE TABLE archive_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            agent_name TEXT,
            app_id TEXT,
            summary TEXT NOT NULL DEFAULT '',
            file_path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            data_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE archive_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    conn.commit()
    conn.close()


def test_cli_check_exits_nonzero_when_a_database_is_behind(tmp_path, capsys):
    from taosmd.cli import main

    _seed_legacy_archive(tmp_path)

    rc = main(["--data-dir", str(tmp_path), "migrate", "--check"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "archive_index" in out
    assert "PENDING" in out
    # A dry run must not have changed anything.
    conn = sqlite3.connect(tmp_path / "archive-index.db")
    assert _user_version(conn) == 0
    conn.close()


def test_cli_migrate_then_check_is_clean(tmp_path, capsys):
    from taosmd.cli import main

    _seed_legacy_archive(tmp_path)

    assert main(["--data-dir", str(tmp_path), "migrate"]) == 0
    capsys.readouterr()
    assert main(["--data-dir", str(tmp_path), "migrate", "--check"]) == 0
    assert "PENDING" not in capsys.readouterr().out


def test_cli_status_json_lists_every_database(tmp_path, capsys):
    from taosmd.cli import main

    _seed_legacy_archive(tmp_path)
    assert main(["--data-dir", str(tmp_path), "migrate", "--status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_db = {r["db"]: r for r in payload["databases"]}
    assert set(by_db) == set(migrations.REGISTRY)
    assert by_db["archive_index"]["exists"] is True
    assert by_db["archive_index"]["current"] is False
    assert by_db["claims"]["exists"] is False
