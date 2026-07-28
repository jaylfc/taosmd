"""collections.db must be opened like every other store: WAL, busy timeout."""

from taosmd.collections import CollectionStore


def test_collections_db_is_in_wal_mode(tmp_path):
    store = CollectionStore(tmp_path)
    try:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        store.close()


def test_collections_db_has_a_busy_timeout(tmp_path):
    store = CollectionStore(tmp_path)
    try:
        timeout = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout > 0
    finally:
        store.close()
