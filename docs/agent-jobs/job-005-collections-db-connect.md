# JOB-005: Open collections.db through the shared _db.connect helper

Read docs/agent-jobs/README.md first and follow its absolute rules.

- Tracks issue #202.
- Branch: `fix/collections-db-connect` (from `origin/master`)
- Commit message: `fix(collections): open collections.db through _db.connect for WAL and a busy timeout`
- PR title: `fix(collections): open collections.db through _db.connect for WAL and a busy timeout`
- Allowed files: `taosmd/collections.py`, `tests/test_collections_db_connect.py`
  (new) ONLY.

## The bug

Every persistent store in the package opens its database through
`taosmd/_db.py`'s `connect` helper, which enables WAL and sets a busy timeout
so a contended writer waits and retries instead of failing immediately with
`database is locked`. `CollectionStore` alone uses a bare `sqlite3.connect`.

`taosmd/collections.py` line 116:

```python
        self._conn = sqlite3.connect(str(path / "collections.db"))
```

Collections indexing runs on the service loop and can overlap with search
requests that read collection rows and grants, so this is a latent
concurrency failure, not only an inconsistency.

This has already been checked for you: nothing in `CollectionStore` reads or
depends on the journal mode. There is no `PRAGMA journal_mode`, no `VACUUM`,
no `ATTACH`, no `isolation_level` handling, and no test asserting the files
present in the data directory. So this is a genuine one-line swap. If you
find otherwise, that changes the job: STOP per the STOP conditions below.

## Steps

1. `git fetch origin && git checkout -b fix/collections-db-connect origin/master`
2. Read `taosmd/_db.py` in full (about 60 lines) and
   `taosmd/collections.py` from line 100 to line 165.
3. Confirm for yourself that nothing depends on the journal mode:

```
grep -E -n "journal|VACUUM|ATTACH|isolation_level|backup" taosmd/collections.py
```

   This must print nothing. If it prints anything, STOP.
4. Write the test FIRST, in a new file `tests/test_collections_db_connect.py`:

```python
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
```

5. Run it and watch it FAIL:
   `python3 -m pytest tests/test_collections_db_connect.py -q`
   Expect the journal-mode test to report `delete` instead of `wal`. If it
   already passes, the fix is already in: STOP and say so in a PR comment
   rather than committing a no-op.
6. Fix `taosmd/collections.py`. Two edits, nothing else:

   a. Add the import. The file already has other relative imports; put this
      one with them, matching the style used in `taosmd/access_tracker.py`:

```python
from . import _db
```

   b. Change line 116 from

```python
        self._conn = sqlite3.connect(str(path / "collections.db"))
```

   to

```python
        self._conn = _db.connect(str(path / "collections.db"))
```

   Leave the `self._conn.row_factory = sqlite3.Row` line immediately below it
   exactly as it is. Do NOT remove `import sqlite3` from the top of the file:
   `sqlite3.Row` still needs it. Confirm with
   `grep -n "sqlite3\." taosmd/collections.py` that other uses remain.
7. `python3 -m pytest tests/test_collections_db_connect.py -q` (2 passed),
   then the FULL suite `python3 -m pytest -q -m "not slow"`, then
   `python3 -m ruff check taosmd/collections.py tests/test_collections_db_connect.py`.
8. One commit, push, open the PR. PR body: the inconsistency, the one-line
   fix, confirmation that nothing in the store depended on the journal mode,
   and the test count. Reference issue #202. Do not merge.

## STOP conditions

- If the grep in step 3 finds anything, or any existing test asserts which
  files exist in the data directory, STOP. WAL creates `collections.db-wal`
  and `collections.db-shm` sidecar files, and a test that counted files would
  now fail. That makes this a judgment call rather than a swap, so it stops
  being your job. Open the PR with the test only, describe what you found,
  and stop.
- If the full suite shows failures in collections tests after the swap, do
  NOT start adjusting those tests. STOP, open the PR with what you have, and
  list the failures.
- Issue #202 is one of several databases-and-schema issues. Do not touch any
  other store, and do not go near `taosmd/migrations.py`.
