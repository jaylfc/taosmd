# Schema migrations

Every SQLite database taOSmd owns carries its own schema version in the
database file itself, via SQLite's native `PRAGMA user_version`. `taosmd/migrations.py`
holds the ordered list of steps per database and the runner that applies them.

Each store runs the migrator on open, before it issues its first query, so an
upgraded install repairs its own schema the next time it starts. `taosmd migrate`
exists for the cases where you want it out of band.

## Why it exists

Before this, there was no version stamp and no upgrade path. Schemas were
created with `CREATE TABLE IF NOT EXISTS` and evolved by hand-rolled
`try/except ALTER TABLE` guards in three store modules. That works only while
every change is a whole new table or a column added behind its own bespoke
guard. `CREATE TABLE IF NOT EXISTS` does nothing to a database that already has
the table, so editing a `CREATE TABLE` statement to add a column reaches new
installs and no existing one, and every query naming the column then fails at
runtime on exactly the stores that hold real data.

## The two rules

Both are enforced by tests in `tests/test_migrations.py`.

**Idempotency rule.** Applying a step to a database that already has the change
must be harmless. Use `CREATE TABLE / INDEX IF NOT EXISTS` and the
`add_column()` helper, which skips the `ALTER` when the column is already
present. The runner relies on this: a database whose schema is ahead of its
version stamp in some places but not others will re-run steps it may already
have.

**Transaction rule.** A step must not commit, roll back, or call
`executescript()`. The runner opens `BEGIN IMMEDIATE` around each step and
stamps `user_version` inside that same transaction, so a failure rolls back the
schema change and the version together, never leaving a half-applied schema or
a version that lies. `executescript()` issues an implicit COMMIT before it
runs, which would silently break that guarantee; use `exec_script()` from the
same module for multi-statement SQL.

## Adding a migration

1. Write the change as a function taking a connection.
2. Write a `detect` probe: a cheap read that answers "is this change already
   present in the schema?", usually via `has_column()`, `table_exists()`, or
   `index_exists()`.
3. Append a `Migration` to that database's tuple in `REGISTRY` with the next
   version number.

```python
def _archive_index_retention_class(conn):
    add_column(conn, "archive_index", "retention_class", "TEXT")


_ARCHIVE_INDEX = (
    ...,
    Migration(
        3, "archive_index_retention_class", _archive_index_retention_class,
        lambda c: has_column(c, "archive_index", "retention_class"),
    ),
)
```

Never renumber, reorder, or edit a step that has shipped. Stores in the field
are already stamped past it, so an edit reaches nobody and a renumber makes
their stamps mean something different. Correct a shipped step by appending a
new one.

The `detect` probe is not optional. It is what lets a database created before
versioning existed be recognised and stamped rather than replayed; see below.

## Where the schema constants fit

Each store module keeps its `SCHEMA` string of `CREATE TABLE IF NOT EXISTS`
statements and still runs it on open, and version 1 of each database is a
"baseline" migration that runs the same constant. That redundancy is
deliberate, and it draws the line you need to remember:

- Adding a **whole new table or index** can go in the schema constant. The
  `IF NOT EXISTS` form genuinely reaches existing databases.
- **Changing an existing table** (adding, renaming, or retyping a column) must
  be a migration. The schema constant cannot deliver it.

## The zero-version problem

Databases created before this framework report `user_version = 0` while already
carrying a fully modern schema, because nothing ever stamped them. A runner
that trusted the zero would replay every step against a live store.

So on an unstamped database the runner does not migrate first, it *looks*
first. It walks the migration list in order asking each step's `detect` probe
whether that change is already in the schema, and the longest leading run of
steps that answer yes becomes the baseline: the version the database is stamped
to with nothing executed. Only the remainder is applied.

That single mechanism covers every case:

| database state | detected | result |
| --- | --- | --- |
| new empty file | nothing | built from step 1 to the latest version |
| created by an older build, modern schema | everything | stamped to latest, zero schema writes |
| created by an older build, mid-era schema | a prefix | stamped to the prefix, the tail applied |
| already stamped and current | not consulted | one PRAGMA read, no-op |

The walk stops at the first step that does not detect itself, rather than
skipping over it. A later step that happens to be present is re-applied, which
is safe under the idempotency rule and is the conservative choice: it never
assumes a step ran.

## Inspecting and applying out of band

```
taosmd migrate --status     # per-database version, latest, pending steps
taosmd migrate --check      # dry run; exits 1 if anything is pending
taosmd migrate              # apply pending migrations to the data dir
taosmd migrate --status --json
```

`--check` is the deploy pre-flight: it writes nothing and exits non-zero when
any database in the data directory is behind, so a rollout can refuse to start
against a store it would silently need to upgrade. Databases that do not exist
yet are reported as absent and are not treated as pending work by `--check`,
since each store builds its own on first open.
