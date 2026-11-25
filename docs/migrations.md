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
taosmd migrate --probe     # read-only store format probe, exit code is the verdict
taosmd migrate --probe --json
```

`--status` runs the same detect walk the runner would, so an unstamped store
whose schema is already complete is reported as needing only a stamp, with an
empty pending list, rather than appearing to face a replay of every step.

`--status` and `--probe` open each database read-only and immutable, so they
create no `-wal` or `-shm` files and do not switch a database's journal mode.
The one exception for `--status`: when a non-empty `-wal` or `-journal` already
exists it falls back to a normal connection so the report reflects that content.

`--probe` is the format check to run before an upgrade. It looks at each
database's `PRAGMA user_version` against the latest known step, and at the
vector store's `store_meta` marker and the real row encoding (JSON `[...]`
vectors versus base64 float32). It never calls a store class, so it converts
nothing, and it changes nothing on disk. Exit codes:

| code | state   | meaning |
|------|---------|---------|
| 0    | current | every database opens as is: at the latest step, or needing only a version stamp (reported as "stamp only"), or not created yet. A missing or empty data dir is a first install and counts |
| 10   | old     | a schema step is pending, or the vector store still holds JSON vectors. Legacy JSON vectors are a supported state served as is, not an error; converting them is optional and never happens on open |
| 11   | unknown | unreadable, a live writer or uncheckpointed `-wal`, an unparseable marker, or rows that contradict the marker |
| 12   | newer   | a `user_version` above this build's latest, or marker keys or a `dense_format` this build does not know |

A 0.4.x `binary_quant` or `late_interaction` store (a marker with no
`dense_format`) is `current`: it opens as is and nothing converts it.

When databases disagree the worst wins: newer, then unknown, then old.
`.db` files that are not in the migration registry are listed as `unversioned`
and do not affect the exit code. From Python, `taosmd.store_probe.probe_store(dir)`
returns the same report (overall state plus per-database path, state,
`user_version`, `latest`, reason).

`--check` is the deploy pre-flight: it writes nothing and exits non-zero when
any database in the data directory is behind, so a rollout can refuse to start
against a store it would silently need to upgrade. Databases that do not exist
yet are reported as absent and are not treated as pending work by `--check`,
since each store builds its own on first open.

## Release metadata trailer rule

`scripts/release_meta.py` parses changelog trailer lines with this rule. It normalises each line (NFKC, remove Unicode format/Cf and U+00AD, casefold, strip), then strips leading markup (whitespace, `>`, `#`, `-`, `*`, `+`, `_`, `~`, backticks, `|`, checkboxes, numbered/lettered list markers, `<!--`). A trailer attempt is a remaining line that starts with `migration`/`migrations` or `manual[\W_]*steps?`, optional whitespace, and a colon-like char (`:`, Unicode colon forms after NFKC, or `=`). An exact canonical form parses (`Migration: one-way`, `Migration: reversible`, `Manual-Step: <non-empty text>`); any other trailer attempt refuses with exit 2 naming file, line, and line text. Prose that merely mentions migration (e.g. `migrate`, `migrating`) is ignored. Trailer attempts inside closed ```/~~~ fences are ignored; an unclosed fence refuses. A confusable-key homoglyph followed by a colon-like char refuses.

## Data relocation

`taosmd data status [--json]` reports the data dir, per-store sizes, free
space on its filesystem, and a `low_space` flag (free < 10% or < 5 GiB by
default).

`taosmd data relocate <new_dir>` moves the data directory in a guarded
sequence: refuse if stores are open for write, check target free space, copy
(SQLite via the backup API, other files with fsync), verify (SHA-256 and
`PRAGMA integrity_check` plus row counts), then switch the configured data
dir atomically. The old directory is kept with a marker until
`taosmd data relocate --confirm-remove-old` removes it after re-verifying
the new copy. Exit codes: 0 moved, 2 refused, 4 failed and cleaned up, 1
unexpected.
