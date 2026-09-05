### Fixed

- `migrations.status_all()` now opens each present database through the shared
  `taosmd._db.connect` helper, so it picks up the package-wide WAL journal
  mode and busy timeout instead of taking a bare `sqlite3.connect`. The
  absent-database branch is still skipped, and the new test in
  `tests/test_migrations.py` fails if the bare connect is restored.
