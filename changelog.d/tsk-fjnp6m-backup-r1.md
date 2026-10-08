### Added
- `taosmd backup create`, `verify`, and `restore` commands for atomic,
  WAL-safe backup of the data dir. SQLite files are copied via
  `sqlite3.Connection.backup()`; `config.json` is excluded by default and
  included only with `--include-secrets`.
