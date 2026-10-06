### Added

- `taosmd backup create/verify/restore` for consistent point-in-time snapshots of the whole data dir. SQLite files are copied via `sqlite3.Connection.backup()` so WAL pages are included. `config.json` is excluded by default (it holds bearer tokens) and included only with `--include-secrets`. Backups verify against a MANIFEST.json with sha256 hashes and integrity checks. Restore is zero-loss: it never overwrites existing content and supports `--move-existing` to rename an existing target.
