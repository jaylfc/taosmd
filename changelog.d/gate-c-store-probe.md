### Added

- `taosmd migrate --probe [--json]` and `taosmd.store_probe.probe_store()`: a read-only check of the on-disk store format. It reports each database as current, old, unknown or newer than this build, with exit codes 0, 10, 11 and 12, and creates no `-wal` or `-shm` files.

### Fixed

- `taosmd migrate --status` and `--check` no longer create `-wal`/`-shm` files or switch a database to WAL mode just to report its version.
