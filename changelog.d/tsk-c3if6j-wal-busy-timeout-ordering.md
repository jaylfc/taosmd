### Fixed
- `_db.connect` now sets `PRAGMA busy_timeout` before `PRAGMA journal_mode=WAL`.
  The WAL pragma takes a brief exclusive lock to rewrite the database header;
  when busy_timeout is armed first, concurrent first-time opens block-and-retry
  instead of raising `OperationalError: database is locked` immediately.
