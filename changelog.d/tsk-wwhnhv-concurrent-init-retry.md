### Fixed
- Concurrent first-time store initialisation no longer loses rows when multiple
  processes race the CREATE TABLE / CREATE INDEX DDL.  `_db.run_schema` retries
  `executescript` on transient ``database is locked`` errors, and all 15 store
  init sites plus the `ReceiptStore` now use it.  The realistic trigger is a
  CLI process racing a running server against the same fresh data directory.
