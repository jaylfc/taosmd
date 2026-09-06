### Fixed
- `ReceiptStore` (a2a-receipts.db) now opens through `taosmd._db.connect`, engaging
  WAL journal mode and a 5000 ms busy timeout. Previously it bypassed the shared
  helper via a raw `sqlite3.connect`, so the file ran in rollback-journal mode and
  raised `database is locked` under contention instead of waiting and retrying.
  `_db.connect` gained a keyword-only `check_same_thread` parameter (default
  `True`, preserving every existing single-arg call site) so `ReceiptStore` can
  pass `check_same_thread=False` without a thread-affinity crash. The busy-timeout
  test pins against `_db.BUSY_TIMEOUT_MS` at a non-default value so the PRAGMA
  assertion discriminates; deleting the PRAGMA makes it fail.
