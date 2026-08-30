### Fixed
- `ReceiptStore` (a2a-receipts.db) now opens through `taosmd._db.connect`, engaging
  WAL journal mode and a 5000 ms busy timeout like the other stores. Previously it
  bypassed the shared helper via a raw `sqlite3.connect`, running in rollback-journal
  mode with a zero busy timeout and raising `database is locked` under contention
  instead of waiting and retrying. `_db.connect` gained an optional keyword-only
  `check_same_thread` parameter (default `True` to preserve every existing call
  site) so `ReceiptStore` can pass `check_same_thread=False` without a thread-affinity
  crash.
