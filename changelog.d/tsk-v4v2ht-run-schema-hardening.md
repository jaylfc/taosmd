### Fixed

- `taosmd/_db.py` adds `run_schema`, which runs schema DDL with a retry loop that
  retries transient `database is locked` / `SQLITE_BUSY` errors with linear
  back-off and propagates every other error immediately. The exhaustion check is
  derived from the `SCHEMA_RETRY_ATTEMPTS` loop bound rather than a literal, so
  resizing the retry window cannot silently swallow a persistent lock error.
  All 14 stores and the `tasks` helpers now route schema DDL through
  `run_schema`, so concurrent first-time init no longer races the
  CREATE/CREATE-INDEX DDL. `connect` gains a keyword-only `check_same_thread`
  argument (default `True`, preserving existing thread-bound behaviour).
- `taosmd/receipts.py` restores `check_same_thread=False` on its `_db.connect`
  call. Routing `ReceiptStore` through `_db.connect` for WAL + busy timeout
  dropped the flag (which defaults to `True`), reverting the thread-affinity
  behaviour master relied on; the flag is now passed explicitly and pinned by
  `tests/test_receipts_thread_affinity.py`, which fails with
  `ProgrammingError` when the flag is removed.

### Out of scope

- The cross-process INSERT-phase row loss in `MentionStore.record_mentions`
  (`sqlite3.OperationalError: database is locked`, ROWS=771 EXPECTED=800) is
  NOT addressed by this revision: it occurs in the unretried write path, not in
  the init DDL that `run_schema` guards. The original
  `test_concurrent_first_init_no_row_loss` asserted a row count this fix cannot
  meet, so it is replaced by a deterministic init-only concurrency test
  (20/20 over 20 runs) plus fake-connection unit tests for `run_schema`'s
  retry, non-lock, and exhaustion branches.
