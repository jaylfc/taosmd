### Fixed
- MentionStore opened its SQLite connection without `check_same_thread=False`, so
  the thread-affine connection raised `ProgrammingError` when the A2A mention feed
  was served from a `ThreadingHTTPServer` request thread other than the one that
  created it. Routed through `taosmd._db.connect(..., check_same_thread=False)`,
  matching the ReceiptStore fix.
