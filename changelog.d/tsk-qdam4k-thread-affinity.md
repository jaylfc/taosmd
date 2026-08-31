### Fixed

- All sixteen remaining `_db.connect` call sites outside `taosmd/_db.py` now
  pass `check_same_thread=False`.  The helper's default remains `True`, so the
  stdlib thread guard is preserved everywhere it was not explicitly revoked.
  This removes the `ProgrammingError` raised when a store connection is created
  on the main thread and subsequently used from the service-loop thread in the
  Python-API-then-serve deployment shape.  Added thread-affinity regression
  tests for `ArchiveStore` and `TemporalKnowledgeGraph` in
  `tests/test_thread_affinity.py`.
