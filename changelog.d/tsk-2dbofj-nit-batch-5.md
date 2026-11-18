### Fixed
- sync_merge: remap peer `superseded_by` to local ids in a second pass after all inserts
- sync_merge: ensure file ends with exactly one newline
- api: add `sid` to `seen` immediately after archive write for empty items with ids
- archive: move `os.open` for directory fsync inside try/except to handle Windows PermissionError
- benchmarks/e053_npu_speed: use `LOOP_CAP_TOKENS` constant instead of hardcoded 512
