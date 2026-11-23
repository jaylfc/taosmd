### Fixed

- `Archive.record()` now assigns and durably persists the record sequence id before the raw JSONL write and stores `seq` in the line; `Archive.rebuild()` restores the index from the raw `.jsonl`/`.jsonl.gz` files using each line's `seq` (or the frozen `legacy-id-map` for pre-`seq` lines), so rebuilding the index can never renumber records and existing raw files are never rewritten. New seq values start above the highest existing id.
