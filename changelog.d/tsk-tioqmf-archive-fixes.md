### Fixed
- Archive writer no longer wedges after a failed index insert: each `record()` now runs in a single transaction with proper rollback on any exception, so subsequent records succeed.
- Sequence allocation now uses `MAX(id)` from the index, preventing collisions when older code wrote higher ids than the persisted counter.
- Legacy id map is now built automatically and idempotently on first write-open (`init()`), removing the need for manual `freeze_legacy_id_map()` calls before rebuild.
- `rebuild()` now takes the write lock, assigns mapped legacy ids first, gives unmapped legacy lines fresh ids above all mapped/stored ids (no collisions), and performs the delete/reinsert in one atomic transaction so a crash mid-rebuild leaves the old index intact.
- Cross-process safety strengthened: the new test uses a barrier with 300 records per process run 3 times, which would fail without `BEGIN IMMEDIATE`.
- `init()` no longer rescans all raw files on each open: the sequence counter and legacy map watermark are persisted in the index db, so reopens are fast (sub-millisecond).
