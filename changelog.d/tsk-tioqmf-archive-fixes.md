### Fixed

- Archive record ids are now stable. `ArchiveStore.record()` allocates a sequence number before the raw write, stores it as `seq` in the JSONL line and uses it as the index row id, so rebuilding the index from the raw files gives every record the id it had. Existing raw `.jsonl` and `.jsonl.gz` files are never rewritten.
- Each `record()` is one SQLite transaction (`BEGIN IMMEDIATE`, allocate, raw write, index insert, counter update, commit). Any failure rolls back and re-raises, so a failed index insert no longer leaves the writer unusable. Two processes sharing a directory cannot be handed the same id. A raw line left without an index row after a failure is picked up by the next `rebuild()`.
- New ids start above the larger of the stored counter and the highest index id, so rows written by an older release to the same directory never collide after an upgrade.
- Index rows written before `seq` existed keep their ids through an append-only `legacy-id-map` file in the archive directory. It is built automatically when the store is opened and again at the start of `rebuild()`; nothing needs to call `freeze_legacy_id_map()` any more.
- `rebuild()` takes the same write lock as `record()` and swaps the index in one transaction, so a failure part way leaves the old index untouched. Legacy lines the map does not know get fresh ids above every id in use.
- Opening a store no longer scans every raw file: the counter and the legacy-map watermark are kept in the index database, and only the tail of the newest raw file is read.
