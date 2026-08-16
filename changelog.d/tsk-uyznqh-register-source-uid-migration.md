### Fixed

- Registered the `archive_index_source_uid` migration (step 3) in `_ARCHIVE_INDEX` so the `source`/`source_id` columns and `idx_archive_source_uid` partial unique index are created on fresh installs and applied on upgrade; also extended `ArchiveStore.record()` to accept `source` and `source_id` so `a2a_import` can write tagged rows
