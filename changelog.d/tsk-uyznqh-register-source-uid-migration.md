### Fixed

- Registered the `archive_index_source_uid` migration (step 3) in `_ARCHIVE_INDEX` so the `source`/`source_id` columns and `idx_archive_source_uid` index are actually created on fresh installs and applied during upgrades; previously the migration existed but was never reached, causing `a2a_import` to raise `OperationalError: no such column: source_id` on every first call