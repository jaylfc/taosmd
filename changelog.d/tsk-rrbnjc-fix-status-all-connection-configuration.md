### Fixed

`taosmd/migrations.py:610` now uses `_db.connect()` to open SQLite connections consistently across all database reads, ensuring both `status_all()` and `migrate_all()` use the same database configuration (WAL mode with busy timeout). This fixes an inconsistency where one read path was silently opted out of the database configuration that every other path receives.