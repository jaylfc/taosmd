### Fixed
- Fixed SQLite WAL journal mode race condition by moving busy_timeout before journal_mode and adding retry loop for transient lock errors during WAL initialization