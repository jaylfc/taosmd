### Fixed

- One malformed embedding row (invalid JSON or dimension mismatch) no longer makes the vector store unopenable. During f32 migration, such rows are now skipped with a warning, logged by row ID, and counted in the migration result. All other rows are still converted. Readers already skip unreadable rows, matching v0.4.0 behaviour.
