### Fixed

- `GET /a2a/receipts` now binds the verified caller identity and rejects cross-agent reads with 403 unless the caller is the message sender. `GET /a2a/messages/{id}/receipts` returns all rows for the sender but filters to the caller's own row for other verified agents. Standalone mode (no registry verifier) is unchanged.
