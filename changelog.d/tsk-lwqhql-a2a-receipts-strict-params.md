### Fixed

- `GET /a2a/receipts` and `GET /a2a/messages/{id}/receipts` now reject unknown query parameters with `400` listing the allowed parameters, matching the other eight `/a2a` GET endpoints. A misspelled or renamed parameter previously failed open and returned a page instead of an error.
