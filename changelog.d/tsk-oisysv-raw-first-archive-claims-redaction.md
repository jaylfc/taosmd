### Fixed

- `ingest` and `ingest_batch` now archive every item byte-verbatim (no strip, no empty-skip); empty/whitespace items are still archived but not embedded or claimed. `ingest` returns `archived` count including empties; `ingest_batch` returns `ingested` (archived AND indexed) and `skipped` (empty or duplicate id) with all items archived zero-loss.
- Claims extraction now receives redacted text (via `taosmd.secret_filter.redact_secrets`) so secrets never land in `claims.db`.
