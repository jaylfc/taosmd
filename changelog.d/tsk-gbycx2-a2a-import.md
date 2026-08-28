### Added

- `POST /a2a/import`: idempotent batch import of external chat envelopes onto the A2A bus. Accepts a JSON array of envelope dicts, each requiring `source`, `source_id`, `from`, `body`, and `ts`. Deduped on the pair `(source, source_id)` so re-importing the same channel produces no duplicates. Original timestamps are preserved verbatim. The whole batch is rejected if any envelope is invalid, composes with exporter clients that pre-validate. Uses the same registry auth gate as `POST /a2a/send`.
