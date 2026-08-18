### Added

- `GET /a2a/mentions` returns messages that mention the authenticated reader plus their reply chains, with thread-scoped anti-bypass visibility (#211). `?reader=` is normalised before identity checks so `@bob`, `Bob`, and `bob` all round-trip correctly.
- `a2a_mentions_feed()` service function and `can_read()` anti-bypass guard.
- `MentionStore` index for @handle extraction from A2A bodies and the `recipient` field.
