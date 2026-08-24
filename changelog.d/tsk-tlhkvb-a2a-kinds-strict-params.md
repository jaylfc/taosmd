### Added

- `POST /a2a/send` accepts `kind: chat|alarm|ack|digest|receipt|review|system` (default `chat`), stores it on the envelope, and returns it in every read path. Unknown kinds are rejected with 400 naming the allowed set.
- One-shot migration `a2a_migrate_kinds()` backfills `kind` on historical A2A messages by body-prefix convention (`[AUTOMATED` -> alarm, `[AUTO-ACK]` -> ack, `[REVIEW]` -> review, else chat). Idempotent: re-running yields zero migrated rows.
- Every `/a2a` GET endpoint now rejects unknown query parameters with 400 listing the valid ones, generalising the existing `since` < 1e9 guard to close the silent-tolerance defect class.
