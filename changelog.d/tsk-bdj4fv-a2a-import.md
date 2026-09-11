### Added

- `POST /a2a/import` for idempotent batch import of external chat envelopes onto the A2A bus, with issuer-pinned registry auth, batch length cap, and import dedup.

### Fixed

- `/a2a/import` auth path now routes through `_registry_verifier.authorize()` so the token issuer is pinned, matching `/a2a/send` and `/a2a/mentions` on every check.
