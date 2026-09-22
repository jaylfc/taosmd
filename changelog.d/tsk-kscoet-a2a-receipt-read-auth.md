### Fixed

- Gate `GET /a2a/messages/{id}/receipts` and `GET /a2a/receipts` on a verified registry identity. Unauthenticated or forged bearer tokens now return 401 when a registry verifier is configured; standalone mode (no verifier) continues to allow reads without a token.
