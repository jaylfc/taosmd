### Fixed

- Document that in `GET /a2a/messages/{id}/receipts` and `GET /a2a/receipts` the auth gate runs before `_validate_a2a_params`. When a registry verifier is configured, an unauthenticated request with an unknown query parameter now returns 401 (not 400). Standalone mode (no verifier) is unchanged; behaviour is correct and not modified, only the write-up was missing.
