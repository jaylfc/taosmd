### Fixed

- `GET /a2a/messages/{id}/receipts` and `GET /a2a/receipts` now reject unknown query parameters with HTTP 400, matching the strict-params contract documented for every other `GET /a2a/*` endpoint. Previously these two handlers never called `_validate_a2a_params`, so a misspelt parameter (e.g. `after=` or `since_id=` on the message-receipt path) was silently dropped and a typo like `sinc=` on `/a2a/receipts` was indistinguishable from a genuine 404 miss.
