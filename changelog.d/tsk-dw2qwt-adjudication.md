## tsk-dw2qwt: Adjudicating PR #423

### Fixed

- FINDING A: Removed inert param validator from GET /a2a/inbox and /a2a/inbox/unhandled endpoints
  - Line 1896: Removed _validate_a2a_params call from _handle_a2a_inbox in taosmd/http_server.py
  - Line 1983: Removed _validate_a2a_params call from _handle_a2a_inbox_unhandled in taosmd/http_server.py
  - This ensures both new GET endpoints properly reject unknown query parameters as documented

- FINDING B: Standardized POST /a2a/inbox/advance and POST /a2a/ack auth fallback
  - Added ?consumer= query parameter fallback for both POST endpoints
  - Now matches behavior of their GET siblings (/a2a/inbox and /a2a/inbox/unhandled)
  - Endpoints now support both token-derived consumer and query-parameter consumer in standalone installs

### Documentation

- Updated taosmd/docs/a2a-comms.md to document the ?consumer= parameter for POST /a2a/inbox/advance and POST /a2a/ack
