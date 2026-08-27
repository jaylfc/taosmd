### Fixed

- Corrected quoted error strings in `docs/verify-merged-assertions.md` and
  `tests/test_a2a.py` for the `test_http_a2a_blocks_without_body_returns_400`
  test. The service-layer `ValueError` message is `body must be a
  non-empty string` (from `taosmd/service.py:436`), not the HTTP handler's
  `'body' (non-empty string) is required` (from `taosmd/http_server.py:1546`).
