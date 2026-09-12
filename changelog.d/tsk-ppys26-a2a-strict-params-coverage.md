### Added

- Strict-param 400 tests for `GET /a2a/inbox`, `GET /a2a/inbox/unhandled`, `GET /a2a/threads/{thread}/messages` (blank-value case), and `GET /a2a/threads/{thread}/members`, covering both empty-valued (`?bogus=`) and non-empty (`?bogus=x`) unknown query parameters.
