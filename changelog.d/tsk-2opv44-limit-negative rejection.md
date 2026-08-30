### Fixed

- `GET /a2a/messages?limit=-1` now returns 400 instead of returning the entire message feed. SQLite treats `LIMIT -1` as unbounded, so negative limits are rejected with `_BadRequest`.