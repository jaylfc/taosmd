### Fixed

- `GET /memories` now rejects a negative `limit` with HTTP 400 (previously `limit=-1` flowed straight to SQLite `LIMIT ?`, where -1 means unbounded, leaking the entire archive). `limit=0` returns zero rows as documented, and positive limits are capped at 500.
