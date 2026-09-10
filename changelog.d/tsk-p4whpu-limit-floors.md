### Fixed

- Nine GET handlers (`/search`, `/graph`, `/graph/activations`, `/pending`, `/a2a/mentions`, `/a2a/threads/{thread}/messages`, `/tasks`, `/tasks/ready`, `/tasks/edges`) now reject a negative `limit` with HTTP 400 and cap positive limits. Previously `limit=-1` flowed straight to SQLite `LIMIT ?`, where -1 means unbounded. The previously inert `_A2A_MSG_MAX_LIMIT` constant is now wired into the two touched A2A handlers. `limit=0` returns zero rows as documented.
