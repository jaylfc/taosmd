### Fixed

- `GET /search?limit=-1`, `GET /graph?limit=-1`, `GET /graph/activations?limit=-1`, `GET /pending?limit=-1`, `GET /tasks?limit=-1`, `GET /tasks/ready?limit=-1`, `GET /tasks/edges?limit=-1`, `GET /a2a/messages?limit=-1`, `GET /a2a/mentions?limit=-1`, `GET /a2a/inbox?limit=-1`, `GET /a2a/inbox/unhandled?limit=-1`, `GET /a2a/threads/{thread}/messages?limit=-1` now return 400 instead of returning all results. SQLite treats `LIMIT -1` as unbounded, so negative limits are rejected with `_BadRequest`.

- `GET /tasks/edges` negative limit now returns 400 instead of flooring to 1 row (`max(1, min(limit_i, 500))` changed to `min(limit_i, 500)` with negative check first).

### Added

- Limit ceiling constants now enforce maximum results: `_SEARCH_MAX_LIMIT=100`, `_GRAPH_MAX_LIMIT=300`, `_GRAPH_ACTIVATIONS_MAX_LIMIT=100`, `_PENDING_MAX_LIMIT=20`, `_TASK_LIST_MAX_LIMIT=50`, `_TASK_READY_MAX_LIMIT=20`, `_A2A_MESSAGES_MAX_LIMIT=50`, `_A2A_MENTIONS_MAX_LIMIT=50`, `_A2A_INBOX_MAX_LIMIT=1000`, `_A2A_THREAD_MESSAGES_MAX_LIMIT=200`. Requests exceeding these caps are clamped server-side.

- Limit ceiling tests added for `/search`, `/pending`, `/a2a/inbox`, `/a2a/inbox/unhandled` (in test_http_server.py and test_a2a_inbox_auth.py).
