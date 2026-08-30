### Added

- `GET /tasks/edges` endpoint that lists task edges with optional `from_id`, `to_id`, `type`, `project`, and `limit` filters; the `project` parameter scopes results to edges whose both endpoints belong to the named project via two ANDed EXISTS clauses, and the limit is capped at 500.
