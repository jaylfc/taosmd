### Added

- `GET /tasks/edges` endpoint for reading active task edges with optional
  `from_id`, `to_id`, `type`, and `limit` filters.

### Fixed

- Cross-project read leak on the remote path: `GET /tasks/edges` now forwards
  the token-bound `project` through `service.task_list_edges` to
  `remote.task_list_edges` (which previously dropped it), so a project-bound
  token reading via a shared-server deployment can no longer see sibling
  project edges.
- Unbounded `limit` on `GET /tasks/edges`: negative values are rejected with
  400 and the upper bound is capped at 500.
