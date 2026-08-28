### Added

- `GET /tasks/edges` endpoint for reading active task edges with optional
  `from_id`, `to_id`, `type`, and `limit` filters.

### Fixed

- Cross-project read leak: a project-bound token now sees only edges whose
  both endpoints live in the token's project.
- Unbounded `limit` on `GET /tasks/edges`: negative values are rejected with
  400 and the upper bound is capped at 500.
