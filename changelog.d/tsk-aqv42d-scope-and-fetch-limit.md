### Fixed

- `taosmd.retrieval.retrieve` no longer caps the per-source fetch limit at the agent fan-out times three when the caller asks for more hits than that ceiling. `fetch_limit` is now `max(limit * 3, result_limit)`, so a caller limit above the fan-out ceiling (fan-out 10, limit 50) returns the full 50 in-scope hits instead of 30. The fan-out still controls retrieval breadth when it is the binding constraint; only the caller's explicit limit can raise the ceiling above it.

### Added

- `tests/test_recipes_search.py::test_primary_vector_query_receives_project_scope` pins that the fast and minimal strategies' primary `_query_source(...)` branch passes the caller's `project` and `search_agents` to the vector source, instead of relying on the fallback branch to cover scope (the existing scope tests only reached the fallback).
- `tests/test_recipes_search.py::test_limit_above_fanout_ceiling_is_honoured` seeds 60 in-scope hits with fan-out 10 and limit 50 and asserts 50 come back.
