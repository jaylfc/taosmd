### Fixed

- All `int(limit)` parse sites in `taosmd/http_server.py` now reject negative limits with HTTP 400 and pin an upper ceiling so callers cannot request unbounded result sets. The two handlers missed by the prior PR (#482), `_handle_a2a_inbox` and `_handle_a2a_inbox_unhandled`, are included. `/tasks/edges` now returns 400 for `limit=-1` to match the project-wide negative-limit convention.
