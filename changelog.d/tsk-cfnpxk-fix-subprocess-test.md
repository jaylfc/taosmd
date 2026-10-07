### Fixed
- `test_do_fire_runs_as_subprocess` now forces `a2a_send` failure via `TAOSMD_SERVER_URL=http://invalid.invalid` so the fallback log assertion is reachable; previously the local archive path succeeded in the sandboxed HOME, leaving the log unwritten and the test red.
