### Fixed

- `taosmd/_db.py` raises `BUSY_TIMEOUT_MS` from 5000 to 30000, giving SQLite
  connections a longer window to wait out `database is locked` contention before
  raising. This prevents `test_concurrent_first_init_is_deterministic` from
  flaking under host load when multiple fork workers race first-time schema init
  on the same fresh database.
