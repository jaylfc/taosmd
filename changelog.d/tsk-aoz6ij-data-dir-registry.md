### Fixed

- `api.ingest` and `api.ingest_batch` now pass `data_dir=` to `ensure_agent` and `update_stats`,
  so the agent registry (`agents.json`) and agent-memory/ are written under the served data dir
  instead of `$HOME/.taosmd`. This resolves the registry/data split that caused backups to miss
  the registry and broke data-dir isolation for served installs.
