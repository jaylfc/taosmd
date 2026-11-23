### Fixed

- `service.stats` now correctly reads the agent registry from the served `data_dir` instead of the default `$HOME/.taosmd` location. The module-level `get_agent` function in `taosmd.agents` now accepts an optional `data_dir` parameter, matching the behavior of `ensure_agent` and `update_stats`.
