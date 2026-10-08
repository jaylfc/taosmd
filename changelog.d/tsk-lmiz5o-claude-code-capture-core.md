### Added
- Claude Code capture hook core: `taosmd.hooks` package with per-session cursor table (`capture-cursors.db`), a JSONL transcript parser matched against real Claude Code transcript shape, sync logic that reads by byte offset, and CLI commands `taosmd hooks run` and `taosmd hooks sync`.
