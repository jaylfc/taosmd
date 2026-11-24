### Fixed
- CLI now warns about legacy `./data` stores from v0.4.0 installs when using the default data dir. The warning lists all found stores (vector-memory.db, knowledge-graph.db, claims.db, archive-index.db, archive/, agent-memory/, agents.json) and shows the exact `taosmd migrate --import-legacy ./data` command for copy-then-verify migration. Never auto-moves data.
