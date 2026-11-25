### Fixed

- `effective_fanout` and sibling registry helpers now pass `data_dir` through to the served agent registry; non-default served data dirs copy a stranded default `~/.taosmd/agents.json` once on first use, warn when both registries exist, and agent CLI commands print the registry path they edit.
