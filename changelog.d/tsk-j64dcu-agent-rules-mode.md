### Added

- `agent_rules(mode, search_tool, write_tool)` in `taosmd/__init__.py` now accepts `mode` in {None, "taosmd", "both", "framework"} and tool-name overrides; two new package-data files `docs/agent-rules-mode-taosmd.md` and `docs/agent-rules-mode-both.md` provide the per-mode supplements with `{search_tool}`/`{write_tool}` placeholders.