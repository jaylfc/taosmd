### Added

- Claude Code capture hooks: `taosmd hooks run <event>` reads hook JSON from stdin and syncs the session transcript into the archive via `ingest_batch`; `taosmd hooks sync [--session ID | --all]` runs the same sync by hand.
- New `taosmd/hooks/` package with per-session cursor table (`capture-cursors.db`), transcript parsing (user/assistant messages, tool_use/tool_result entries, skipped types), stable item ids (`claude-code:<session_id>:<uuid>` with sha256 fallback), and 5-second hard timeout.
- Scrubbed Claude Code transcript fixture at `tests/fixtures/claude-code-transcript.jsonl` for parser validation.
