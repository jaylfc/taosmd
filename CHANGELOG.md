# Changelog

## 0.3.0

First PyPI release. `pip install taosmd` now works (previous versions were
source-install only).

### Added
- **Project identity layer.** `get_project_id()` derives a deterministic project
  fingerprint from the normalized git remote origin URL, so different agents in
  the same repo share a project shelf without coordinating. Project-scoped
  storage tags memories with the project id; cross-agent reads are per-agent
  private by default and opt-in via `search(query, agent, project, also_include=[...])`.
  Librarian discovery adds `list_projects()` and `list_shelves()`. Wired across
  the Python API, HTTP server, MCP server, CLI, and remote client.
- **Realtime A2A wake.** `taosmd a2a-watch` streams new bus messages over SSE
  (one line each) for instant pickup; `taosmd a2a-bridge` runs a trigger command
  per new message with the message JSON on stdin, to wake a dormant session.
  Both reuse the `a2a-poll` cursor semantics: id-dedup (exactly-once across a
  reconnect) and client-side `--exclude`.
- **Dashboard Projects view** and an upgrade to Vite 8.

### Changed
- LoCoMo leader updated to MaxSim+rerank under a tri-judge methodology (lenient
  `gemma4:e2b` plus strict `llama3.1:8b` and `qwen3:4b-instruct-2507`); the old
  strict `qwen3:4b` judge is retired. See `docs/benchmarks.md`.
- Default server port unified to 7900 across code, scripts, and docs.

### Fixed
- `a2a-poll` writes its state file atomically (tmp + os.replace) and handles an
  unreachable bus gracefully (one-line error, non-zero exit, no traceback).
- `serve` closes its coroutine cleanly when the service loop stops.
- Packaging exports the project-identity surface and ships runtime data files
  (`docs`, `webui`, `skills`) in the wheel.

## 0.2.0

End-to-end memory system: 97.0% Judge accuracy on LongMemEval-S. MCP stdio
server, local HTTP/REST API (`taosmd serve`), reconcile, and remote client mode.

## 0.1.0

Initial release: 97.2% Recall@5 on LongMemEval-S.
