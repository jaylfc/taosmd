# Changelog

## Unreleased

### Added
- **Bulk ingest with idempotent re-import.** `POST /ingest/batch` (and
  `taosmd.ingest_batch()` / `RemoteClient.ingest_batch()`) shelves a list of
  `{"text", "id"?, "metadata"?}` items in one call. Each item's `id` (the
  caller's stable content hash) is preserved as `source_id` and used to skip
  already-ingested items, so a migration batch can be re-POSTed safely.
  Returns `{"ingested", "skipped"}`. Built for the taOS user-memory
  unification (taOS#25): one-shot cutover of `user_memory_chunks` into a
  shared `user-memory` agent namespace.
- **BM25-only search mode.** `?mode=bm25` on `GET|POST /search` (and
  `mode="bm25"` on `taosmd.search()`) skips query embedding and recipe
  resolution entirely and returns BM25-ranked hits in the unchanged contract
  shape, with `confidence` carrying the sigmoid-normalised BM25 score. The
  search-as-you-type path for short-form user memory (sub-300ms SLA; the
  BM25 index reuses the fusion-path cache). Uses `bm25s` when installed and
  falls back to a dependency-free Okapi BM25 (same k1/b) when not.

### Fixed
- A benchmark run where every QA errored previously wrote an empty results
  file and exited 0; `locomo_runner.py` now exits 1 (and warns when failures
  outnumber successes).
- `taosmd a2a-poll` could silently drop messages older than the most recent
  500 after a long gap between polls. The state file now also keeps a
  per-channel timestamp cursor used to bound the fetch, with a warning if
  the window still overflows.
- `GrantsVerifier.has_grant` no longer raises (surfacing as a 500 on every
  A2A send) when a grant carries a non-numeric `expires_at`; unparseable
  expiries are treated as expired.
- `verifier_from_url` now pins the expected JWT issuer to the taOS registry
  by default; pass `expected_iss=None` to opt out explicitly.
- `make_server` warns when a registry verifier is configured without a
  grants verifier (identity checked but grants silently skipped).
- Cross-encoder reranking works under transformers 5.x again
  (`token_type_ids` are requested explicitly when the ONNX session needs
  them).

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
