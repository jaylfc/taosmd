# Changelog

## Unreleased

Zero-loss integrity fixes. `ingest` and `ingest_batch` no longer swallow vector-write failures: when the embedder is down every internal embed backend returns an empty vector and `VectorMemory.add()` returns -1, which both paths previously ignored, so turns landed in the archive but silently never became searchable. Both now report `vector_failures` and `degraded: true` in the result and log a WARNING naming the embed backend; they still do not raise on embed failure (sqlite and redaction errors propagate by design), because the archive write stands and `reconcile` can repair the gap once the embedder is back. `reconcile --repair` now re-adds rows with the same provenance and metadata the hot ingest path writes (the `project` scope, the `archive_span_id` linking the row to its archive span, and the nested user metadata carrying `source_id` and `forget_after`), so repaired rows stay visible to project-scoped search and the claims gate, and a re-POST of a repaired batch still dedupes instead of duplicating. The archive now rolls its open daily file at day boundaries instead of only when the year/month directory changes: previously a process running across midnight UTC kept appending to yesterday's JSONL while the index pointed at today's, corrupting file_path/line_number provenance. And `ingest_batch` now carries the same archive-span linkage and claims extraction as single `ingest`, so batch-ingested memories are no longer second-class for verification and the recall gate.

Provider prefix stripped at the model boundary. Generator-profile registry values are `provider:model` strings (`ollama:qwen3.5:9b`); the resolved string was previously sent verbatim as the model field to Ollama, which rejected it, and fact extraction silently degraded to regex. A new `generator_profiles.split_provider()` splits a leading known provider token (model names contain colons, so `qwen3.5:9b` passes through unchanged) and all three HTTP call paths (`memory_extractor` chat completions, `emem_event_lift` pass-2 lift, and the nightly `catalog_pipeline` enrichment/crystallize path) now send the bare model name; the prefixed form is kept for storage and CLI display. The regex fallback after an LLM failure now logs a WARNING (first failure per model and URL; repeats drop to debug) instead of a silent debug line.

Agent registry unified on the canonical data dir. `AgentRegistry` (and the CLI's `--data-dir` default) previously fell back to the CWD-relative `./data`, while the service layer resolved to `TAOSMD_DATA_DIR` / `~/.taosmd`, so ingest/search auto-registration and per-agent generator profiles landed in a split-brain `$CWD/data/agents.json` that the server never read. The default now resolves through the canonical resolver (`taosmd.config._resolve_data_dir`); explicit `data_dir` arguments are unchanged. If a legacy `./data/agents.json` exists it is never auto-moved; a WARNING (once per process, on every boot, for as long as the stranded file exists) tells you where to move it.

Task-aware generator profiles: the answer generator is selectable by workload, not just by hardware tier. A data-driven generator-profile registry maps a workload to the best model per tier (`balanced` = qwen3.5:9b, the multi-session and long-context default; `factual-recall` = gemma4:12b on a 12 GB GPU and llama3.1:8b on 8/4 GB, for single-fact retrieval QA), resolved with precedence pin > per-agent profile > global > recipe > retrieval-only. Select it from the `taosmd generator-profile` CLI or the dashboard Settings view (global or per agent). The 8 and 4 GB factual picks are confirmed by the E-023 low-tier benchmark.

A2A bus registry-auth verify-and-warn: the bus can verify registry-minted EdDSA-JWT agent identities (signature, sub matches the sender, and an active a2a_send grant) against the taOS agent registry. The default mode logs a warning and still accepts a post that fails the check, so the live multi-agent bus keeps working while agents migrate off self-asserted handles; an `a2a_auth_enforce` flag flips it to reject. Dormant until a registry URL is configured.

`taosmd reindex` for per-agent embedder cutover (PR #175): a single-shot command that clears an agent's vector rows and rebuilds them from the zero-loss archive, re-embedding every turn under the currently configured embedder (for example MiniLM to arctic-embed-s). `--agent` cuts over one agent at a time, `--check` dry-runs, and because the archive is never touched a reindex is always safe to re-run.

Judge verdict-parser fix in the LongMemEval harness (PR #176): the benchmark judge parser checked for the substring "CORRECT" after upper-casing, and "INCORRECT" contains "CORRECT", so every INCORRECT verdict had been scored as a pass since 2026-04-13. Published end-to-end Judge scores were corrected accordingly: on the full 500 with the shipped qwen3.5:9b generator, 74.6 percent became 42.8 (Qwen judge) / 51.2 (llama judge). The 97.0 percent Recall@5 headline is judge-free and unaffected.

## 0.4.0

Provable Memory and the memory cockpit. Adds the claims layer (facts carry their
archive-span provenance and are verified asynchronously against only those spans,
with a `prefer_verified` recall gate now on by default), the standalone
dashboard's memory cockpit (themed Home/Overview over `GET /stats`, a Memory
Explorer knowledge-graph galaxy over `GET /graph`, semantic categories, and a
time scrubber), a typed memory-controls registry with a `/controls` API and
dashboard Settings, and embedder hardening so a missing dense model fails loudly
at setup instead of silently degrading retrieval at serve time.

### Added
- **Dashboard memory cockpit.** The standalone web dashboard grows from a read-only
  inspector into a cockpit, themed to taOS's macOS dark and light schemes with a toggle.
  A **Home** overview reads a new `GET /stats` aggregate (memory growth, a
  verification-coverage donut over the claims rate, top categories, top agents and
  projects, and a recent-memory browse) with a scope selector across all memory, a
  reserved `user` namespace, and any individual agent (`GET /stats?scope=` plus a new
  `GET /memories`). A **Memory Explorer** tab renders the temporal knowledge graph as a
  bundled, offline `d3-force` galaxy over a new read-only `GET /graph`, with superseded
  facts faded and click-to-detail on each entity. Semantic categories
  (`taosmd/categories.py`) classify memories with a deterministic keyword classifier, and
  the librarian-LLM and knowledge-graph-type classifications are marked upgrade paths. All
  charts are hand-rolled SVG, every view is keyboard-navigable and ARIA-labelled, and the
  whole dashboard works fully offline with no CDN.
- **Memory controls (registry + `/controls` API + dashboard Settings + docs).** A typed
  controls registry (`taosmd/controls.py`) is the single source of truth for every
  user-facing lever (scope, type, default, cost, pros and cons), consumed by the
  standalone dashboard, a new `GET`/`POST /controls` HTTP API (current settings plus
  schema, the Minimal/Quality/Integrity presets, and validated per-control writes), the
  config store (`config.get_controls` / `set_control` / `get_runtime_overrides`), and a
  README "Configuration and controls" section with a drift-guard parity test. The runtime
  controls (`reranker`, `fusion`, `adjacent_turns`) now overlay the active recipe per
  query, so a dashboard change takes effect on the next search; `late_interaction` is
  documented as a store-scope choice (a re-index). The `prefer_verified` recall gate now
  ships on by default (tri-judge evidenced, E-018: served-hallucination 0.040 to 0.000 at
  no measured accuracy cost), a safe no-op until the verify-pass is populated.
- **Claims layer (Provable Memory), default-off.** A new additive `taosmd/claims/`
  package that extracts facts as claims carrying their archive-span provenance,
  verifies them asynchronously against only those source spans with a
  cross-family local-LLM entailment judge (fail-closed: an unparseable or failed
  verdict stays unverified, never promoted), and demotes (never deletes)
  unverified/unsupported claims from recall behind a `prefer_verified` flag on
  `search()` (default `off`, so existing behaviour is unchanged). New
  `ClaimStore` (zero-loss sqlite, rebuildable from the archive, exposes the live
  hallucination rate), `taosmd verify` and `taosmd claims status` CLI, and a
  store-mode gate that is a pure function. This turns F-009 (the measured 18.8
  percent extraction-hallucination rate) into a live always-on quality signal.
  The default flip is gated on the pre-registered E-009 experiment.
- **Temporal date-range lever (`retrieve(temporal={...})`), default off.** A
  deterministic natural-language temporal parser (`taosmd/temporal.py`) that
  turns expressions like "last week", "in May 2023", "Q1 2026", or "on 8 May,
  2023" into date ranges and applies them as a retrieval post-stage, either
  boosting in-range hits or filtering to the range (fail-open: hits without a
  parseable timestamp are kept, and a filter that would leave zero hits
  returns the original list). Handles epoch floats, ISO strings, and LoCoMo
  conversation timestamps; an optional reference time resolves relative
  expressions against a conversation rather than the wall clock. Complements
  the keyword-signal `temporal_boost.py` (ordering anchors); this lever is
  for explicit date ranges. Benchmark status is recorded honestly in
  docs/research-report.md E-005: LoCoMo cannot measure this lever (only 9.3
  percent of its temporal-category questions contain a parseable range
  expression), so no LoCoMo claim is made; the lever targets live agent
  queries ("what did we decide last week").
- **Pause/resume checkpointing for the LoCoMo runner.** New `--ckpt`,
  `--resume`, and `--pause-flag` flags with conversation-granularity sidecar
  checkpointing (`benchmarks/bench_checkpoint.py`): after each conversation,
  its rows are fsync-appended to `<out>.ckpt.jsonl`; resume verifies a config
  hash (never silently mixes configs), pre-loads completed rows, and skips
  finished conversations including their expensive ingest. Touching the pause
  flag file stops the run cleanly between conversations (exit 3) with resume
  instructions. Sidecar is removed on successful completion; final result
  JSON shape is unchanged.
- **Compact mode on `GET /a2a/messages` (`fields`, `format`).** Two optional
  query parameters for token-frugal bus consumers (LLM agents reading the
  A2A feed): `fields=<csv>` (e.g. `fields=id,from,body`) projects each
  message down to the named keys, with unknown names ignored so consumers
  stay forward-compatible; `format=ndjson` emits one JSON object per line
  (`application/x-ndjson`) instead of the wrapped `{"messages": [...]}`
  envelope. Both compose with the existing `thread`/`since`/`limit`
  parameters and default to the previous behaviour when absent. Built on a
  bus request from the hermes agent (lean non-MCP read path).
- **Living research report (`docs/research-report.md`).** First edition of the
  living technical report: abstract, methodology (external tri-judge protocol,
  hardware tiers, dataset provenance), full results tables with provenance to
  docs/benchmarks.md, first-class negative results section (N-001 through
  N-006), four pre-registered experiments (E-001 through E-004) with verbatim
  kill criteria, reproducibility commands, and a navigable finding index (F-001
  through F-008). README benchmark section links to it.
- **Retrieval-time TTL filter (`forget_after` / `forget_reason`).** Callers
  may include `forget_after` (unix float) and an optional `forget_reason`
  (string) in user metadata when calling `VectorMemory.add` or the HTTP
  `POST /ingest` and `POST /ingest/batch` endpoints. Once `forget_after`
  passes, the row is hidden from `search()` and `search_bm25()` exactly like
  a superseded row. The raw row is never deleted (zero-loss: the archive is
  untouched). Non-numeric or missing `forget_after` values are silently
  ignored so existing memories are unaffected. The feature requires no schema
  change; the fields live in the existing `metadata_json` column alongside
  other user metadata. Inspired by supermemory's `forgetAfter` concept;
  implemented as a zero-loss filter at retrieval time rather than a hard
  delete. `_load_active_rows` accepts an optional `now` parameter (defaults
  to `time.time()`) so tests can control the clock without monkey-patching.

- **Three-number bench summary in the LoCoMo runner.** The `_summary`
  aggregate and the `overall` block of the result JSON now carry
  `mean_latency_ms` (mean per-row retrieval + generation time),
  `p95_latency_ms`, and `mean_context_tokens` (mean context chars sent to
  the generator divided by 4). `_process_qa` stores `context_chars` per row
  so the estimate is exact. `_print_summary` displays the triple on its own
  line below the accuracy table. The existing accuracy columns (F1, BLEU-1,
  Judge, R@K) are preserved without modification. Inspired by supermemory's
  MemScore philosophy of never collapsing accuracy, latency, and context cost
  into a single number.

- **Admin surface: shelf lifecycle and A2A channel admin (taOS#774).** New
  `taosmd/admin.py` module and six HTTP endpoints gated behind the configured
  server token (fail-closed: 403 when no token is set, 401 on wrong token).
  Shelf lifecycle: `POST /shelves` creates or returns an existing shelf
  (idempotent by `shelf_id`; shelf_id must match `^[a-z][a-z0-9_-]{0,62}$`);
  `POST /shelves/{id}/archive` soft-hides the shelf's active vector rows by
  stamping `valid_to` and embedding a `hidden_by: shelf-archive:<ts>` marker
  in each row's metadata so `POST /shelves/{id}/unarchive` can restore exactly
  those rows and nothing else (rows superseded for other reasons such as
  corrections are not resurrected). `?expect_empty=true` returns 409 without
  archiving when the shelf has active rows. Archive and unarchive events are
  appended to the zero-loss archive. A2A channel admin: `POST
  /a2a/admin/delete-channel` soft-deletes a channel so it is hidden from
  `/a2a/channels` and `/a2a/messages` responses while messages remain in the
  archive; `POST /a2a/admin/rename-channel` adds a channel alias so sends to
  the old name are redirected and reads of the new name include history from
  the old name (stored rows are not mutated); `POST
  /a2a/admin/supersede-message` hides one message by id from feed responses.
  The deleted-channels set, alias map, and superseded-message set are persisted
  in `data/a2a-admin-state.json` via atomic tmp+os.replace writes.
- **Late-interaction retrieval lever and `lateint-9b` recipe.** VectorMemory
  now supports ColBERT-style token-level MaxSim scoring (`late_interaction=True`,
  `colbert_model="answerdotai/answerai-colbert-small-v1"`) as an opt-in
  retrieval mode. Each memory stores a full token matrix (seq_len x 384 float16)
  instead of a single pooled vector; retrieval scores each document by
  mean_q(max_t(q_t . d_t)) (ColBERT MaxSim). Two backends: the existing MiniLM
  ONNX path (backbone token vectors) and sentence-transformers (falls through to
  pylate when installed, for the trained projected space). A pylate loader is
  included for ColBERT models that apply a projection head; it auto-detects the
  output dimension and stores it per-instance in `_token_dim` so mixed-dim
  corpora fail loudly rather than silently writing garbage. The new `lateint-9b`
  recipe uses answerai-colbert-small-v1 with mem0_additive fusion, top_k 10,
  retrieval_top_k 20, adjacent_neighbors 2, and no reranker. Measured scores on
  full-1540 (tri-judge): gemma4:e2b 0.716, llama3.1:8b 0.388,
  qwen3:4b-instruct-2507 0.656, qwen3:14b 0.542. The backbone MaxSim path
  (this PR) replaces dense cosine with no reranker download required; the pylate
  projected-space comparison is queued separately. The `--late-interaction` and
  `--colbert-model` flags are wired into the LoCoMo runner for benchmarking.
- **Project-scoped grants (taOS#744).** Registry grant rows now carry an
  optional `project_id` field; `GrantsVerifier.has_grant` accepts a matching
  `project_id` keyword argument and a grant row with no `project_id` acts as
  a global grant that matches any project. Data endpoints (ingest, search,
  tasks) optionally bind the verified `project_id` claim from a Bearer token
  to the request's `project` field, so a caller cannot supply a different
  project than the one their token was minted for; the binding is
  token-optional and introduces no lockout for token-free requests.
- **Task graph component.** New `taosmd/tasks.py` module implements a
  dependency-aware task graph backed by SQLite (`tasks.db` under the data
  dir). Tasks have content-hash IDs (`t-<sha256[:12]>`) that are safe for
  concurrent multi-agent creation without coordination. A `task_edges` table
  expresses `blocks`, `parent`, `relates`, and `duplicates` relationships.
  Edges are soft-removed (never deleted). A `ready_tasks` SQL view derives
  the ready queue directly from the edge table: a task is ready when it is
  open and has no active blocking edge whose source task is still live.
  Every mutation is appended to the zero-loss archive before touching
  projection tables, so `rebuild_from_archive()` can replay the full history
  into fresh tables at any time.
  Exposed via: 7 HTTP endpoints (`POST /tasks`, `GET /tasks`, `GET
  /tasks/ready`, `GET /tasks/prime`, `POST /tasks/{id}`, `POST
  /tasks/{id}/edges`, `POST /tasks/{id}/edges/remove`); `taosmd tasks
  add|list|ready|prime|start|close|block` CLI subcommands; and 5 MCP tools
  (`task_add`, `task_ready`, `task_prime`, `task_update`, `task_edge`).
  The `GET /tasks/prime` endpoint (and `taosmd tasks prime` CLI) returns a
  token-budgeted session-bootstrap briefing covering ready, in-progress,
  blocked, and recently-closed tasks, suitable for direct injection into an
  agent system prompt.
  Concept credit: the dependency-graph, ready-queue, and prime ideas come
  from beads (github.com/gastownhall/beads); this is an independent minimal
  implementation for the taOSmd substrate.
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
- Setup now preflights the dense embedder so a missing model is loud, not
  silent. A recipe writes `embed_model` (for example `arctic-embed-s`) into
  config, but the ONNX files are fetched separately by `scripts/setup.sh`, so a
  provisioning path that skips that script left the model absent and the store
  silently fell back to a different embedder at serve time. Because a store's
  vectors are tied to the embedder that wrote them, that returned meaningless
  retrieval with no error. `auto_setup` gains a `_preflight_embedder_model`
  check (mirroring the enricher-model preflight) that verifies the recommended
  embedder's ONNX files are on disk and prints the exact fetch command when they
  are not, and `VectorMemory`'s ONNX load-failure warning now names the model
  directory, points at `scripts/setup.sh`, and flags the vector-space mismatch.
- The live serve path now builds the vector store in the storage mode its
  recipe asks for. `late_interaction`/`colbert_model` lived only in
  `recipes.py` and were never read at store construction, so an 8 GB GPU
  install that `recommend()` pointed at the `lateint-9b` recipe silently ran
  plain dense retrieval. `config.vector_memory` now carries the storage-format
  mode, `_ensure_stores` honours it, and `auto_setup` seeds it from the
  recommended recipe. A `store_meta` marker records the mode a store was built
  in; reopening it in a conflicting mode (dense vs late-interaction vs
  binary-quant) raises `StoreModeMismatch` with a re-embed-from-archive
  instruction rather than silently serving wrong-mode results. The archive is
  zero-loss, so a re-embed is always possible.
- The data-endpoint grants gate only fired when a Bearer token carried a
  `project_id` claim, so a verified GLOBAL token whose holder had no active
  grant could still write through `/ingest` and the other bound endpoints.
  The gate now fires for every verified token (project-bound or global);
  tokenless requests keep the no-lockout behaviour. Found by the taOS#744
  end-to-end exercise against the real registry feeds. Grant `expires_at`
  values in ISO-8601 (the registry's actual feed format) are now parsed
  properly instead of being dropped as unparseable.
- The stdlib fallback inspector page escapes quote characters, closing an
  attribute-context XSS path via A2A sender names rendered into the
  dashboard (the bundled React dashboard was never affected).
- Generation failures in the LoCoMo runner were stored as zero-scoring
  `[generation_error: ...]` predictions, letting a broken setup (for example
  a missing Ollama model) impersonate a catastrophically bad generator. They
  now count as failed QAs and an all-failed run aborts loudly.
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

End-to-end memory system: 97.0% Recall@5 on LongMemEval-S (corrected 2026-06-14; this was mislabelled "Judge accuracy" in earlier editions, the measurement is Recall@5). MCP stdio
server, local HTTP/REST API (`taosmd serve`), reconcile, and remote client mode.

## 0.1.0

Initial release: 97.2% Recall@5 on LongMemEval-S.
