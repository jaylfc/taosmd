# Changelog

## Unreleased

`GET /version` with a capability list (new `taosmd.capabilities` module). The server now publishes what the running build actually supports, because neither a status code nor a version number could answer that. `taosmd serve` renders the dashboard SPA on unknown non-API paths, so `GET /collections` returns `200 text/html` on a build with no collections code, and an integrator who "verified" a route by checking for a 200 got a confident yes from a server that could not do the thing (this really happened, against the wrong service). Semver does not close the gap either: features land continuously between bumps, and a production box sat a month stale without anyone noticing even though `GET /health` already reported a version. `/version` returns `{"version", "commit", "commit_source", "built_at", "built_at_source", "capabilities"}` and `GET /health` gains the same `capabilities` list alongside its existing `status` and `version` keys, which are unchanged (taOS and the dashboard consume both). Both endpoints are unauthenticated by design, joining `/health` in `_PUBLIC_PATHS`, so monitoring and drift probes keep working on a token-secured box; they expose build identity and capability identifiers only (no paths, no tokens, no configuration). Capabilities are **stable contract identifiers with an explicit version suffix** (`collections.v1`, `grants.v1`, `temporal.v1`, `a2a.v1`, `tasks.v1`, `ingest.v1`, `search.v1`, `graph.v1`, `shelves.v1`), not feature names: a breaking change to a wire contract becomes `collections.v2`, so a client pinned to `collections.v1` sees the capability disappear (a visible break it can act on) rather than `collections` silently meaning something new; additive changes keep the identifier. The list is derived at request time by probing the running build (each identifier is declared next to the module and symbols that implement it, and is advertised only if they resolve), so deleting or renaming an implementation deletes the claim instead of leaving a stale boast, and a divergence test asserts every declared capability's routes exist in the real dispatcher. The commit sha is resolved once at first call and cached, never per request and never by shelling out: `git rev-parse` in a request path can block on a lock or a slow filesystem, so the git plumbing is read directly from the filesystem (`.git/HEAD` -> loose ref or `packed-refs`, including the `gitdir:` indirection used by worktrees and submodules), with an optional packaged `taosmd/_build_info.py` stamp taking precedence for wheel and container builds. Every step degrades to `null` rather than raising, so a pip install with no checkout and no stamp still gets a working endpoint.

Collections, Phase 1 (docs MVP, per `docs/specs/codebase-indexing-collections-design.md`): named containers of content indexed from a folder, queryable by granted agents alongside conversation memory. A collection is a first-class row (`created -> indexing -> ready | error`, plus reversible `archived`) with typed project links (`{type: taos|git, id}`, metadata only, never access-granting) and per-agent grants (`(canonical_id, scope='collection', collection_id)` unique rows, enforced at search time). Indexing wires the previously-unwired loader framework into a real ingest path: a gitignore-aware walker (stdlib rules; VCS/dependency/hidden dirs, binaries, oversized files, and symlink escapes skipped) feeds files a registered loader claims through a zero-dep paragraph chunker into `ingest_batch` under the collection's own agent namespace, with per-chunk content-hash ids so re-index dedups unchanged files; changed and deleted files have their old rows superseded (`valid_to` + marker), never deleted. The feature is off by default: the new `collections.allowed_roots` config list (or `TAOSMD_COLLECTIONS_ALLOWED_ROOTS`) must name the directories collections may index, and `source_path` is containment-checked (`resolve_within`) at create and at every index. Surfaces: HTTP (`POST /collections` and `POST /collections/{id}/index` admin-gated with async 202+poll indexing, `DELETE /collections/{id}` archives; list/get/link/unlink/grants on the data plane; `collection`/`collections`/`collections_only` on search), CLI (`taosmd collections list|create|index|link|unlink|grant|revoke`), and MCP (`memory_list_collections`, `collection` on `memory_search`). Collection hits carry `collection_id`/`file_path`/`source` metadata. A per-collection `embedder` field is stored and returned now (the mechanism for the code-embedder bake-off); Phase 1 always indexes with the global default. `benchmarks/collections_eval.py` pre-registers the file-level Recall@5 eval over the repo's own docs.

Admin token separation (#154, phase 1). Admin operations are now gated by a dedicated `admin_token`, distinct from the data-plane `server_token`. Previously the server token gated every data and A2A endpoint AND the admin surface, so on a token-less deployment the only way to authorize an admin op was to set a server token, which locked out every agent on the data plane for the duration of the admin window (this hit the Pi bus in production for about three minutes during a channel cleanup). Now the admin write routes (`POST /shelves`, `POST /shelves/{id}/archive|unarchive`, `POST /a2a/admin/delete-channel|rename-channel|supersede-message`) are exempt from the data-plane token gate and enforce the admin token themselves. Resolution prefers `admin_token` and falls back to `server_token`: existing token-secured installs keep working unchanged; setting only `admin_token` gates admin while leaving data and A2A endpoints open; with both set the data plane is gated by `server_token` and admin by `admin_token`, so a caller holding only the server token cannot run admin ops; with neither set the admin surface still fails closed (403). Configure via `admin_token` in config, the `TAOSMD_ADMIN_TOKEN` env var, or `taosmd config set-admin-token`. Phase 2 (isolating admin operations from the single service loop so a slow admin op cannot stall data reads/writes) is not part of this change and is tracked separately.

Scoping fix: `reindex` now carries the `project` scope and provenance through when it rebuilds an agent's vector rows from the archive. The rebuild loop stamped only `{"agent": agent}` plus role/timestamp, dropping the `project` tag, the `archive_span_id`, and the nested user metadata (`source_id`/`forget_after`). Losing the project tag was a cross-project scope leak: a project-scoped row came back project-untagged but still agent-tagged, so after a reindex it surfaced in a DIFFERENT project's search for the same agent. `reindex` now mirrors `reconcile`'s metadata reconstruction exactly, so a reindexed row is indistinguishable from a reconcile-repaired one in scope and provenance: it stays visible to its own project's search, stays invisible to other projects, keeps its archive-span linkage for the claims gate, and keeps its `source_id` so a re-POST of the same batch still dedupes. Zero-loss was never at risk (the archive is untouched); this is a scoping-correctness repair.

Concurrency fix: config and generator-profile writes are serialized through the service loop. `POST /controls` and `POST /generator-profile` (global and per-agent) read-modify-wrote the JSON config file directly on the request thread, so two overlapping POSTs could interleave their read and write and drop one update (last-writer-wins on the whole file). Both handlers now marshal the read-merge-write onto the single service loop so writes are applied one at a time and no concurrent update is lost.

Concurrency fix: `TemporalKnowledgeGraph.add_entity` performs its read-merge-write atomically under `BEGIN IMMEDIATE`. The non-destructive merge (keep first-seen name and concrete type, upgrade only the `unknown` placeholder, merge properties additively) read the existing row and wrote the merged row in separate statements, so two writers racing on the same entity could each read the pre-merge state and the second write could clobber the first's merge. The read and the write now run inside one immediate transaction, so concurrent `add_entity` calls on the same id serialize and every merge is preserved.

Docs: removed em dashes from prose and headings in `AGENTS.md` and `taosmd/docs/a2a-comms.md`, replacing them with commas, colons, or periods as reads best. Markdown table placeholder cells are left untouched.

Zero-loss fix: batch dedupe no longer breaks once an item's TTL expires. `existing_source_ids()` (the set that makes `POST /ingest/batch` idempotent) was built on `_load_active_rows`, which the earlier `forget_after` work taught to hide expired rows. So once a batch item's `forget_after` passed, its `source_id` dropped out of the dedup set and re-POSTing the same id re-ingested it, writing a second archive and vector row on every re-POST (unbounded duplication of hidden-but-present content). Dedup now reads all physically-present rows via a new `include_expired` path on `_load_active_rows`: a TTL-expired row is only hidden from recall, it still exists on disk, so its id still dedupes. Superseded rows (`valid_to` set) stay excluded from the set, so re-importing intentionally-cleared content still re-adds it, and recall-time TTL behavior is unchanged (search still hides expired rows).

Security fix: `POST /tasks` with `depends_on` now enforces the same project scope as the edge endpoints. Create was the one graph-mutating path that skipped `_enforce_edge_project_scope`: it applied token binding but then passed `depends_on` straight into `create_task`, which makes a blocks edge with no scope check. A project-scoped registry token could therefore create cross-project blocks edges and enumerate foreign task existence (a real id returned 200, a bogus id returned a 400 naming the missing task). When a token binds a project, each `depends_on` id is now checked to belong to that project before the task is created, returning the same non-enumerating 403 the edge endpoints use (foreign and nonexistent ids are indistinguishable). Tokenless and standalone behavior is unchanged.

Zero-loss fix: `TemporalKnowledgeGraph.add_entity` no longer silently overwrites an existing entity's type or properties. Because `add_triple` re-adds every subject/object on each write, the same entity is constantly re-inserted with whatever type the current extraction guessed (frequently the `unknown` placeholder, or a conflicting per-mention guess). The old `ON CONFLICT(id) DO UPDATE` was last-writer-wins, so a concrete classification (for example the `agent`/`lesson` types written by `crystallize`) was clobbered back to `unknown` by a later default-typed mention, and a re-added name flipped the display casing, both with no record. Triple relationships were already tombstoned via `valid_from`/`valid_to`, but entity attributes were mutated in place. `add_entity` is now non-destructive: it keeps the first-seen `name`, keeps the first-seen concrete `type` and only upgrades the `unknown` placeholder to a concrete type (enrichment, never a downgrade), merges `properties` additively with existing values winning on a key clash, and preserves the original `created_at`. Properties were always the `{}` default in practice (no caller passes them today), so that arm is forward-safety; the type and name overwrites were live. Deliberate re-classification with old-value history would need the triple-style temporal treatment, which no caller requires, so the lightweight merge is used rather than a heavier versioning table.

TTL fix: `forget_after` supplied via `ingest_batch` now actually expires. The retrieval-time TTL filter read `forget_after` only from the top level of a vector row's metadata, but `ingest_batch` nests the caller's per-item metadata under `meta["metadata"]` (alongside the top-level `agent` tag), so a `forget_after` passed through `POST /ingest/batch` or `ingest_batch(items=[{"metadata": {"forget_after": ...}}])` never hid anything and the row stayed visible forever. The filter now reads `forget_after` from both the top level and the nested user-metadata dict (top level wins if both are set), so expiry works identically whether the row came from a flat single `ingest` / low-level `add` or from batch ingest (including reconcile-repaired batch rows, which nest the same way). Zero-loss is unchanged: expired rows are only hidden from recall, never deleted. The prior batch TTL test hand-built a flat row shape that the real batch path never produces, which masked the bug; it now uses the true nested shape and a new end-to-end test drives `ingest_batch` directly.

Task update endpoint (`POST /tasks/{id}`) now applies the same token-bound project scoping as the edge endpoints: a registry token bound to a project can only mutate that project's tasks, with the same non-enumerating 403 for foreign and nonexistent ids.

Task edge endpoints now enforce token binding and project scoping. `POST /tasks/{id}/edges` and `POST /tasks/{id}/edges/remove` skipped `_apply_token_binding`, unlike every other `/tasks` handler, so with a registry and grants configured a verified token whose grant had been revoked could still mutate the task graph, and a project-scoped token could add or remove edges on tasks belonging to other projects. Both handlers now route through the same binding as task create/list/update (grant-less verified tokens get 403), and because `task_edges` has no project column, a token-bound project is enforced by requiring BOTH referenced task ids to belong to that project via the tasks table before the graph is touched. The cross-project refusal is a non-enumerating 403: a task in a foreign project and a task that does not exist return an identical response. Tokenless and standalone (no registry) behavior is unchanged.

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
