# Codebase indexing and collections: design proposal

**Workstream:** the collections feature (feed folders, docs, and codebases to a named container agents can query) plus the taOS integration surface for it. This doubles as the integration contract for the taOS Projects app, in the same spirit as [INTEGRATION-memory-config.md](../INTEGRATION-memory-config.md): section 4 is the part @taOS-dev builds against.
**Status:** proposal for review. Nothing here is implemented and nothing should be until the phasing and the open questions in section 8 are agreed.
**Starting state:** verified against master by a fresh code survey, cited inline in section 1.

## 1. Problem and evidence

Codebase indexing is table stakes for coding agents now. Kilo Code ships it, Cursor ships it, the graphify-style tools ship it, and our own `taosmd/project.py` docstring already names Kilo Code as a framework we expect on the other end. taOSmd's audience is agents working on user codebases from low-end and multi-machine setups, and the hive-memory direction is one shared memory server that every agent and coding tool on the network connects to. A memory server for coding agents that cannot index the code it is helping with is missing the most-requested surface.

What exists today, verified on master:

- **No folder or file ingest anywhere.** `POST /ingest` and `POST /ingest/batch`, MCP `memory_ingest`, and `api.ingest` / `api.ingest_batch` are all text/turn-shaped. The CLI has no ingest command at all.
- **`taosmd/loaders/` exists but is unwired.** It is a cognee-style `LoaderInterface` ABC with typed blobs (`ChatBlob` / `TranscriptBlob` / `EmailBlob` / `DocBlob`), a `pick_loader()` registry with extension and MIME picking plus a `register_loader()` extension point, and `_safety.py` guards (`check_size`, `resolve_within` path containment). Only tests call it. `BlobType` already comments `code` as a future kind.
- **The keying primitive is built.** `taosmd/project.py` `ProjectResolver` / `get_project_id()` derives a stable project id from the git remote origin (normalized, sha256, 12 hex), falling back to `.taosmd/project.toml` then a cwd hash.
- **The dedup substrate is built.** `api.ingest_batch` dedups on a caller-supplied stable id (a content hash) and preserves per-item metadata verbatim. That is exactly what incremental re-index needs: hash the file, skip unchanged.
- **Scoping exists but is tag-shaped.** Shelves (agent registrations, admin-token-gated create/archive/unarchive) and projects (git-fingerprint metadata tags filtered at query time) both exist, with discovery surfaces: `GET /projects`, `GET /shelves?project=`, CLI `projects` / `shelves`, MCP `memory_list_projects` / `memory_list_shelves`.
- **The write path would mangle code.** `vector_memory.add` embeds the whole text as one row; there is no splitter anywhere. The extractor splits sentences and emits conversational fact-triples. Feed it a Python file and it will produce garbage triples from docstrings and mangled fragments of code. Code needs its own path with an extraction bypass.
- **Reindex rebuilds from the archive, not from files.** `reindex` / `reconcile` re-embed archive rows; they have no concept of a source directory to re-walk.
- **Zero-loss is a hard rule.** The archive is append-only, delete is a reversible alias for archive, only wipe destroys. Collections must inherit this, not route around it.

So the pieces are lying on the bench: loaders, safety guards, project keying, hash-dedup batch ingest. What is missing is the container concept, the folder walker, the code path, and the API.

## 2. The concept: collections

A **collection** is a named, typed container holding indexed content from one source. Concretely:

- `kind`: `codebase` | `docs` | `mixed`.
- `source`: a folder path or a git repo checkout. When the source is a git repo, the collection is keyed by the `ProjectResolver` fingerprint so every agent and machine that opens the same repo resolves to the same collection.
- First-class rows, not metadata tags: `id`, `name`, `kind`, `source_path`, `project_id?` (git fingerprint when applicable), `created`, `last_indexed`, `status` (`empty` / `indexing` / `ready` / `error`), and `stats` (file count, chunk count, errors).

The contrast with today's scoping matters. Shelves and projects are tags on conversation rows: they scope memories an agent wrote. A collection is a container for content that was indexed from a source, with its own lifecycle (create, index, re-index, archive) independent of any conversation. The two compose rather than compete: a collection **grant** gives an agent (or everything on a shelf) query access to the collection, and a collection **link** attaches it to one or more projects so project-scoped discovery finds it. A collection can serve multiple projects; a monorepo's docs collection linked to three downstream projects is the obvious case.

## 3. Ingest pipeline, phased

The build plan is deliberately staircase-shaped: each phase ships something usable and the later phases are droppable without stranding the earlier ones.

### Phase 1: wire what exists (docs collections, no new parsing tech)

- A gitignore-aware folder walker, `ingest_folder`, using the existing `_safety.py` guards: `resolve_within` for path containment against the collection's `source_path`, `check_size` per file.
- Each file goes through `pick_loader()`; `DocLoader` already handles md/txt/rst. Unhandled extensions are skipped and counted in `stats.errors`, not fatal.
- Per-file content-hash ids into the existing `ingest_batch`, which gives dedup and metadata preservation for free.
- Every chunk carries `collection_id`, `file_path` (relative to source root), and the content hash in metadata.

That is a complete docs-collection MVP: point it at a repo's docs folder and query it. No new dependencies, no new parsing, roughly a walker plus a table plus the API skeleton.

### Phase 2: the code path

- `CodeLoader` and `BlobType.CODE`.
- Language-aware chunking. Two candidates: tree-sitter (correct symbol boundaries for every language, but a real native dependency with per-language grammars, which is a poor fit for the Pi-tier dep matrix), or a zero-dep regex/indentation function-boundary splitter (top-level `def` / `class` / `function` / brace-block heuristics, imperfect on edge cases but dependency-free). taOSmd's minimalism ladder says start zero-dep with an `# upgrade-path: tree-sitter` comment and let the Phase-2 eval (section 5) tell us whether boundary quality is actually the binding constraint. My lean is the zero-dep splitter first; I would only take the tree-sitter dep if the eval shows chunk boundaries are what is costing recall.
- Every code chunk carries `file_path`, `symbol` (best-effort from the splitter), `line_start` / `line_end`, and `language` in metadata.
- **Extraction bypass:** code chunks skip the sentence splitter and the fact-triple extractor entirely. They are embedded as chunks, archived as chunks, and never enter the conversational KG. This is a routing decision at ingest, keyed on `BlobType.CODE`, not a new pipeline.

### Phase 3: incremental re-index

- `POST /collections/{id}/index` on an already-indexed collection re-walks the source: mtime pre-filter, then content hash. Unchanged files are skipped by the existing `ingest_batch` dedup. Changed files get their new chunks ingested and their old chunks **superseded** (the existing `supersede` machinery in the vector layer), never mutated or destroyed. Deleted files supersede too. Zero-loss holds: the archive keeps every version, reindex adds and supersedes, only wipe destroys.
- Where the source is a git repo, `git diff --name-only <last_indexed_commit>` is a cheaper walk; the hash walk is the fallback for non-git folders. Both produce the same supersede semantics.
- A filesystem watcher is explicitly later, if ever. Polling on demand (agent asks, or taOS triggers on project open) covers the real use without a daemon.

## 4. The API contract (what @taOS-dev builds against)

Same conventions as the existing surface: bearer token on the data plane, the dedicated admin token on admin routes (`_is_admin_route` / `_check_admin_token` in `http_server.py`), JSON in and out, errors as `{"error": ...}` with 400/404.

### Endpoints

```
POST   /collections                        (admin) {"name", "kind", "source_path"?, "project_id"?}
                                           -> {"collection": {...}, "created": true}
GET    /collections [?project=<id>]        -> {"collections": [ {id, name, kind, source_path,
                                              project_id, status, stats, last_indexed, links, grants} ]}
GET    /collections/{id}                   -> {"collection": {...}} with full stats
                                              (file_count, chunk_count, last_indexed, errors: [...])
POST   /collections/{id}/index             (admin) -> {"status": "indexing", "job": "<id>"}
                                              async; poll GET /collections/{id} until status
                                              is "ready" or "error"; stats update live
POST   /collections/{id}/link              {"type": "taos" | "git", "id": "<project id>"}
POST   /collections/{id}/unlink            same body; metadata only, never touches content
POST   /collections/{id}/grants            {"agent": "<name>"} -> grant query access
DELETE /collections/{id}/grants/{agent}    -> revoke
DELETE /collections/{id}                   (admin) -> archive, reversible; content rows are
                                              hidden from query, nothing is destroyed.
                                              Destruction only via the existing wipe surface.
```

Creation and indexing are admin-token-gated. I weighed leaving creation on the data plane for convenience, but indexing reads the server's filesystem at a caller-supplied path; that is an operator decision, not an agent decision, and it should sit behind the same token that gates shelf creation. Grants and link/unlink are metadata and stay on the data plane.

**Allowed roots.** The server config gains `collections.allowed_roots` (a list of directories). `source_path` must resolve inside one of them via `resolve_within`, checked at create time and again at every index. No allowed roots configured means collection creation is off. This is the single most important line in the contract from a safety standpoint.

### The two project ids

The taOS project id (`prj-xxx`) and the taOSmd git-fingerprint `project_id` (12 hex) are different namespaces and the link table must not pretend otherwise. A link row is `{type: "taos" | "git", id: "..."}`; a collection carries a list of them. `GET /collections?project=` matches against either type. taOS always links with `type: "taos"`; agents resolving via `get_project_id()` land on `type: "git"`. A collection for a repo that taOS also manages will typically carry one of each, and that is fine and expected.

### Query

- `POST /search` (and `GET /search`) gain an optional `collection` (single id) or `collections` (list) parameter. When present, hits from those collections are included alongside conversation memory; a `collections_only: true` flag restricts to them. Grants are enforced per requesting agent.
- Result rows that come from a code collection carry `file_path`, `symbol`, `line_start` / `line_end`, and `language` in their metadata, so a coding agent can jump straight to the source. Docs-collection rows carry `file_path`.
- MCP: a new `memory_list_collections` tool, and a `collection` parameter on `memory_search`, mirroring how `project` works today.
- CLI: `taosmd collections list | create | index | link | unlink | grant`, following the existing subcommand style (`shelves`, `projects`, `tasks`).

### Ownership split with taOS

- **taosmd owns:** the collection rows, the walker and loaders, chunking, embedding, supersede semantics, grants enforcement, the whole API above, and the standalone dashboard's minimal view of it.
- **taOS owns:** the Collections panel in the taOS Projects app: list collections, create-from-path (a folder picker constrained to the allowed roots the server reports), link/unlink to `prj-*` ids, index trigger with status/progress polling, and grants management per agent. taOS renders against the endpoints above and adds no semantics of its own; if the panel needs a field the API does not return, that is a contract change here first.

## 5. Evaluation plan (pre-registered, before build completion ships)

Retrieval quality on a real repo, judged-free, before any default surface ships. The self-repo is the honest test bed: index taosmd itself.

- **Query set:** 30 to 50 questions with known file/symbol answers, written before the index is built. Mix of "where is X implemented" (function/class lookup), "how does Y work" (concept spread over a file), and "what handles Z" (routing/dispatch questions). Gold labels are file paths, optionally symbols.
- **Metric:** Recall@5 on file-level hits. A hit counts if any of the top-5 results' `file_path` matches a gold file. No LLM judge anywhere in the loop.
- **Kill criterion:** ship default surfaces only if all three hold: file-level Recall@5 >= 0.8 on the self-repo set, AND indexing the taosmd repo completes in under 5 minutes on the reference tier, AND zero-loss verified end to end (reindex after an edit supersedes the old chunks and destroys nothing; the archive replays both versions). Miss any leg and the feature stays behind a flag with the miss recorded in the research report.
- Phase 1 gets the same treatment with a smaller docs-only question set over `docs/` before Phase 2 starts, so the walker and container plumbing are validated independently of the code splitter.

## 6. Costs and risks

- **Filesystem access from the server process.** This is the big one. The mitigations are already stated in the contract: explicit `allowed_roots` (off by default), `resolve_within` containment at create and at every index, `check_size` per file, and admin-token gating on create and index. Symlinks that escape the root are rejected by `resolve_within` by construction.
- **Index size on low-end tiers.** A mid-size repo is thousands of chunks. `binary_quant` is recall-neutral at 32x smaller vectors on conversational text (the CLAG/binq experiment); whether that neutrality holds for code chunks is unmeasured. Plan: default `binary_quant` on for `codebase` collections, but verify it inside the section-5 eval (one arm quantized, one not) before the default is real. If code recall drops, the default flips off and the report says so.
- **Embedder fit for code.** `arctic-embed-s` is text-tuned. It will be serviceable on code (identifiers and docstrings carry a lot of the signal) but a code-tuned small embedder is plausibly a real win. That is a follow-up bake-off in the E-010 style, per-collection embedder choice as the mechanism; it must not block Phase 1, and the section-5 kill bar is set assuming the text embedder.
- **Complexity versus minimalism.** This is the largest surface addition since shelves. The phasing is the mitigation: Phase 1 is a walker plus a table over existing parts, and each later phase has to justify itself against a pre-registered bar. The tree-sitter decision is the sharpest single complexity call and it is deferred until the eval says boundaries matter.
- **Write amplification on re-index.** Supersede-not-mutate means edited files accumulate archive history. That is the zero-loss deal working as intended, but stats should surface superseded-chunk counts so an operator can see the growth, and wipe remains the pressure valve.

## 7. What this does not do

- No LSP, no jump-to-definition service, no call-graph or symbol-graph KG. A graphify-style symbol graph over indexed code is a plausible Phase 4 and is explicitly out of scope here.
- No write-path changes to conversation memory. The extractor, the KG, verification, and every existing ingest surface are untouched; code chunks bypass them rather than modify them.
- No new defaults on existing behavior. Every current benchmark number stands; collections are additive and off until created.
- No filesystem watcher daemon in any phase here.
- No remote-source fetching (git clone by URL, web crawling). Sources are local paths inside allowed roots, full stop.

## 8. Open questions for Jay

1. **tree-sitter, yes or no?** My lean is the zero-dep splitter with `# upgrade-path:` and let the eval decide, but if you already know we will want tree-sitter for the Pi-excluded tiers, taking the dep once at Phase 2 is cheaper than migrating chunk boundaries later.
2. **Admin gating on creation.** I gated create and index behind the admin token because indexing reads the server filesystem. The alternative is data-plane creation constrained to allowed roots. I think the token is right, but it does make the taOS panel hold the admin token, which is worth a deliberate yes.
3. **Ship Phase 1 alone first?** Docs collections are useful on their own (index a project's docs folder, grant it to the agents). Shipping it before the code path exists gets real usage feedback on the container model early. My lean is yes.
4. **Code-embedder bake-off timing.** Follow-up experiment after Phase 2 lands, or run it in parallel with Phase 2 so the per-collection embedder mechanism is designed in from the start?
5. **taOS UI ownership.** Section 4 puts the Collections panel in the taOS Projects app and keeps only a minimal view in the standalone dashboard. If you want the standalone dashboard to be feature-complete for non-taOS users, that roughly doubles the UI work and should be scoped now, not discovered later.
