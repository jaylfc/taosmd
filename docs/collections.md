# Collections: feed folders to your agents

A collection is a named container of content indexed from one folder. Point
it at a repo's `docs/` (or any documentation folder inside an allowed root),
index it, grant agents access, and they can query it alongside their
conversation memory. Phase 1 indexes docs-shaped files (md, txt, markdown,
rst, plus anything another registered loader claims, such as `*.chat.json`);
the code path is Phase 2.

## Enable the feature (off by default)

Collections read the server's filesystem, so they are disabled until an
operator opts in by listing the directories collections may be created
under:

```json
// ~/.taosmd/config.json
{ "collections": { "allowed_roots": ["/srv/docs", "/home/jay/repos"] } }
```

Or via the environment: `TAOSMD_COLLECTIONS_ALLOWED_ROOTS=/srv/docs:/home/jay/repos`.
A collection's `source_path` must resolve inside one of these roots
(symlink escapes are rejected), checked at create time and again at every
index. An empty list means collections are off.

## Lifecycle

```bash
# create (admin operation over HTTP; local CLI works directly)
taosmd collections create --name "repo docs" --kind docs --source /srv/docs/myrepo

# index (walks the folder, chunks, embeds; re-run any time)
taosmd collections index col-ab12cd34ef56

# give an agent query access
taosmd collections grant col-ab12cd34ef56 my-agent

# attach to a project for discovery (taOS prj-* id or git fingerprint)
taosmd collections link col-ab12cd34ef56 --type git --id abc123def456

taosmd collections list
```

Statuses: `created -> indexing -> ready | error` (and `archived` after a
delete). Re-indexing is incremental: unchanged files are skipped by content
hash, changed and deleted files have their old rows superseded (hidden from
recall, never destroyed; the archive keeps every version). A file that still
exists but whose content has been emptied is retired the same way, and
reported separately as `files_emptied` rather than `files_deleted`, so a
blanked file is never mistaken for one that vanished from disk.

Index stats (on the collection row, and in the `GET /collections/{id}`
response) always carry the same keys:

```text
files_indexed      files whose content was chunked and ingested this pass
files_unchanged    files skipped because their content hash matched
files_deleted      previously-indexed files no longer present on disk
files_emptied      previously-indexed files still present but now empty
files_total        files currently tracked in the collection's hash state
chunks_ingested    chunks written this pass
chunks_skipped     chunks the batch deduped away
chunks_superseded  active rows retired this pass (never destroyed)
errors             up to 20 per-file failure strings
```

`files_deleted` and `files_emptied` count disjoint sets, and both are
present (as `0`) when nothing was retired.

The walker respects `.gitignore` files (simplified rules), skips VCS and
dependency directories, hidden directories, binaries, and oversized files.
A tree with more than 20000 ingestable files errors the index with a clear
message instead of grinding through it; point the collection at a smaller
folder (a `docs/` directory, not a monorepo root).

One index runs per collection at a time: starting an index while one is
already running returns `409` (poll `GET /collections/{id}` until the
status settles at `ready` or `error`, then retry).

## Querying

Search merges granted collections in with conversation memory:

```bash
curl -s localhost:7900/search -d '{
  "query": "how do I configure the widget?",
  "agent": "my-agent",
  "collections": ["col-ab12cd34ef56"],
  "limit": 5
}'
```

- `collections` (list) or `collection` (single id) adds collection content.
- `collections_only: true` restricts the search to the collections.
- Grants are enforced per requesting agent: a collection the agent holds no
  grant for contributes nothing (and its existence is not revealed).
- Collection hits carry `collection_id`, `file_path` (relative to the
  source root), and `source: "collection"` in their metadata.

Trust model: grants protect collections from ungranted *agents*, not from
holders of the server token. Grant and revoke live on the data plane, so
anyone presenting the server's bearer token can manage grants (and could
grant themselves access); collection access is therefore exactly as strong
as the server token. Treat the token as the security boundary and grants
as the per-agent scoping mechanism inside it. Only create/index/delete sit
behind the separate admin token, because those touch the server's
filesystem.

Over MCP: `memory_list_collections` lists them; `memory_search` takes a
`collection` parameter.

## HTTP surface

Data plane (bearer token when one is configured):

```text
GET    /collections [?project=<id>]
GET    /collections/{id}
POST   /collections/{id}/link      {"type": "taos"|"git", "id": "..."}
POST   /collections/{id}/unlink    same body
POST   /collections/{id}/grants    {"agent": "..."}
DELETE /collections/{id}/grants/{agent}
POST   /search                     with collection/collections/collections_only
```

Admin (dedicated admin token, fail-closed):

```text
POST   /collections                {"name", "kind", "source_path", "embedder"?}
POST   /collections/{id}/index     -> 202; poll GET /collections/{id}
DELETE /collections/{id}           -> archive (reversible)
```

The optional `embedder` field is stored and returned per collection (the
per-collection embedder mechanism); Phase 1 always indexes with the global
default embedder.

## Zero-loss guarantees

Delete archives, it never destroys: the collection row, its vector rows,
and its archive entries all stay on disk; archived collections simply stop
contributing to search. Re-index supersedes replaced rows with the same
`valid_to` machinery corrections use. Destruction remains exclusive to the
wipe surface.
