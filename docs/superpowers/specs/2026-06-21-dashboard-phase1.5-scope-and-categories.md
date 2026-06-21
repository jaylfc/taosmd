# Dashboard Phase 1.5: memory scope (user + per-agent) and semantic categories

Refines the Phase 1 Overview (PR #170). Two additions Jay asked for: viewing an individual agent's memory and the user's own memory, and real semantic categories rather than the agents/projects stand-in. Stays offline, accessible, and provable-memory-aware.

## A. Memory scope and the user namespace

### The user namespace (convention established now)
Reserve `user` as a dedicated namespace (the agent name `user`). It is the human's own memory space, distinct from each AI agent's working memory, and it is what the share-to-collect (Phase 3) and ambient-capture (Phase 6) lanes write into. This mirrors Memoire's "Your Data" vs "AI Assistants" split and is the home for the PiecesOS-style features. It is reserved now even though it is empty until those lanes land, so the architecture is right early.

### Scope selector and browse
- The Memory view gains a scope picker: `User` (the user namespace), one entry per registered agent, and `All` (aggregate across every namespace). The default is `All` until the user corpus fills.
- Within a scope, the view supports browse (a reverse-chronological list of recent memories) in addition to the existing search. A new read endpoint backs browse.
- The Overview reflects the selected scope: `All`/`User` shows the whole picture; picking an agent scopes the stat cards, growth, verification, categories, and activity to that agent.

### Backend
- `archive` already keys rows by `agent_name`; `user` is just a reserved value of it. No schema change.
- New `GET /memories?scope=&limit=` (scope is `all` | `user` | an agent name) returning recent `{text, agent, kind, ts, confidence}` rows for browse, via `api.list_memories(scope, limit, data_dir)` + a `service`/`remote` wrapper.
- `GET /stats?scope=` accepts an optional scope and filters the aggregation to that `agent_name` (default `all`). `dashboard_stats(scope=...)` adds a `WHERE agent_name = ?` to the archive helpers when scope is a specific agent or `user`.

## B. Semantic categories (graceful degradation)

### Taxonomy (fixed)
`Identity & Preferences`, `Activities & History`, `Work & Learning`, `People & Social`, `Communication & Content`, `Sources & Access`, plus `Uncategorized`.

### Two sources, chosen by setup
- **LLM-enriched setups** (a memory model / librarian generator is configured): each memory is classified into one category by the librarian at ingest (a single-label step folded into the existing enrichment, or a cheap dedicated call), stored as a `category` tag in the row metadata.
- **Non-enriched setups** (no generator, e.g. lite-pi): derive categories from the knowledge graph instead, mapping the top entity/predicate types into the buckets with a small keyword rule table. No LLM cost. This is the fallback.
- Detection: enriched if `config.get_memory_model()` (or the active recipe's generator) is set; otherwise fallback.

### Backend
- `taosmd/categories.py`: the fixed `CATEGORIES` list, a keyword rule table for the KG/non-enriched path, and `classify(text, *, enriched, kg_types=None) -> str` returning a category id. Pure and unit-testable.
- At ingest, when enriched, store the category on the memory metadata. (When not enriched, categories are computed at read time from the KG, so no write is needed.)
- `dashboard_stats` gains `categories: [{name, count}]`: counts of the stored category when enriched, else the KG-derived counts.

### Dashboard
- The Overview gets a `Top categories` card (Memoire-style horizontal bars), shown alongside the existing Top agents / Top projects (categories nested within the structure, not replacing it).
- The Memory view gains a category filter (chips) that narrows browse/search within the current scope.

## Phasing within 1.5
- **1.5a** (first): the scope model. The `user` namespace convention, the scope selector, the `GET /memories` browse endpoint, scoped `/stats`, and the Memory-view browse + per-agent/user views. This directly answers "view individual agents and the user's memory."
- **1.5b**: categories. `categories.py`, the enriched-vs-fallback classification, the `categories` field on `/stats`, the Overview card, and the Memory-view category filter.

## Testing
- Scope: `list_memories` filters by scope; `/memories` endpoint shape; scoped `/stats` reflects only the chosen agent.
- Categories: `classify` returns a valid category for enriched and fallback paths; the fallback never calls an LLM; `/stats` categories sum to the memory count (within Uncategorized).
- Frontend: the scope selector and category chips render and re-fetch; Playwright smoke in both themes.

## Out of scope
The share-to-collect and ambient-capture lanes that actually fill the `user` namespace (Phases 3 and 6). Phase 1.5 establishes the namespace and the views over whatever is there today.
