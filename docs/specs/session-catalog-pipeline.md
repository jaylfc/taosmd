# Session Catalog Pipeline

## Goal

Add a timeline directory layer to taOSmd that processes raw zero-loss archive files into structured session records with LLM-generated descriptions, per-session split files for fast loading, and integration with the crystallize pipeline. The archive files are never modified — the catalog is a derived, regenerable index.

## Architecture

Three-stage event-driven pipeline, each stage idempotent and distributable across the taOS worker cluster:

```
Archive JSONL (read-only, source of truth)
  │
  ├─ Stage 1: Splitter (no LLM, instant)
  │    ├── Detect session boundaries (30-min gap heuristic)
  │    ├── Group events by agent_name (handle interleaved multi-agent archives)
  │    ├── Write session split files: data/sessions/YYYY/MM/DD/session-NNN-<category>-<slug>.jsonl
  │    └── Create catalog DB entries with line pointers to original archive
  │
  ├─ Stage 2: Enricher (LLM, tiered)
  │    ├── Tier 1: heuristic only (done by splitter — topic = first summary, category = "other")
  │    ├── Tier 2: qwen3-4b (Pi NPU, adequate quality)
  │    ├── Tier 3: qwen3.5:9b+ (GPU worker, best quality)
  │    ├── Generate: topic summary, description, category, sub-session boundaries
  │    ├── Auto-dispatch to best available hardware via worker cluster
  │    └── Update catalog DB entries in-place (idempotent — can re-enrich with better model)
  │
  └─ Stage 3: Crystallizer (LLM, builds on enricher output)
       ├── Takes enriched catalog session as input (reads from split file)
       ├── Generates narrative crystal digest + outcomes + lessons
       ├── Feeds lessons back into KG as triples
       └── Links crystal to catalog session via catalog_session_id FK
```

## Trigger

- **Daily cron at 3am**: indexes the previous day's archive through all three stages
- **Manual trigger**: via API (`POST /api/memory/catalog/index`) or taOSmd management app
- Only processes completed days by default. Force-indexing today creates entries with a `partial: true` flag that get reprocessed next day.

## Storage

### Catalog Database (`data/session-catalog.db`)

**sessions table:**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| date | TEXT | YYYY-MM-DD |
| start_time | REAL | Unix timestamp |
| end_time | REAL | Unix timestamp |
| topic | TEXT | LLM-generated or heuristic topic |
| description | TEXT | LLM-generated session description |
| category | TEXT | One of: coding, debugging, research, planning, conversation, configuration, deployment, testing, documentation, brainstorming, review, maintenance, other |
| archive_file | TEXT | Path to source archive JSONL |
| line_start | INTEGER | First line in archive for this session |
| line_end | INTEGER | Last line in archive for this session |
| split_file | TEXT | Path to derived session split file |
| turn_count | INTEGER | Number of events in this session |
| tier | INTEGER | Processing quality tier (1=heuristic, 2=small LLM, 3=large LLM) |
| partial | BOOLEAN | True if session is from an incomplete (today's) archive |
| created_at | REAL | When this catalog entry was created |

**sub_sessions table:**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_id | INTEGER FK | Parent session |
| start_time | REAL | Unix timestamp |
| end_time | REAL | Unix timestamp |
| topic | TEXT | Sub-session topic (LLM-detected topic shift) |
| description | TEXT | Brief description |
| line_start | INTEGER | First line in split file |
| line_end | INTEGER | Last line in split file |

**catalog_fts**: FTS5 virtual table over topic + description + category.

**catalog_meta**: Key/value for pipeline state tracking (last_indexed_date, last_enricher_model, etc).

### Session Split Files

```
data/sessions/           (gitignored, regenerable)
  2026/
    03/
      12/
        session-001-coding-taosmd-embeddings.jsonl
        session-002-research-mempalace.jsonl
```

- Each file contains raw JSONL events extracted from the archive for that session
- Filename: `session-NNN-<category>-<slug>.jsonl` (NNN is unique per day, prevents collisions)
- Entire `data/sessions/` is derived — delete and re-run splitter to regenerate

### Crystal Linkage

The existing `crystals` table in `crystallize.py` gets a new column: `catalog_session_id INTEGER` (FK to sessions.id). The crystallizer's entry point accepts a catalog session ID and reads from the split file instead of requiring raw conversation turns.

## Agent Isolation

Each agent has its own isolated memory instance — its own archive directory, its own catalog database, its own vector store. There is no multi-agent interleaving to handle. The splitter processes a single agent's archive at a time. The user's personal archive is a separate instance.

This matches taOS's architecture: agents run in isolated containers, memory lives on the host per-agent via `dbPath` routing, and frameworks are swappable components that don't touch the memory layer directly.

## Processing Tiers

| Tier | Model | Hardware | Quality | Speed |
|------|-------|----------|---------|-------|
| 1 | Heuristic (no LLM) | Any | Basic — gap detection, first-summary topic | Instant |
| 2 | qwen3-4b | Pi NPU / CPU | Adequate — good topics, categories | ~2s/session |
| 3 | qwen3.5:9b+ | GPU worker | Best — nuanced descriptions, accurate sub-sessions | ~5s/session |

Tier selection is automatic: the pipeline checks what hardware/models are available and uses the best option. Catalog entries track which tier processed them. Re-enriching with a higher tier overwrites the previous results.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/catalog/index` | POST | Trigger indexing for a date or date range |
| `/api/memory/catalog/date/{date}` | GET | Get all sessions for a date |
| `/api/memory/catalog/range` | GET | Sessions within a date range |
| `/api/memory/catalog/search` | GET | FTS search across session topics/descriptions |
| `/api/memory/catalog/session/{id}` | GET | Session detail + sub-sessions |
| `/api/memory/catalog/session/{id}/context` | GET | Session detail + raw archive lines for agent context |
| `/api/memory/catalog/stats` | GET | Catalog statistics |
| `/api/memory/catalog/rebuild` | POST | Drop and rebuild entire catalog |

## Integration Points

### Intent Classifier

New query type: `timeline`. Detection signals:
- Day/date references: "Tuesday", "March 12th", "yesterday"
- Activity questions: "what was I working on", "what happened", "show me sessions"
- Time-of-day: "morning", "afternoon", "evening"

In `thorough` memory strategy (default), the intent classifier is used for result **weighting** in RRF — timeline-classified queries get a boost on catalog results. In `fast` strategy, it's used for source **selection** — try catalog first, cascade if no match.

### Context Assembler

For timeline queries:
- L1 loads catalog session metadata (core facts tier — always present)
- L2 loads relevant session split file content (if catalog results rank high in RRF)
- L3 loads crystal narrative + linked KG facts for deep recall

### Crystallize Pipeline

`crystallize.py` gets a new method: `crystallize_from_catalog(session_id)` that:
1. Reads the session split file via the catalog's `split_file` path
2. Parses events into conversation turns
3. Runs the existing LLM crystallization
4. Stores the crystal with `catalog_session_id` FK

## Out of Scope (deferred to later specs)

- **Parallel fan-out retrieval / outer RRF** — spec 2
- **Cross-encoder reranking** — spec 2
- **Per-agent memory_strategy setting** — spec 2
- **taOSmd management app (frontend)** — spec 3
