# Librarian Design Spec — 2026-04-15

## Overview

This spec covers the second half of the Librarian system: the prompts and glue not yet on `feat/librarian-controls`. It assumes the reader has read `taosmd/prompts.py::LIBRARIAN_PERSONA`, `taosmd/agents.py`, `taosmd/session_catalog.py`, `taosmd/catalog_pipeline.py`, and `taosmd/intent_classifier.py` on that branch. The six-holdings model, the seven already-implemented prompts, and the `is_task_enabled()` flag surface are frozen. What is being decided here:

- The taxonomy the Librarian uses to file transcripts (project / topic / subtopic) and the prompt that assigns it.
- The verification and contradiction-check prompts and when they fire.
- The handoff between the regex `intent_classifier` and the LLM `routing_prompt`.
- The remaining prompt stubs (disambiguation, citation, redaction, cross-reference).
- The precise call site where `is_task_enabled()` gates each enrichment task.
- An A/B eval harness distinct from LongMemEval-S.
- What standalone callers (OpenClaw, Hermes, LangChain) need beyond `ingest()` / `search()`.

## Architecture Decisions

### AD-1. Classification runs before enrichment, not after.
The intake classifier is the first LLM call after split. It gates everything downstream — the hierarchy label it emits is what lets retrieval prefilter by project/topic before doing vector search. This is the inversion of the current pipeline (where category is emitted by `session_enrichment_prompt` as a free-text word). The new order inside `catalog_pipeline.index_day`:
`split → intake_classify → enrich → contradiction_check (per triple) → crystallize → reflect`.

### AD-2. The taxonomy is emergent, not hardcoded.
No fixed ontology. The Librarian maintains a per-agent `taxonomy.json` of seen `(project, topic, subtopic)` paths. The intake prompt is shown the current taxonomy and instructed to reuse existing paths when plausible and only coin a new path when no existing one fits. New paths have a pending/confirmed flag; after N sessions under the same new path it promotes to confirmed.

### AD-3. Multi-label classification is first-class.
A session can belong to multiple `(project, topic, subtopic)` triples with weights summing to 1.0. Storing `primary` as the top label and `labels[]` as the full list. Retrieval uses the full list for filtering; UI uses the primary.

### AD-4. `routing_prompt` is a fallback, not a replacement.
`intent_classifier.classify_intent()` is the default. The LLM router fires only when (a) regex scores tie below a threshold, (b) the intent classifier returns `EXPLORATORY` (the "I don't know" bucket), or (c) the caller explicitly asks for LLM routing. This keeps the hot path at microseconds.

### AD-5. Verification gates *return*, contradiction-check gates *file*.
`verification_prompt` runs at retrieval time against top-K candidates, used as a late reranker / confidence downweight. `contradiction_check_prompt` runs at ingest time before each Card Catalogue write. They share a JSON schema (`supports | contradicts | silent` + quote) so the implementations can share parsing.

### AD-6. Contradictions are never destructive.
When `contradiction_check_prompt` finds a conflict, the old card is superseded, not deleted. The supersede chain is already modeled in the KG. A new card inherits a `supersedes: <old_card_id>` edge and carries a `conflict_reason` from the check. Deletion is never a librarian action.

### AD-7. `is_task_enabled()` is checked at the orchestrator, not inside individual extractors.
Extractor functions stay pure (they don't know about agents). The pipeline orchestrator (`catalog_pipeline.py` and `retrieval.py`) wraps every LLM enrichment or extraction call in the flag check.

## Component Designs

### 1. `intake_classification_prompt` (issue #9) — the missing keystone

**Purpose.** First-pass triage. Assigns each session a ranked list of `(project, topic, subtopic)` labels before any other enrichment runs.

**Call site.** New stage between split and enrich in `CatalogPipeline.index_day`. Runs once per session after `split_day`, before `enrich_session`.

**Inputs.**
- `session_text` (first ~4000 chars from split file)
- `existing_taxonomy` (serialized tree of confirmed paths for this agent)
- `agent_name`

**Output schema.**
```json
{
  "labels": [
    {"project": "taOS", "topic": "UI", "subtopic": "dark mode", "weight": 0.7, "is_new_path": false},
    {"project": "taOS", "topic": "backend", "subtopic": "memory", "weight": 0.3, "is_new_path": false}
  ],
  "rationale_quote": "verbatim line from the session supporting the top label"
}
```

**Prompt body** (append to `persona_for(agent_name)`):
> Task: File this session under the right project / topic / subtopic.
> You are shown the agent's existing taxonomy. Reuse an existing path when the session fits one. Only coin a new path when no existing path is close. Coining a path is a commitment — the Reading Room will file sessions under it forever.
> A session often spans multiple labels. Return up to 3 labels with weights summing to 1.0. The top label is the "primary" filing; secondary labels let retrieval catch it from either direction.
> Quote one line from the session that most supports the primary label. If you cannot quote a supporting line, return `labels: []` and let the heuristic fallback take it.
> Existing taxonomy: {taxonomy}
> Session: {session_text}

**Taxonomy management.**
- Per-agent `data/agent-memory/{name}/taxonomy.json`: `{paths: [{project, topic, subtopic, status: "confirmed"|"pending", first_seen, uses}]}`
- New paths enter as `pending` with `uses=1`. When `uses >= 3` across distinct sessions, flip to `confirmed`.
- `pending` paths are shown to the LLM but marked `(pending)` in the prompt so it prefers confirmed ones.

**Storage in SessionCatalog.** Add three columns to `sessions` (see Data Model). The existing `category` column stays as the legacy category; the new columns are the Librarian's hierarchical filing.

**Retrieval use.** `retrieve()` accepts a new optional `filter_paths: list[str]` (e.g. `["taOS/UI/*"]`). When set, candidate sessions are prefiltered by taxonomy match before vector scoring. Unset = current behaviour.

**Failure mode.** Empty `labels` → fall back to the old heuristic category from split. Session is still indexed, just not hierarchically filed. Logged to retry queue.

### 2. `routing_prompt` (issue #6)

**Difference from `intent_classifier`.** `intent_classifier` is regex over ~50 patterns, returns one of 7 intents in microseconds, zero tokens. `routing_prompt` is an LLM call, returns a ranked set of holdings to query with weights, and handles queries where the regex returns `EXPLORATORY` or where multiple intents tie.

**Handoff logic** (new helper `retrieval.plan_retrieval(query, agent)`):
```python
intent = classify_intent(query)
strategy = get_search_strategy(query)
if intent == EXPLORATORY or _tie_score(query) < 2:
    if is_task_enabled(agent, "query_expansion"):  # reuse flag
        strategy = llm_route(query, agent)
return strategy
```

**Output schema.**
```json
{
  "holdings": [
    {"name": "catalogue", "weight": 0.7, "reason": "asks about a stated fact"},
    {"name": "hall", "weight": 0.3, "reason": "verify verbatim"}
  ],
  "expected_form": "fact | quote | summary | timeline | pattern"
}
```

**Prompt body.**
> Task: Decide which holdings to query for this question. You may pick multiple. Weights sum to 1.0.
> Holdings: hall (verbatim), stacks (semantic), catalogue (structured facts), reading-room (sessions), digests, reference (reflections).
> Also predict the expected form of the answer so the assembler can choose the right response template.
> Query: {query}

**When LLM routing disagrees with regex.** LLM wins for that query. Disagreements are logged to `data/agent-memory/{name}/routing_disagreements.jsonl` for offline pattern tuning.

### 3. `verification_prompt` (issue #7) and `contradiction_check_prompt` (issue #10)

**Shared schema.**
```json
{"verdict": "supports | contradicts | silent", "quote": "<verbatim line from evidence>", "note": "<one sentence>"}
```
One JSON parser for both.

**`verification_prompt` — retrieval-time.**

Call site: inside `retrieve()` after cross-encoder rerank, applied to top-K where K defaults to 3. For each candidate, the prompt receives `(query, candidate_text, hall_of_records_quote_for_candidate)`.

Action on verdict:
- `supports`: confidence bonus +0.1
- `contradicts`: confidence penalty -0.4, flag `conflict=true` on the result
- `silent`: no change

Prompt body:
> Task: Is this candidate answer supported by, contradicted by, or unaddressed by the verbatim Hall of Records quote below?
> Do not reason about plausibility. Only decide based on the quote.
> If the quote does not mention the fact either way, the answer is `silent`.
> Quote a substring of the evidence to support your verdict. `silent` verdicts return an empty quote.
> Query: {query}
> Candidate: {candidate_text}
> Hall of Records evidence: {hall_quote}

**`contradiction_check_prompt` — ingest-time.**

Call site: inside `catalog_pipeline._file_card`, before inserting any new KG triple. Search existing triples with same subject and semantically-close predicate; if found, run the prompt.

Action on verdict:
- `contradicts`: existing triple gets `superseded_by = new_id`, both kept, new one carries `conflict_reason`
- `supports`: drop the new triple (duplicate), increment `confirmation_count` on the old one
- `silent`: file the new triple normally

Prompt body:
> Task: Does this new triple contradict an existing one in the Card Catalogue for {agent_name}?
> Two triples contradict when they cannot both be true at the same time for the same subject. "Jay prefers Python" and "Jay prefers Go" contradict. "Jay prefers Python" and "Jay knows Go" do not.
> Temporal facts with non-overlapping validity windows do not contradict even if they seem to — they are history.
> New: {new_triple}
> Existing: {existing_triple_list}

### 4. `disambiguation_prompt` (issue #8)

**Call site.** Retrieval returns >1 candidate with near-identical scores (delta < 0.05) and the same subject entity. Triggered only when the caller opts in via `retrieve(disambiguate=True)`.

**Output.**
```json
{"choice_index": 2, "reason": "user explicitly named the 2024 version"}
```

Returns `-1` if genuinely ambiguous; caller then returns all candidates with `needs_user_disambiguation: true`.

### 5. `citation_format_prompt` (issue #11)

**Call site.** `ContextAssembler.format_citation(hit)`, invoked per retrieval hit when the consumer requests formatted output.

**Output.**
```json
{"quote": "<exact substring>", "attribution": "Hall of Records, 2026-03-12 14:23", "confidence": 0.87}
```

The `quote` field MUST be a verbatim substring of the source. Implementer validates with `assert quote in source_text` and retries once on failure; after two failures, falls back to a deterministic template (first 120 chars + ellipsis).

### 6. `redaction_prompt` (issue #12)

**Call site.** Runs on Hall of Records writes in addition to `secret_filter.redact_secrets()` (regex), not instead of it. Regex runs first (fast), LLM runs second (catches novel secrets regex misses: custom tokens, PII in prose).

**Output.**
```json
{"spans": [{"start": 142, "end": 178, "reason": "looks like a bearer token"}]}
```

Implementer replaces each span with `[REDACTED:<reason>]` in the indexed copy. **Hall of Records retains the original unredacted text** — redaction applies to Stacks, Digests, and Reference Desk only. The Hall is never edited.

Quality bar: false positives are worse than false negatives (losing recall). Prompt is tuned conservative: "only flag a span if removing it would not change the meaning of the surrounding sentence."

### 7. `cross_reference_prompt` (issue #13)

**Call site.** Nightly batch over the Card Catalogue, not at ingest. Input: a cluster of cards sharing a subject. Output: `see_also` edges between cards.

**Output.**
```json
{"edges": [{"from": "card_42", "to": "card_117", "relation": "see-also", "reason": "both describe Jay's editor"}]}
```

Edges are filed in a new `card_edges` table. Retrieval uses these to expand a hit — if you hit card_42, card_117 is attached as context.

## Data Model Changes

### SessionCatalog (`session_catalog.py`)

```sql
ALTER TABLE sessions ADD COLUMN primary_project TEXT DEFAULT '';
ALTER TABLE sessions ADD COLUMN primary_topic TEXT DEFAULT '';
ALTER TABLE sessions ADD COLUMN primary_subtopic TEXT DEFAULT '';
ALTER TABLE sessions ADD COLUMN labels_json TEXT DEFAULT '[]';
ALTER TABLE sessions ADD COLUMN classified_at REAL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_sessions_path
  ON sessions(primary_project, primary_topic, primary_subtopic);
```

FTS5: extend `catalog_fts` to include path components so topic-search matches labels. Requires FTS rebuild (derived, acceptable).

### Per-agent taxonomy file

`data/agent-memory/{name}/taxonomy.json`:
```json
{
  "version": 1,
  "paths": [
    {"project": "taOS", "topic": "UI", "subtopic": "dark mode",
     "status": "confirmed", "first_seen": 1712000000, "uses": 17}
  ]
}
```

### KG additions

```sql
CREATE TABLE IF NOT EXISTS card_edges (
    id INTEGER PRIMARY KEY,
    from_card INTEGER NOT NULL,
    to_card INTEGER NOT NULL,
    relation TEXT NOT NULL,
    reason TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX idx_card_edges_from ON card_edges(from_card);
```

`supersedes` and `contradicts` edges write here alongside the existing supersede-chain fields.

### Routing-disagreement log

`data/agent-memory/{name}/routing_disagreements.jsonl` — append-only, used offline to tune `intent_classifier` patterns.

## `is_task_enabled()` Wiring

**New helper** in `agents.py`:
```python
def run_if_enabled(agent: str, task: str, fn, *args, fallback=None, **kw):
    """Invoke fn only when the task is enabled; else return fallback."""
```

**Gate points:**

| Task | File | Call site | Fallback when disabled |
|------|------|-----------|------------------------|
| `fact_extraction` | `catalog_pipeline.py` | around `extract_facts_from_text` | skip (no cards filed) |
| `preference_extraction` | `catalog_pipeline.py` | `extract_preferences` | skip |
| `intake_classification` | `catalog_pipeline.py` | new stage before `enrich_session` | heuristic category from split |
| `crystallise` | `catalog_pipeline.py` | `cs.crystallize()` | skip, session stays tier 1 |
| `reflect` | `reflect.py` | `InsightStore.synthesize()` | skip |
| `catalog_enrichment` | `catalog_pipeline.py` | `catalog.enrich_session()` | keep heuristic topic |
| `query_expansion` | `retrieval.py` | `expand_query_llm()` | fall back to `expand_query_fast` |
| `verification` | `retrieval.py` | verification wrapper on top-K | skip confidence adjustment |

`contradiction_check`, `disambiguation`, `citation_format`, `redaction`, `cross_reference` have no per-task flag in v1 — they run when master `enabled` is true. Add flags only if token cost measurement shows them necessary.

**Skip semantics.** Disabled = silent skip with a structured log entry `{agent, task, reason: "disabled"}`. Never raises.

## Eval Design

### Why not LongMemEval-S alone

97.0% Recall@5 proves vector + rerank works on single-hop retrieval. It does not measure stale-fact serving, multi-hop coherence, or routing accuracy — these are where the Librarian earns its tokens.

### Three complementary test axes

#### Axis A — Contradiction Accumulation

50 scenarios. Each is a sequence of 3–6 sessions where session N states a fact updating or contradicting session N-1. Ground truth: the latest fact.

**Metric.** `stale_rate = fraction of queries where the stale fact is served as current`. Break out by: sessions between contradiction and query; explicit vs implicit contradiction.

#### Axis B — Routing Sensitivity

200 queries, 50 each in four buckets: `factual`, `preference`, `temporal`, `pattern`.

**Metric.** Accuracy per bucket across three pipeline configs. The Librarian earns its cost when its accuracy on preference and pattern buckets is >5pp higher than vector-only.

#### Axis C — Long-Horizon Coherence

20 conversations of 100–300 turns. Test question at turn N+k references a fact from the first 20 turns. k ∈ {25, 50, 100, 200}.

**Metric.** Recall@5 on the early-turn fact; degradation curve across k.

### Three pipeline configurations

| Config | Components |
|--------|-----------|
| `vector-only` | VectorMemory + embedding search, no rerank, no KG, no assembler |
| `full-pipeline` | Embedding + cross-encoder + KG + temporal boost; no LLM routing, no verification, no intake classification |
| `full+librarian` | `full-pipeline` + intake classification at ingest + LLM routing on EXPLORATORY + verification on top-3 + contradiction_check on new cards |

### Primary KPI

**Net answer quality per 1000 tokens spent on enrichment.**
- Numerator: composite of (1 - stale_rate on A) + accuracy gain on B + recall@k on C, normalised.
- Denominator: total LLM tokens consumed by the Librarian over the test run, measured per-task.

Target to call the Librarian "worth it": ≥15% composite quality gain over `full-pipeline` at ≤3x token cost.

**Implementation.** `eval/librarian_eval.py`, runnable as `python -m taosmd.eval.librarian`. Writes `eval/reports/{date}/{config}.json`. Dataset fixtures in `eval/fixtures/` as JSONL per axis.

## Standalone Framework Integration

`taosmd.ingest()` currently writes to the Archive but does not trigger the catalog pipeline. Add:

1. `taosmd.ingest(transcript, agent=..., enrich=True)` — when `enrich=True` (default), runs split + intake_classify synchronously after archive write. taOS callers pass `enrich=False` and run the nightly pipeline themselves.

2. `taosmd.flush(agent)` — forces a full enrichment pass for any un-enriched sessions. For standalone apps that want to batch at their own cadence.

3. Minimal startup idiom (document in AGENTS.md):
```python
import taosmd
taosmd.register_agent("my-agent")  # auto-registers on first ingest if omitted
taosmd.ingest(transcript)           # after each turn
hits = taosmd.search(query)         # before each response
```

No persistent daemon required for standalone. Everything is sync-on-demand.

## Open Questions

1. **Taxonomy collisions across agents.** Separate per-agent taxonomy (current spec) vs shared install taxonomy. Defer until cross-agent search is needed.

2. **Pending path promotion threshold.** `uses >= 3` is a guess. May need `uses >= 3 AND spans >= 2 distinct days`. Decide after first week of dogfooding.

3. **Verification coverage.** Run on every retrieval, or only below a confidence band? v1: run on top-3 only, gated by `is_task_enabled(agent, "verification")`.

4. **Cross-reference batch cadence.** Nightly in spec but arbitrary. Could be triggered by "every N new cards under a subject." Not load-bearing for v1.

5. **LLM token accounting.** Eval KPI requires per-task token counts. Needs a small `token_ledger.py` wrapping each prompt call. Not a blocker for spec but required before eval can produce the primary KPI.

6. **Disambiguation UX.** When `disambiguation_prompt` returns `-1`, the current `search()` signature has no way to surface "please pick one." Options: add `needs_user_disambiguation` field to return, or add `search_with_clarification()` entry point. Leaning toward the former.

## Critical files for implementation

- `taosmd/prompts.py` (feat/librarian-controls) — add new prompts here
- `taosmd/catalog_pipeline.py` — add intake_classify stage, wire `run_if_enabled`
- `taosmd/session_catalog.py` — schema migration, taxonomy management
- `taosmd/retrieval.py` — `plan_retrieval()`, verification wrapper
- `taosmd/agents.py` — `run_if_enabled()`, new LIBRARIAN_TASKS entries
- `eval/librarian_eval.py` — new file, three-axis eval harness
