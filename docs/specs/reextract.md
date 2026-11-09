# Re-extraction over the archive (`taosmd reextract`)

Status: APPROVED by Jay 2026-11-09 (issue #658): Q1 A, Q2 A, Q3 B (KG triples are
in scope, section 5). No code ships with this document.

## Goal

Claims are minted only at ingest time. When the extractor improves, history that
was already ingested stays extracted by the old extractor forever. The archive
keeps every turn verbatim, so the raw material is there: the shadow corpus in
#658 had 76 archive rows matching "apple" and one claim mentioning it.

`taosmd reextract` re-runs an extractor over archived conversation spans,
deduplicates the output against existing claims and KG triples, records which extractor
processed which span, and never rewrites what an earlier run produced. It is
resumable and safe to run again after every future extractor change.

## What exists today (master 8ded52e6)

- Both ingest paths (`taosmd/api.py` `ingest` and `ingest_batch`) archive each turn
  as `event_type="conversation"` with the text in `data_json["content"]`, then
  call `claims_from_text(text, span_id)` (`taosmd/claims/extract.py`). The span id
  is the `archive_index.id` of that row.
- `claims_from_text` wraps the regex extractor only
  (`memory_extractor.extract_facts_from_text`) and stamps
  `source_extractor="regex"`. There is no version on that stamp.
- `ClaimStore.add_claim` (`taosmd/claims/store.py`) deduplicates at insert on the
  exact-normalised text (`norm`, case and whitespace folded). On a match it
  UPDATEs the existing row's `archive_span_ids` to the union and keeps the row's
  status. Paraphrases stay separate rows. This is the #653 contract and runs on
  every ingest today.
- `ArchiveStore.query` already filters on `event_type`, `agent_name`, `since`,
  `until`, with `limit`/`offset` paging.
- An LLM extractor exists (`memory_extractor.extract_facts_with_llm`, prompt from
  `taosmd/prompts.py` `extraction_prompt`), but it feeds the KG and vector paths,
  not claims.
- Precedent for a one-shot idempotent backfill: `service.a2a_migrate_kinds`
  (a second run reports `migrated == 0`).

## Design

### 1. Extractor identity

Every extractor gets an id string that changes whenever its behaviour changes.
Coverage and provenance are keyed by this id.

- Regex: `regex@<N>`, with `N` a constant declared in `taosmd/claims/extract.py`.
  A test pins a hash of the regex pattern source, so editing a pattern fails CI
  until `N` is bumped in the same change. This replaces the bare `"regex"` stamp
  for new claims. Existing rows keep `"regex"`, and the spec treats it as
  `regex@0`.
- LLM: `llm:<model>:<sha256(extraction prompt template)[:12]>`. A model or prompt
  change produces a new id without anyone having to remember a version bump.

### 2. Coverage log (new table, append-only)

A new table in `claims.db`:

```
reextract_log(
    span_id        INTEGER NOT NULL,   -- archive_index.id
    extractor_id   TEXT    NOT NULL,
    run_id         TEXT    NOT NULL,
    extracted_at   REAL    NOT NULL,
    claims_found   INTEGER NOT NULL,   -- facts the extractor returned
    claims_new     INTEGER NOT NULL,   -- rows inserted
    claims_matched INTEGER NOT NULL,   -- folded into an existing row by norm
    target         TEXT    NOT NULL,   -- 'claims' or 'kg'
    PRIMARY KEY (span_id, extractor_id, target)
)
```

Rows are inserted and never updated or deleted. A span counts as covered for an
extractor and target when it has a row for that `extractor_id` and `target`, including spans that
yielded zero facts.

Rejected alternatives:
- A column on `archive_index`. The archive is the permanent contract, so
  bookkeeping for a derived store does not belong in it.
- Inferring coverage from `claims.source_extractor`. A span that yields no facts
  leaves no claim row, so "processed, found nothing" looks the same as "never
  processed". Re-stamping `source_extractor` would also mutate old rows.

### 3. The pass

```
taosmd reextract [--since YYYY-MM-DD] [--agent NAME] [--extractor regex|llm]
                 [--target claims|kg|all] [--limit N] [--dry-run] [--data-dir DIR]
```

1. Resolve `extractor_id` for the chosen extractor.
2. Page through `archive.query(event_type="conversation", agent_name=..., since=...)`
   in ascending `id` order. Skip spans that already have a `reextract_log` row
   for this `extractor_id`.
3. For each span, read `data_json["content"]`. A row with no string content is
   counted as `skipped_no_content`, logged with `claims_found=0`, and never
   retried.
4. Run the extractor and pass each claim through the dedup contract (section 4).
5. Write the span's `reextract_log` row in the same transaction as its claim
   writes. A crash therefore leaves each span fully done or not started, and the
   next run resumes where this one stopped.
6. `--dry-run` runs steps 1 to 4 read-only and reports what would change.
7. Return and print `{"run_id", "extractor_id", "processed", "skipped_covered",
   "skipped_no_content", "claims_new", "claims_matched", "kg_new", "kg_matched",
   "kg_deferred"}`. Running again with the
   same arguments reports `processed == 0`.

With no filters, every conversation span not yet covered by the current
extractor id is processed. Other event types (A2A, tool calls, errors) are out
of scope.

New claims land `status='unverified'` for the existing verify pass. Live ingest
and `reextract` can run concurrently: both write under `BEGIN IMMEDIATE`, as
`add_claim` already does.

`--target` defaults to `all`. Claims and KG are covered separately, so a span
can be done for one target and pending for the other.

### 4. Dedup contract (claims)

`reextract` reuses `add_claim` semantics exactly: dedup on exact `norm`, no fuzzy
matching, and an existing row's text, status and `source_extractor` are never
changed. A new row gets the new `extractor_id` as its `source_extractor`.

When a re-extracted claim matches an existing row whose `archive_span_ids` lacks
this span, the span id is unioned into that row's `archive_span_ids`, exactly as
live ingest does under #653 (Q1 A). No text, status or `source_extractor` changes.

### 5. KG triples (Q3 B)

The extractor's facts are also written as KG triples. Re-extraction reads old
history, so the live KG write path is NOT reusable as-is: `add_triple` only
dedups against an ACTIVE identical triple, and `add_triple_with_contradiction_check`
with `auto_resolve` invalidates the current value (stamping `valid_to = now`) and,
given `vmem`, soft-supersedes vector chunks. Replaying a 2025 "I live in London"
through that path would close today's `lives_in Paris` and hide its vector rows.
`reextract` therefore uses its own insert rule, `KnowledgeGraph.add_historical_triple`:

1. `valid_from` is the span's `archive_index.timestamp`, never `now`.
2. Dedup on subject, predicate and object across ALL rows, open or closed. A
   match is counted as matched and nothing is written, so a superseded fact is
   never resurrected as current.
3. For a predicate in `SINGULAR_PREDICATES`, compare against the subject's other
   values for that predicate:
   - A value whose `valid_from` is later than the span: the re-found triple is
     history. It is inserted closed, with `valid_to` set to the earliest such
     later `valid_from`.
   - Only earlier values, or none: the re-found triple would become the current
     value. It is NOT written. It goes to the pending-decisions queue
     (`kind="contradiction"`, evidence = the span text) for `taosmd review`,
     whatever its confidence.
4. `reextract` never calls `invalidate`, never auto-resolves, and never passes
   `vmem`, so no existing KG or vector row is changed. That is stronger than the
   claims path, which unions span ids, and it is deliberate: a triple row has no
   span-id provenance column in use today.
5. `source` is `reextract:<extractor_id>:span:<span_id>`.

The coverage row for `target='kg'` is written after the span's triple writes.
The KG lives in its own database, so the two writes are not one transaction. A
crash between them re-runs the span, and rule 2 makes the re-run write nothing
new. For `target='kg'` rows, the `claims_*` columns hold the
triple counts (found, inserted, matched); deferrals appear only in the run summary.

### 6. Files a lane would touch

`taosmd/claims/extract.py` (extractor ids, LLM claim path), `taosmd/claims/store.py`
(coverage-log reads/writes), `taosmd/migrations.py` (`reextract_log` table),
`taosmd/knowledge_graph.py` (`add_historical_triple`), `taosmd/service.py`
(`reextract()`), `taosmd/cli.py` (subcommand). Expected split is four cards:
extractor ids plus coverage table, the regex claims pass plus CLI, the KG target,
and the LLM extractor path.

## Acceptance

- Fixture test: archive N conversation spans with known facts, run `reextract`,
  assert the claim counts, then run again and assert `processed == 0` and no
  table changed (row counts plus a content hash of `claims`).
- Crash test: fail mid-run after K spans, re-run, and assert every span is
  covered exactly once.
- Never-mutate test: the text, status and `source_extractor` of every
  pre-existing claim row are byte-identical after the run (only
  `archive_span_ids` may grow), and every pre-existing `kg_triples` row and every
  vector row is byte-identical.
- KG history test: KG holds `user lives_in Paris` (valid_from today). Re-extract a
  span dated last year saying "I live in London". Assert Paris is still current,
  London is inserted closed with `valid_to` = Paris's `valid_from`, and no vector
  row changed. Then re-extract a span dated tomorrow saying "I live in Rome":
  assert nothing is written to the KG and one pending decision exists.
- KG no-resurrection test: a closed `lives_in Berlin` triple plus a re-found
  Berlin span leaves the KG unchanged.
- Measurement (the #658 probe): on the shadow corpus, record the claim count and
  the `apple foundation models on-device llm` probe outcome before and after
  `reextract --extractor llm`, naming the model, the corpus snapshot and the
  command. The regex pass alone is not expected to close this gap; the issue's
  missing claims need the LLM extractor.

## Decisions (Jay, 2026-11-09)

- **Q1 A.** A re-found claim's span id is unioned into the matching row, as live
  ingest does.
- **Q2 A.** `regex` is the default extractor; `llm` is opt-in.
- **Q3 B.** KG triples are in scope now (section 5). Vector re-extraction is not.

## Out of scope

Vector re-extraction, non-conversation event types, scheduling it in
the nightly cron, and deleting or rewriting any claim or triple row.
