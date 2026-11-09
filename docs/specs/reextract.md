# Re-extraction over the archive (`taosmd reextract`)

Status: SPEC for Jay's approval (issue #658). No code ships with this document.
Implementation cards are filed only after the open questions below are answered.

## Goal

Claims are minted only at ingest time. When the extractor improves, history that
was already ingested stays extracted by the old extractor forever. The archive
keeps every turn verbatim, so the raw material is there: the shadow corpus in
#658 had 76 archive rows matching "apple" and one claim mentioning it.

`taosmd reextract` re-runs an extractor over archived conversation spans,
deduplicates the output against existing claims, records which extractor
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
    PRIMARY KEY (span_id, extractor_id)
)
```

Rows are inserted and never updated or deleted. A span counts as covered for an
extractor when it has a row for that `extractor_id`, including spans that
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
                 [--limit N] [--dry-run] [--data-dir DIR]
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
   "skipped_no_content", "claims_new", "claims_matched"}`. Running again with the
   same arguments reports `processed == 0`.

With no filters, every conversation span not yet covered by the current
extractor id is processed. Other event types (A2A, tool calls, errors) are out
of scope.

New claims land `status='unverified'` for the existing verify pass. Live ingest
and `reextract` can run concurrently: both write under `BEGIN IMMEDIATE`, as
`add_claim` already does.

### 4. Dedup contract

`reextract` reuses `add_claim` semantics exactly: dedup on exact `norm`, no fuzzy
matching, and an existing row's text, status and `source_extractor` are never
changed. A new row gets the new `extractor_id` as its `source_extractor`.

The one open point is what happens when a re-extracted claim matches an existing
row whose `archive_span_ids` lacks this span. See Q1.

### 5. Files a lane would touch

`taosmd/claims/extract.py` (extractor ids, LLM claim path), `taosmd/claims/store.py`
(insert path for Q1, coverage-log reads/writes), `taosmd/migrations.py`
(`reextract_log` table), `taosmd/service.py` (`reextract()`), `taosmd/cli.py`
(subcommand). Expected split is three cards: extractor ids plus coverage table,
the regex pass plus CLI, and the LLM extractor path.

## Acceptance

- Fixture test: archive N conversation spans with known facts, run `reextract`,
  assert the claim counts, then run again and assert `processed == 0` and no
  table changed (row counts plus a content hash of `claims`).
- Crash test: fail mid-run after K spans, re-run, and assert every span is
  covered exactly once.
- Never-mutate test, whose form depends on Q1: the text, status and
  `source_extractor` of every pre-existing claim row are byte-identical
  after the run.
- Measurement (the #658 probe): on the shadow corpus, record the claim count and
  the `apple foundation models on-device llm` probe outcome before and after
  `reextract --extractor llm`, naming the model, the corpus snapshot and the
  command. The regex pass alone is not expected to close this gap; the issue's
  missing claims need the LLM extractor.

## Open questions for Jay

- **Q1. A re-found claim matching an existing row.**
  (A, recommended) Union the span id into the existing row's `archive_span_ids`,
  exactly as every live ingest already does under #653. This adds provenance and
  changes no fact, status or text, and the recall gate (`status_for_spans`) sees
  the link with no read-path change.
  (B) Strict no-UPDATE: leave the existing row untouched and record the link only
  in `reextract_log` (or a link table). That needs `status_for_spans` and any
  other span reader to union the log in.
- **Q2. Default extractor.** (A, recommended) `regex` by default, with `llm`
  opt-in, since an LLM pass over a large corpus is real compute and its ids churn
  with every model change. (B) `llm` by default when an LLM is configured.
- **Q3. Scope.** (A, recommended) Claims only in this spec, with KG and vector
  re-extraction as a later spec on the same coverage log. (B) Fold KG triples in
  now.

## Out of scope

KG and vector re-extraction (Q3), non-conversation event types, scheduling it in
the nightly cron, and deleting or rewriting any claim row.
