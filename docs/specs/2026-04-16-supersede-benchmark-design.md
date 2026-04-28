# Supersede-chain benchmark — design spec

**Workstream:** Phase C #1 in `docs/specs/2026-04-16-taosmd-next-steps.md`
**Depends on:** Phase A baseline LoCoMo result (so we have a fair comparison)
**Replaces:** Axis A in `eval/librarian_eval.py` (which flat-lined because the fixtures were synthetic and didn't exercise real temporal conflict)

**2026-04-29 update — pivoted to LongMemEval `knowledge-update` subset.** A 20-question spot-check of LoCoMo cat-2 (temporal) found **0 / 20 require supersede disambiguation** — LoCoMo doesn't exercise the failure mode. *But* a research pass discovered that **LongMemEval-S has a `knowledge-update` category — 78 questions, multi-session dialogue, multiple `answer_session_ids` per QA, license MIT** — that is exactly the supersede shape, and we already run LongMemEval-S at 97.0%. The right benchmark already exists; we just need to filter to KU and add the comparison configs.

Plan: report KU subset as headline (third-party citation-friendly), keep the synthetic-fixture path as supplementary (covers domains LME-KU might miss). Synthetic supplement is auxiliary, not primary.

## How this differs from Axis A's failure

Axis A used 15 hand-crafted Q-A pairs with no surrounding conversation context — the LLM saw the question + the facts in isolation and could disambiguate without retrieval. The benchmark scored 1.0 trivially because there was nothing to fail at.

The new supersede dataset must:

1. **Embed contradicting facts in realistic multi-session conversations** with at least 30+ turns of unrelated content between the contradicting statements. The retrieval system must actually *find* the fact among noise, not be handed it on a plate.
2. **Use turn-level timestamps** spanning weeks-to-months — supersede chains rely on temporal ordering. Synthetic timestamps must be plausible.
3. **Include the *current* fact and at least one *outdated* fact** for each test case, with the question explicitly asking for the current state ("currently", "now", "as of today").
4. **Diverse domains** — jobs, relationships, locations, preferences, possessions, beliefs, health, projects. ~50 scenarios distributed across these so we're not measuring one specific failure mode.
5. **Each scenario manually reviewed** before inclusion. Reject anything where the answer is obvious without retrieval (e.g. "Jay just told me he works at OpenAI in this same session"), or where the contradiction is implausible.

## What we're measuring

**Question:** Does taosmd's supersede-chain mechanism (`is_fact_superseded_prompt` + the KG's supersede relationships) actually improve recall when a user has stated two contradictory facts at different points in time?

**Failure mode we want to catch:** An agent that naively embeds both "I use Flask" (March) and "I'm on FastAPI now" (last week) and retrieves both at query time, leaving the LLM to guess which is current. The correct answer is *always the newer one* — the question is whether the system surfaces it without relying on the LLM's post-hoc disambiguation.

## Why LoCoMo category 2 is the right dataset

LoCoMo category 2 = **temporal reasoning**. 321 QAs across the 10 conversations, roughly 32 per conversation. Questions like:

- "When did Caroline go to the LGBTQ support group?" → "7 May 2023"
- "When did Melanie paint a sunrise?" → "2022"
- "What year did Jay switch from Flask to FastAPI?" → requires detecting the switch event

These are ready-made supersede scenarios. The dataset already has the temporal ground truth — we just need to compare system responses.

## The two configurations

| Config | KG ingest | Retrieval | Query expansion |
|--------|-----------|-----------|-----------------|
| **baseline** | none | `VectorMemory.search` only | none |
| **full+supersede** | LLM extracts facts into KG, `is_fact_superseded_prompt` runs on conflicting facts to build supersede chains | `retrieve(strategy="thorough", sources={"vector": vmem, "kg": kg})` with KG temporal filters respecting supersede chains | `query_expansion_prompt` per the existing Librarian run |

## Implementation plan (post-pivot to LongMemEval-KU)

**Phase 1 — runner (~half day):**

Fork `benchmarks/longmemeval_runner.py` → `benchmarks/longmemeval_ku_runner.py`. Key differences:

- Filter to `question_type == "knowledge-update"` only (78 of 500 questions).
- Three configs to compare:
  - `baseline-vector`: `VectorMemory.search` only — what we measure today.
  - `vector+kg`: ingest with `process_conversation_turn` to populate KG, retrieve via `retrieve(strategy="thorough", sources={"vector": vmem, "kg": kg})` *without* supersede edge traversal.
  - `full+supersede`: same as vector+kg, but with `is_fact_superseded_prompt` run on conflicting facts at ingest time, and retrieval respects supersede edges (returns *only* the latest fact in a chain, or marks the older one explicitly).
- External judge via `locomo_rescore_streaming.py` with `qwen3:4b` (same as LoCoMo).
- Track supersede-specific metrics on top of F1 + LLM-Judge:
  - `current_hit_rate`: fraction where retrieval surfaced the *current* fact (most recent answer_session_id chain endpoint).
  - `outdated_only_rate`: fraction where retrieval *only* surfaced the older fact (the failure mode we're catching).
  - `chain_traversal_count`: count of QAs where a supersede edge was consulted at retrieval time.

**Phase 2 — run on Fedora (~half day):**

78 questions × 3 configs × ~10s/answer = ~40min generation + ~30-60min external rescore (much smaller than full LoCoMo). Slot into the running chain after `kitchen_sink` lands. Output JSON shipped through the existing rescore pipeline.

**Phase 3 — synthetic supplement (optional, ~1 day):**

If LME-KU shows the supersede mechanism is helping, build a smaller synthetic supplement (~30 cases) covering domains LME-KU might miss (medical history, evolving relationships, possessions changing hands). Reported as auxiliary evidence — third-party LME-KU is the headline, synthetic is the depth-confirmation.

If LME-KU shows supersede is *not* helping, the synthetic supplement becomes diagnostic — does it fail because the failure mode doesn't exist in real conversations, or because our mechanism doesn't catch it? The supplement gives us an answer.

## Success criteria

**Honest bar:** `full+supersede` must beat `vector+kg` on `current_hit_rate` by at least **+10 absolute points** on the 78-question KU subset. Smaller deltas are within noise on this sample size.

**Headline number for the README:** LongMemEval-S `knowledge-update` external-Judge accuracy under `full+supersede`, alongside the 97.0% overall LME-S number. Citation: LongMemEval (Wu et al., ICLR 2025), arxiv 2410.10813.

**Failure case:** if supersede doesn't beat plain KG retrieval on the LME-KU subset, the mechanism is either not wired correctly, not surfacing the right facts at retrieval, or not actually needed (KG retrieval alone might already disambiguate by recency-of-extraction). In any failure case: do NOT claim supersede is a feature in the README until we either fix the mechanism or pivot the claim ("we use a KG with temporal ordering" rather than "we use supersede chains").

## What needs to exist before this is runnable

1. **Baseline LoCoMo result** on cat 2 from the vanilla `locomo_runner.py` — published as the reference point. ✅ done.
2. **KG ingest path for LoCoMo** — the current `locomo_runner.py` ingests only into `VectorMemory`. Supersede chains live in the KG; without KG ingest, nothing to test. New code needed: wire `process_conversation_turn(text, agent_name, kg, archive, source, ...)` from `taosmd.memory_extractor` into the per-turn loop in `_ingest_conversation`. This is the biggest unknown — the LLM fact extraction is ~2-5s per turn; at 400 turns per conversation, that's 15-30 min of ingest per conv, plus 10 conversations = 2.5-5h of ingest alone. Either sample (cat-2-adjacent turns only) or budget accordingly.
3. **Supersede chain traversal confirmed end-to-end.** `taosmd/knowledge_graph.py` has the `supersedes` relationship but nobody has verified it filters correctly at retrieve time. Integration test required before the benchmark; add one to `tests/test_knowledge_graph.py` with a 3-fact chain (A superseded by B superseded by C → query returns C only).
4. **`retrieve()` with KG source actually filters on supersede.** Read `taosmd/retrieval.py::_query_source` for kg — confirm the kg adapter respects `supersedes` edges. If not, patch.

## File to create

`benchmarks/supersede_benchmark.py` — derived from `locomo_runner.py`. Key changes:

- Hardcode `include_cats = {2}` (temporal only).
- Replace the `VectorMemory.search` path with `retrieve(...)` when `--config=full+supersede`.
- Add `kg = KnowledgeGraph(...)` and `archive = ArchiveStore(...)` to the per-conversation setup.
- Swap the ingest loop for `process_conversation_turn(...)` calls (slower but does fact extraction).
- For each QA, retrieve via `retrieve(sources={"vector": vmem, "kg": kg, "archive": archive})`.
- Report F1, BLEU-1, Judge as in LoCoMo runner. Add one extra metric: **supersede_hit_rate** = fraction of QAs where the retrieved facts include at least one kg record with a non-null `superseded_at` chain endpoint. This measures *whether* supersede edges were surfaced at all, not just whether the answer was right.

## Success criteria

**Honest bar:** `full+supersede` delivers **at least +5 F1 points** on cat 2 over `baseline`, at a per-QA token cost **≤3000** (tighter than the general Librarian budget because this is a deeper pipeline).

**Dream bar:** +10 F1 points. That would be publishable as a standalone finding.

**Failure case:** if `full+supersede` is the same as or worse than baseline, the supersede-chain mechanism is either (a) not surfacing relevant facts at retrieval, (b) surfacing them but the LLM answer still gets confused, or (c) the cat-2 questions don't actually require supersede disambiguation (many are "when did X happen" about a single event, which doesn't need supersede). Each failure mode has a different fix; don't ship the "it didn't work" conclusion without investigating which.

## Open questions

1. **Is LoCoMo cat 2 actually supersede-shaped?** Spot-check 20 random cat 2 QAs. Are they "when did X" (single event, no supersede needed) or "which X is current" (requires supersede)? If the former dominates, this benchmark measures temporal *retrieval*, not supersede *resolution*. Still valuable but rename the workstream.
2. **Should the baseline include Librarian query expansion?** If yes, we're measuring supersede in isolation. If no, we're measuring supersede + expansion jointly. Pick one and say so.
3. **Ingest cost budget.** 5h of LLM fact extraction per benchmark run is steep. Options: (a) batch fact extraction offline and cache the KG, (b) use a smaller extraction model, (c) subsample conversations. Decide before building.

## Rough timeline

- 2h — spot-check cat 2 questions to confirm supersede-shape
- 4h — build the runner (most of the code is a fork of `locomo_runner.py`)
- 8h+ — first full run on Fedora (dominated by ingest)
- 2h — analyse, publish or iterate

Total: ~2 working days of deep focus + machine time. Do not start before baseline LoCoMo result is in.
