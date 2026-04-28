# Supersede-chain benchmark — design spec

**Workstream:** Phase C #1 in `docs/specs/2026-04-16-taosmd-next-steps.md`
**Depends on:** Phase A baseline LoCoMo result (so we have a fair comparison)
**Replaces:** Axis A in `eval/librarian_eval.py` (which flat-lined because the fixtures were synthetic and didn't exercise real temporal conflict)

**2026-04-29 update — pivoted to synthetic dataset.** A 20-question spot-check of LoCoMo cat-2 (temporal) questions found **0 / 20 require supersede-chain disambiguation** — every cat-2 question is a "when did X happen" single-event lookup, not "which X is current" with contradictory facts. LoCoMo doesn't exercise the supersede failure mode at all.

Three honest paths considered: (1) defer indefinitely; (2) build proper synthetic fixtures that *actually* exercise the failure mode (Axis A's mistake was the fixture quality, not synthetic-ness itself); (3) keep only the routing benchmark.

Decision: **option 2.** We built supersede chains because we believe they help; we owe ourselves a measurement before claiming users benefit. Below: the methodology to make synthetic fixtures real enough to measure something useful.

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

## Implementation plan (post-pivot)

**Phase 1 — fixture generation (~1 day):**

1. Write `benchmarks/supersede_fixtures/generate.py` — uses qwen3.5:9b on Fedora to generate ~80 candidate scenarios (oversample, we'll cull to 50 after review). Each scenario: 4–8 sessions spanning ~3 months, 30+ turns total, one fact that changes mid-conversation, and a question explicitly asking for the *current* state.
2. Persist to `benchmarks/supersede_fixtures/scenarios.jsonl` with schema:
   ```
   {
     "id": "scn_001",
     "domain": "jobs",
     "sessions": [{"date": "2024-03-12", "turns": [{"speaker": "...", "text": "..."}, ...]}, ...],
     "outdated_fact": "Jay works at Stripe",
     "current_fact": "Jay works at OpenAI",
     "question": "Where does Jay currently work?",
     "answer": "OpenAI"
   }
   ```
3. Manual review pass: I sample-read 20% of generated scenarios and accept/reject. Document the reject criteria so the methodology is reproducible. Reject if: the contradiction is implausible, the answer is given in the same session as the question, or the surrounding turns trivially contain the answer.

**Phase 2 — runner (~1 day):**

Fork `benchmarks/locomo_runner.py` → `benchmarks/supersede_runner.py`. Key differences:

- Ingest path adds KG fact-extraction via `taosmd.memory_extractor.process_conversation_turn` for the `full+supersede` config. ~50 scenarios × ~30 turns × 2-5s extraction = 25-125min total ingest. Acceptable.
- Three configs to compare: `baseline-vector` (vmem only), `vector+kg` (no supersede chain check, just KG retrieval), `full+supersede` (KG + `is_fact_superseded_prompt` + supersede-aware retrieval).
- Per-scenario: ingest → ask question → check whether retrieved facts include the **current** fact and *not* the outdated one (or include both with supersede metadata).
- Metrics: F1 + LLM-Judge as in LoCoMo, plus **supersede-specific metrics**:
  - `current_hit_rate`: fraction where retrieval surfaced the current fact
  - `outdated_filter_rate`: fraction where retrieval correctly *excluded* the outdated fact (or marked it as superseded)
  - `chain_traversal_count`: how often the supersede edge was actually consulted

**Phase 3 — run + analyse (~half day):**

50 scenarios × 3 configs × ~10s/answer = ~25min generation + ~2-3h external rescore. Output `docs/specs/2026-04-29-supersede-benchmark-results.md` with the methodology disclosed up front.

## Success criteria (revised for synthetic dataset)

**Honest bar:** `full+supersede` must beat `vector+kg` (no supersede check) on `current_hit_rate` by at least **+15 absolute points**. Smaller deltas are noise on a 50-scenario set.

**Failure case:** if supersede doesn't beat plain KG retrieval on this construct-to-test fixture set, the supersede mechanism is either not wired correctly or doesn't add value beyond what raw KG retrieval already provides. Either way, this measurement is the gating evidence — don't claim supersede is a feature in the README until this benchmark passes.

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
