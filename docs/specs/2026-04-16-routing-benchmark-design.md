# Routing benchmark — design spec

**Workstream:** Phase C #2 in `docs/specs/2026-04-16-taosmd-next-steps.md`
**Depends on:** Phase A baseline LoCoMo result + supersede benchmark (sharing infrastructure)
**Replaces:** Axis B in `eval/librarian_eval.py` (which trivially scored 1.0 because the fixture was a 15-item pool with one obviously-correct answer — no real routing signal)

## What we're measuring

**Question:** Does the Librarian's intent classifier (`query_route_prompt` + friends) correctly route a question to the memory layer that best serves it? And does routing-aware retrieval beat flat "query all layers" retrieval?

**The concrete hypothesis:** different question types want different memory layers.
- **Single-hop (cat 1)** — single factual span. Vector search over raw turns wins.
- **Multi-hop (cat 3)** — chain facts across sessions. KG traversal (subject → predicate → object chains) beats flat vector.
- **Temporal (cat 2)** — "when did X". Session catalog (timeline-indexed) + supersede chains beat flat vector.
- **Open-domain (cat 4)** — world knowledge + dialogue. Crystals (session digests) + Reference Desk (cross-session insights) add value that raw vector misses.

If the Librarian's router correctly matches question type → layer, we should see per-category improvements that add up to a better overall score than flat thorough mode. If it doesn't, there's either a bug in the router or the layers aren't differentiated enough for routing to matter.

## Why LoCoMo is the right dataset for this

LoCoMo hands us ground-truth category labels on every QA. We don't have to build synthetic routing fixtures (which is what Axis B tried to do and failed). We run the same 1,540-question set three ways and compare.

## The three configurations

| Config | Description |
|--------|-------------|
| **baseline (vector-only)** | `VectorMemory.search(query, limit=10)` only. No routing. Same as Phase A baseline. |
| **flat-thorough** | `retrieve(strategy="thorough", sources={all 5 layers}, limit=10)` — queries every layer, fuses via RRF. No router; everyone gets the same treatment. |
| **router-directed** | Classify intent first (`classify_intent` + `query_route_prompt`), then `retrieve(strategy="custom", memory_layers=[...routed layers...])` with the router's choice. |

We want to know:
- Is flat-thorough better than vector-only? (Expected: yes on multi-hop + open-domain; wash on single-hop.)
- Is router-directed better than flat-thorough? (Expected: yes on token cost, maybe wash on quality. The routing value is **cost**, not recall.)
- Where does router-directed lose to flat-thorough? (Expected: when the router miscategorises.)

## What needs to exist before this is runnable

1. **Phase A baseline** — published LoCoMo vector-only numbers.
2. **Full retrieval stack on Fedora** — archive, catalog, crystals, KG all populated during ingest. The supersede benchmark spec calls out the ingest-cost concern; this benchmark inherits it. Cache the populated stores between runs.
3. **`classify_intent` end-to-end on LoCoMo questions.** Currently `taosmd/intent_classifier.py::classify_intent` returns `{"intent": str, "layers": list[str]}` (verify). Need an integration test: feed 20 LoCoMo questions, assert classification is sensible. This is the single biggest risk — the classifier was trained/tuned on assumed agent-session phrasing, not LoCoMo's third-person narrative phrasing ("When did Melanie paint a sunrise?"). Validate early.
4. **Layer-level metrics.** For router-directed, we need to log per-layer retrieval hit rates. Add `record["layers_queried"]: list[str]` and `record["layer_sources"]: dict[str, int]` (count of retrieved items from each layer).

## File to create

`benchmarks/routing_benchmark.py` — variant of `locomo_runner.py`. Key additions:

- Accept `--config` = `vector-only` | `flat-thorough` | `router-directed`.
- When `--config=router-directed`, call `classify_intent(query)` first and pass `memory_layers=result["layers"]` to `retrieve()`.
- Track per-QA: `category`, `routed_layers`, `layer_source_counts`, `f1`, `bleu1`, `judge`, `tokens_used`.
- Aggregate: per-category scorecard × 3 configs. Also per-category token cost.

## Scorecard shape (what we publish)

```
Category       vector-only              flat-thorough            router-directed
               F1    Judge  tokens      F1    Judge  tokens      F1    Judge  tokens
single-hop     0.xx  0.xx   N           0.xx  0.xx   N           0.xx  0.xx   N
temporal       0.xx  0.xx   N           0.xx  0.xx   N           0.xx  0.xx   N
multi-hop      0.xx  0.xx   N           0.xx  0.xx   N           0.xx  0.xx   N
open-domain    0.xx  0.xx   N           0.xx  0.xx   N           0.xx  0.xx   N
------
overall        0.xx  0.xx   N           0.xx  0.xx   N           0.xx  0.xx   N

Router accuracy (vs expected layer for each category):
  single-hop → vector          [N/M correct]
  temporal  → catalog+kg       [N/M correct]
  multi-hop → kg               [N/M correct]
  open-domain → crystals       [N/M correct]
```

The router-accuracy block is the most interesting part. It's what tells you whether the routing logic matches our hypothesis about which layer serves which question type.

## Success criteria

**Honest bar:**
- `flat-thorough` ≥ `vector-only` by ≥2 F1 points overall. (Validates that non-vector layers pull weight.)
- `router-directed` tokens ≤ 0.6× `flat-thorough` tokens. (Routing's value is cost savings.)
- `router-directed` F1 within 1 point of `flat-thorough` F1. (Router doesn't sacrifice quality for speed.)

**Dream bar:** `router-directed` beats `flat-thorough` on quality too (router chose better layers than fusion).

**Failure case A:** `flat-thorough` is worse than `vector-only`. Means fusion is hurting (KG/catalog/archive adding noise, not signal). Investigate per-layer hit rates; likely fix is to improve the adapters or change RRF weights.

**Failure case B:** Router routes to the wrong layer. Measure with the router-accuracy block; fix is prompt engineering or replacing the classifier.

**Failure case C:** Layers are all empty because LoCoMo ingest didn't populate them. Ingest side-effect problem — fix the ingest wiring.

## Open questions

1. **What's "the right layer" for each LoCoMo category?** The hypothesis table above is our best guess. Should we first run `flat-thorough` with per-layer attribution, look at which layer actually contributed the winning fact per category, and only then build the router against empirical routing? That would be more honest than assuming our architecture-level intuition is correct.
2. **Is the intent classifier a prompt or a heuristic?** `taosmd/intent_classifier.py` — audit before building. If it's a hand-written heuristic, the evaluation measures heuristic quality. If it's an LLM prompt, the evaluation measures prompt quality. They need different follow-up actions.
3. **Do we run all three configs on all 1,540 QAs, or sample?** Full run with 3 configs at concurrency=3 is ~3-4h on Fedora (reusing populated stores). Doable overnight.

## Rough timeline

- 3h — audit intent_classifier.py, run 20-QA classification sanity check
- 4h — build runner (mostly parallel to locomo_runner.py)
- 6-8h — first full run (3 configs)
- 3h — analyse, write up scorecard

Total: ~2 working days + overnight run. Start after supersede benchmark lands so we can share the populated stores.
