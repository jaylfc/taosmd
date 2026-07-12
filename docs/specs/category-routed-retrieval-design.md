# Category-routed retrieval: design proposal

**Workstream:** follow-up to the Jul 10-13 LoCoMo sweep campaign (E-027/E-028, N-020/N-021/N-024). The campaign's main structural finding is a category-tension: the levers that lift MultiHop cost Temporal, so nothing shipped. This spec proposes the obvious next move, routing per-query instead of picking one global config.
**Status:** proposal for review. Nothing here is implemented and nothing should be until the evaluation plan below is agreed.
**Evidence base:** all numbers are LoCoMo, external qwen3:4b judge, from the sweep campaign. The per-category numbers are subset-200 unless marked n=500, and the caveat in section 2 applies to every one of them.

## What we're proposing

A lightweight query classifier inside the retrieval path. Each incoming `retrieve()` gets routed to one of two configs:

- **Wide** (MultiHop-shaped questions: two entities, comparative "both/and" forms, "how are X and Y related"): `candidate_top_k` 100, final top-k 20, possibly timeline-formatted context (open question 2).
- **Base** (everything else, explicitly including temporal "when did" and date-shaped questions): the shipped config, untouched.

Default off, behind a new runtime control. Read-path only. No model changes, no write-path changes.

## The evidence, and the caveat that outranks it

The campaign showed MultiHop and Temporal want opposite retrieval. Wider retrieval alone (N-020, k100/top20) lifted MultiHop +0.093 (0.209 to 0.302) and cost Temporal -0.048 (0.619 to 0.571), net overall neutral (0.545 vs 0.535). Timeline formatting plus wide (N-021 arm T2) pushed MultiHop to 0.372, the session max, but dropped Temporal to 0.508 and OpenDomain from 0.615 to 0.385, again net neutral (0.555). Every context-based MultiHop lever traded the same way. Meanwhile the prompt-framing route died outright: the force-empty-think subset standout failed its pre-registered n=500 confirm on all three legs (N-024), it was subset noise.

If the tension is real, routing is the natural answer: give the wide config only to the queries that benefit from it and keep Temporal on base, and the trade stops being a trade.

**The caveat:** the N-024 confirm also showed that subset-200 per-category numbers are unreliable. The fresh n=500 anchor shifted base MultiHop from 0.209 to 0.279, Temporal from 0.619 to 0.531, and OpenDomain from 0.615 to 0.344. So the per-category deltas above are directional at best, and the routing hypothesis inherits that. It is plausible that part of the "tension" is category-level judge noise at n=200. This spec therefore treats routing as a hypothesis to kill or confirm at n=500, not a win to wire in. No subset number gets trusted again.

## Classifier options

1. **Regex/heuristic (recommended).** The MultiHop question shapes in the dataset are regexable: two proper-noun-ish entities, "both", "and ... and", "how are X and Y", comparative forms. Free, deterministic, unit-testable, zero latency, and honest about what it is. We already run exactly this kind of classifier per query: `taosmd/intent_classifier.py` (`classify_intent`, regex patterns, microseconds, called inside `retrieve()` today). The router should extend that module with a multi-hop shape check rather than grow a new one.
2. **ONNX embedder with category prototypes (fallback experiment).** Embed the query with the existing dense embedder, cosine against a handful of category prototype vectors. Cheap, no LLM, catches phrasings the regexes miss. Worth trying only if the oracle arm (section 5) shows classifier quality is the binding constraint.
3. **LLM micro-call (rejected by default).** Adds per-query latency and a model dependency, which is wrong for our audience (low-end, offline, Pi-class tiers). Not unless both cheaper options fail an oracle gap they should have closed.

The truly minimal first version, per the codebase's `# upgrade-path:` convention, is option 1 cut down to a single rule: route wide only when the query mentions two or more entities. Start there, mark the upgrade path in a comment, and let the oracle arm say whether more classifier is even worth writing.

## Where it lives

- **Control registry:** a new `Control` in `taosmd/controls.py`, id `category_routing`, category "quality", scope "runtime", type "bool", `config_key="controls.category_routing"`, default `False`. It rides the existing plumbing for free: `config.get_controls` / `set_control`, the GET/PUT /controls API, and the dashboard all derive from the registry.
- **Resolution point:** the natural hook is in `taosmd/api.py` `search()`, right after `rc = _apply_runtime_overrides(recipe.retrieval, _config.get_runtime_overrides(data_dir))` and before the `_retrieve(...)` call. When the control is on and the classifier fires, override `rc["candidate_top_k"]` to 100 and the effective limit to 20 for this query only. This keeps the precedence story identical to the existing controls: recipe is the baseline, persisted runtime controls override it, and routing is just one more per-query override.
- **Classifier:** new patterns and a `classify_category()` (or an extra return from `get_search_strategy`) in `taosmd/intent_classifier.py`.
- **One wrinkle to settle:** inside `retrieve()`, `limit` is overridden by `effective_fanout(agent, worker_capabilities)` when an agent is given. A routed top-k of 20 has to compose with per-agent fanout, probably `max(fanout, 20)` when the router fires, but that interaction needs a decision, not an accident.

## Evaluation plan (pre-registered, before any build lands)

n=500 LoCoMo, external qwen3:4b judge, fresh same-n base anchor in the same chain (the N-024 lesson, now mandatory). Three arms:

1. **base**: shipped config, the anchor.
2. **routed**: the classifier live, wide config on routed queries.
3. **oracle-routed**: gold category labels stand in for the classifier, i.e. the ceiling for this whole idea.

**Kill criterion:** ship default-on only if routed >= base +0.02 overall AND no category regresses more than 0.03, both at n=500. Anything less and it stays default-off or dies.

The oracle arm is the diagnostic. If oracle-routed clears the bar but routed does not, the configs work and the classifier is the constraint (try option 2). If oracle-routed itself fails, the per-category configs do not survive n=500 and no classifier will save them; kill the workstream and record the negative result. That second outcome is live, given the caveat.

## Costs and risks

- **Misclassification.** A temporal question routed wide inherits the Temporal cost we measured. The classifier must be precision-biased: when unsure, stay on base. The failure mode of a missed MultiHop question is the status quo; the failure mode of a false positive is a measured regression.
- **Token cost.** The wide config sends more context to the generator on routed queries (top-k 20 vs 5-10, plus adjacency). On a Pi-class tier that can over-budget a tiny generator. Routed queries should be a small fraction of traffic, but the eval should log the routed-query rate and mean context size.
- **Complexity.** taOSmd's minimalism principle applies. A router plus per-category configs is a real complexity step; the two-entity heuristic with an `# upgrade-path:` comment is the honest first rung, and the oracle arm decides whether to climb further.
- **Zero-loss is untouched.** Routing is read-path only. Nothing is written, mutated, or dropped from the store.

## What this does not do

- No reranker changes.
- No new models (option 1 needs none; option 2 reuses the existing embedder).
- No write-path or ingest changes.
- Benchmarks unaffected by default: the control ships off, so every existing number stands.
- Not a prompt-framing lever. That route was tested and killed (N-024).

## Open questions for Jay

1. **Classifier choice.** Happy with regex-first, embedder-prototype as the fallback experiment, LLM rejected? And is the two-entity heuristic minimal enough to be the v1, or too crude to bother benching?
2. **Timeline formatting in the wide config?** It bought the biggest MultiHop reading (0.372) but also hurt OpenDomain hard in T2 (0.615 to 0.385, subset). Plain wide (N-020) left OpenDomain flat. My lean is plain wide for the routed config and leave timeline out, but we could run both as sub-arms if the GPU time is acceptable.
3. **Bother at all?** The n=500 anchor shift means the category-tension itself is partly unproven at scale. The oracle arm answers this cheaply-ish (one extra n=500 arm), but if you'd rather spend the GPU budget elsewhere, the honest alternative is to shelve this until something re-confirms the tension at n=500.
