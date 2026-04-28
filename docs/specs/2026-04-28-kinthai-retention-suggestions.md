# Kinthai retention-scoring suggestions — captured for future-work evaluation

External feedback from `@kinthaiofficial` on tinyagentos issue #182 (memory consolidation), 2026-04-28. Three suggestions for `taosmd/retention.py`'s consolidation/decay scoring weights.

## The suggestions

1. **Bump `relevance` weight from 0.30 → 0.35–0.40.** Rationale per Kinthai: relevance is the single best predictor of whether a memory will be useful in future sessions.

2. **Drop `frequency` weight from 0.24 → 0.15–0.20.** Rationale: high-frequency memories are often trivial (greetings, common phrases) — over-weighting frequency promotes noise.

3. **Add a negative signal for retrieved-but-unused memories.** Rationale: catches the "seemed relevant in retrieval, agent didn't actually reference it in output" failure mode.

## Why we haven't tested these in LoCoMo (and shouldn't)

The LoCoMo benchmark exercises **retrieval-augmented QA over freshly-ingested conversations**. Retrieval at query time uses vector similarity + keyword fusion + cross-encoder rerank. The retention/consolidation scoring weights are **not consulted** during retrieval — they only affect the nightly consolidation pipeline (Light/REM/Deep) which decides what stays "hot" in long-term memory.

LoCoMo's structure: each conversation's turns are ingested fresh, queries run immediately. Nothing has decayed; nothing has been promoted; nothing has been evicted. Changing weights in `retention.py` has zero effect on the LoCoMo numbers.

Running a LoCoMo bench with modified weights would burn ~5h to measure no change — exactly the kind of "blindly add" we want to avoid.

## What would actually validate these suggestions

A benchmark that exercises long-horizon memory management:

- **Supersede chains** (already designed in `docs/specs/2026-04-16-supersede-benchmark-design.md`) — tests temporal-reasoning over contradictory facts across time. Retention's recency + relevance weighting is directly relevant.
- **Multi-store routing** (already designed in `docs/specs/2026-04-16-routing-benchmark-design.md`) — tests promote/demote decisions across vmem, KG, archive. Retention scoring weights determine what gets promoted.
- **Live taOS user data** (when a long-running deployment accumulates real-world session history). Closest to Kinthai's "future sessions" evaluation framing.

For the *negative-signal-on-unused* idea specifically: requires usage-tracking instrumentation we don't have today. A retrieved memory needs to be flagged in the response as "actually referenced" vs "ignored". Implementation paths:

- Per-question post-processing: ask the judge "did the answer rely on this passage?" — expensive (extra judge call per retrieved chunk).
- Token-overlap heuristic: count overlap between retrieved-passage tokens and predicted-answer tokens; high overlap = referenced. Cheap, noisy.
- Attention-extraction from a generator that exposes attention weights — accurate but limited to local generators with that hook.

## Decision

- **Capture the suggestions here**, don't ship them blindly.
- **Revisit when** either the supersede or routing benchmarks ship, OR when we have multi-day taOS user data to ablate against. Kinthai's framing ("predicts future-session usefulness") needs a benchmark that actually has multiple sessions per memory.
- The `negative-signal-on-unused` idea is the most architecturally interesting; if we ever build a usage-tracking layer, that benchmark would need to come with it.

## Provenance

- Comment ID: 4337347476
- URL: https://github.com/jaylfc/tinyagentos/issues/182#issuecomment-4337347476
- Linked external write-up: https://blog.kinthai.ai/why-character-ai-forgets-you-persistent-memory-architecture
