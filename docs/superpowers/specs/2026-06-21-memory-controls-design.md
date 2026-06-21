# Memory Controls Design

Goal: expose every taOSmd memory lever as a runtime-configurable control, surfaced as live UI in the standalone dashboard and as a stable API contract for the taOSmd app inside taOS, with the README documenting every control (what it does, pros and cons, resource cost). The `prefer_verified` recall gate ships on by default (already flipped, fully tri-judge evidenced).

## Architecture

Four layers, one shared schema.

1. Settings store (backend): a `controls` section in `~/.taosmd/config.json`, with `get_controls()` / `set_controls()` helpers following the existing `config.py` pattern (`get_memory_model`/`set_memory_model`, `get_default_recipe`/`set_default_recipe`). The core retrieve/recall path reads these as defaults; an explicit per-call API argument still overrides the stored setting.
2. API: `GET /controls` returns the current resolved settings plus the schema (from `profiles.profiles_schema()`); `PUT /controls` validates and persists an update. Both respect the `managed_by=taos` guard: when taOS owns config, the dashboard write path is disabled (the taOS app owns settings), reads still work.
3. Standalone dashboard: a Settings panel added to the current read-only inspector. Layout = the three profiles (Minimal / Quality / Integrity) as one-tap presets, plus an "Advanced" expander with one toggle/select per individual control. Offline, no CDN, accessible (ARIA, labels, keyboard nav), per the project design skills.
4. taOS app: builds its settings UI against the same `GET/PUT /controls` API + `profiles_schema()`. The handoff is the schema + the two endpoints, delivered to @taOS via `docs/INTEGRATION-memory-config.md` and the bus.

## Controls (the shared schema)

Each control declares: id, label, type, config key, default, category, cost, and a one-line description. Honest semantics matter: some are store/recipe-level, some per-query, one is consumer-side.

- `prefer_verified` (recall gate): off / prefer_verified / strict. Default `prefer_verified` (on). Cost: none at query time; needs the verify-pass populated to do anything (safe no-op otherwise). Category integrity.
- `reranker` (retrieval): off / bge-v2-m3 / ms-marco-MiniLM. Default off (tier-gated; on for Quality). Cost: a cross-encoder pass per query plus a model download; the F-013 accuracy win where the hardware affords it.
- `late_interaction` (MaxSim retrieval): on / off. Default off. Cost: about 110 ms/query on a 16-core CPU; lifts evidence recall 0.64 to 0.85, no GPU or reranker needed.
- `binary_quant`: on / off. Default off. Cost: none (a win); 32x smaller vectors, recall-neutral; for SBC / low-memory tiers.
- `fusion`: rrf / mem0_additive / boost. Default rrf (tier-gated). Cost: none; ranking-strategy choice.
- `adjacent_turns`: integer 0-2. Default 2 on capable tiers, 1 on Pi-class. Cost: wider context window per hit; worth about +0.089 on LoCoMo at 2.
- `embedder`: minilm-onnx / arctic-embed-s. Default arctic-embed-s for fresh low-tier installs (store-level, set at setup, not per-query). Cost: a one-time model fetch; +0.057 judged retrieval over MiniLM at the same dim and latency.
- `self_verify` (consumer-side): documented recommendation, NOT a core toggle. taOSmd retrieves; the consumer generates answers, so a CoVe-style self-verification pass belongs in the consumer's answer-gen. The dashboard shows it as informational (the 74.6% verified-answer config = reranker on + self-verify in your answer-gen), never as a switch that silently does nothing.

Presets:
- Minimal: arctic, no rerank, no late-interaction, prefer_verified off. Fastest/lightest.
- Quality: arctic + reranker + (recommend self-verify in answer-gen), prefer_verified off. Best accuracy where hardware affords.
- Integrity: Quality + prefer_verified on. Best provenance.

(Note: prefer_verified is now on by default globally, so the Minimal preset is the one that turns it off.)

## README documentation requirement

A dedicated "Configuration and controls" section in the README lists every control above with: what it does, when to turn it on, the trade-off (pros and cons), and the concrete resource cost (latency, memory, model download, tier suitability). Each is cross-linked to its benchmarks.md measurement. This is a first-class deliverable, reviewed for completeness against the schema.

## Out of scope

- A core answer-generation path for self_verify (it stays a documented consumer-side recommendation; taOSmd is a memory layer).
- Per-agent or per-project control overrides (global install settings only for v1).
- A settings-history/audit trail (the controls write is a simple last-write-wins config update).

## Testing

- Backend: `get_controls`/`set_controls` round-trip, defaults, validation of bad values, `prefer_verified` resolves from config when no per-call arg.
- API: `GET /controls` shape (settings + schema); `PUT /controls` persists + validates; `managed_by=taos` disables PUT.
- Dashboard: a smoke check that the Settings panel renders and the preset buttons post the right payload (Playwright if wired, else a DOM/string assertion on the served HTML).
- Schema parity: the dashboard + the @taOS contract both derive from `profiles_schema()`, asserted in one test so they cannot drift.
