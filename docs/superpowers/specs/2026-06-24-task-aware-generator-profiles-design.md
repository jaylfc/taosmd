# Task-aware generator profiles

Status: design approved, ready for implementation plan
Date: 2026-06-24
Author: jaylfc

## Problem

The answer-generation model (the "generator") is currently bundled inside each
retrieval recipe as a single field (`generator={"model": "ollama:qwen3.5:9b"}`
in `taosmd/recipes.py`), chosen by hardware tier. But the best generator is not
only a function of hardware: it is a function of the workload.

We have measured this. `docs/benchmarks.md` (lines 515 to 526) already documents
a per-workload generator table at the 12 GB tier: `qwen3.5:9b` is best Overall
(the production default), `llama3.1:8b` is best on LoCoMo Single-hop, and
`mistral-small3.2` is best on Temporal. The N-017 cascade added another row:
`gemma4:12b` wins LongMemEval single-fact QA at full-500 (53.8 strict Qwen /
61.4 cross-family llama, parity with MemOS-lossless) but UNDERperforms
`qwen3.5:9b` on LoCoMo (0.63 vs 0.68, N-007) and BEAM (46 vs 49).

So the generator win is genuinely task-dependent. Flipping the global default to
`gemma4:12b` would trade two benchmarks for one. The fix is to make the generator
a workload-selectable axis, with a safe default, instead of a single global
choice. This productizes the per-workload table that already lives in the docs.

A generator choice is also tier-dependent: the best model for a workload differs
by hardware, and the smallest devices may run no local generator at all. So a
profile maps a workload to the right generator PER TIER, including the option of
no local LLM (retrieval-only) on tiny devices.

## Goals

- Let a user opt into a generator tuned for their workload (for example
  single-fact retrieval QA) without changing the safe default for everyone else.
- Cover the whole hardware range, low-end included: a profile resolves to the
  best generator for the user's tier, or to retrieval-only on devices too small
  to run a worthwhile local generator. Low-end / SBC users are the target
  audience, so they must get workload-aware behaviour too, not just a fallback.
- Keep today's behaviour as the default, never silently changed.
- Make the set of profiles a data-driven, extensible registry, so a new
  workload-to-model mapping is one data row, not a code change.

## Non-goals (YAGNI)

- No automatic workload detection. Selection is explicit, in keeping with the
  "never silently enabled" principle.
- No per-query model swapping. The generator is loaded into VRAM; swapping per
  query is expensive and risks the 12 GB model-swap stall. Selection is per
  deployment / per agent.
- No remote or cloud generation in this spec. Both need a provider-dispatch layer
  that does not exist today (generation calls are Ollama-direct, scattered across
  about seven modules), plus, for cloud, API-key/secrets handling and an
  offline-first privacy story. Those are deferred to a follow-up "generator
  backend abstraction" spec. This spec covers only the `local` and `none`
  backends, which already exist in the codebase. The profile data model is
  designed so a `remote` or `cloud` backend slots in later without a redesign.

## Backends in scope

A profile's per-tier value is a generator backend. This spec supports two:

- `local`: a `provider:model` string for a model run by the local Ollama, for
  example `ollama:qwen3.5:9b`. This is what every recipe uses today.
- `none`: the empty string `""`, meaning no local generator (retrieval-only).
  The `lite-pi` recipe already ships `generator={"model": ""}`, so this is an
  existing, supported state, used on Pi-class devices.

`remote` and `cloud` are explicitly out of scope here (see Non-goals).

## Approach (chosen: orthogonal, tier-aware profile axis)

A new generator-profile registry sits ALONGSIDE the retrieval recipe rather than
inside it. The retrieval recipe keeps choosing retrieval-by-tier; the active
generator profile chooses which model answers, resolved for the detected tier.
This avoids a combinatorial explosion of recipes, is the most literal reading of
"extensible registry", and reuses the per-recipe generator field as the fallback.

Two alternatives were considered and rejected: workload-variant recipes (combin-
atorial explosion, near-duplicate recipes, muddied per-recipe score semantics),
and folding profiles into `resolve_recipe` (cleaner conceptually but changes the
resolution signatures, more invasive). See the decision log.

## Data model: `taosmd/generator_profiles.py`

A dataclass plus a module-level registry, mirroring `recipes.py`.

```python
@dataclass
class GeneratorProfile:
    id: str               # "balanced", "factual-recall"
    label: str            # human label for UIs
    workload: str         # what this profile is tuned for, in plain words
    models: dict          # tier -> backend value: a "provider:model" string
                          #   ("" means the `none` / retrieval-only backend)
    evidence: dict        # {"scores": {...}, "source": "docs/benchmarks.md ..."}
    notes: str            # caveats (when NOT to use it)
```

The `models` map is keyed by the existing tier vocabulary from `recipes.tier_of`
(`gpu-12gb`, `gpu-8gb`, `gpu-4gb`, `pi-npu`, `cpu`). The keys present ARE the
tiers the profile covers; a tier absent from the map falls through to the recipe
generator at resolution (see below). This replaces a single `min_tier` field: a
profile declares exactly which tiers it serves and with what model, instead of a
floor plus a fallback.

Registry API, matching the recipes module:

- `get_profile(profile_id) -> GeneratorProfile | None`
- `list_profiles() -> list[GeneratorProfile]`
- `default_profile_id() -> str` returns `"balanced"`

Seed profiles (only these two ship):

- `balanced` (default). Mirrors the existing recipe generators exactly, so the
  default reproduces today's behaviour on every tier:
  `{gpu-12gb: ollama:qwen3.5:9b, gpu-8gb: ollama:qwen3.5:9b,
  gpu-4gb: ollama:llama3.1:8b, pi-npu: "", cpu: ""}`.
  Workload: multi-session conversational and long-context recall (the LoCoMo /
  BEAM leader). Evidence: the LoCoMo full-1540 tri-judge and BEAM numbers in
  benchmarks.md.
- `factual-recall`. Covers the tiers where we have a defensible factual pick:
  `{gpu-12gb: ollama:gemma4:12b, gpu-8gb: ollama:llama3.1:8b,
  gpu-4gb: ollama:llama3.1:8b}`.
  - `gpu-12gb` = gemma4:12b is confirmed at full-500 (53.8 / 61.4, N-017).
  - `gpu-8gb` = llama3.1:8b. gemma4:12b does not fit 8 GB, and llama3.1:8b is the
    documented factual leader over qwen3.5:9b (LoCoMo Single-hop 0.65 vs 0.53,
    benchmarks.md line 515), so it is the better factual pick at the tier where
    both fit. PROVENANCE CAVEAT: this is a transfer from a LoCoMo Single-hop
    measurement, NOT a LongMemEval-at-8GB measurement; LongMemEval (this
    profile's defining benchmark) has qwen at 42.8 / 51.2 but no llama3.1:8b
    number yet, and the two benchmarks can diverge. Flagged provisional pending a
    LongMemEval-at-8GB confirm (a queued low-tier generator bench).
  - `gpu-4gb` = llama3.1:8b is the documented factual leader (same line 515) and
    is already the `fast-8b` 4 GB recipe generator (fits 4 GB with offload). Same
    transfer caveat as 8 GB: a documented-leader carry-over, not a 4 GB
    measurement.
  - `pi-npu` is intentionally absent: a Pi has no viable local factual generator,
    so it falls through to the recipe (retrieval-only there). A future low-tier
    bench can add it as data.
  - Notes must state plainly that this profile loses on conversational and
    long-context workloads (LoCoMo 0.63 vs 0.68, BEAM 46 vs 49), so it is the
    right pick only for single-fact QA. Evidence: the full-500 LongMemEval
    numbers and the N-017 research-report entry.

Honest limitation to record in the spec and notes: at `gpu-4gb`, `balanced` and
`factual-recall` resolve to the same model (`llama3.1:8b`), because it is the one
viable 4 GB generator we have, so 4 GB has no real differentiation yet (the slot
exists for a better 4 GB factual model after a low-tier bench). At `gpu-8gb` they
DO differ (`balanced` = qwen3.5:9b, `factual-recall` = llama3.1:8b), which is the
intended behaviour, but that 8 GB factual pick rests on the transfer caveat
above, not a LongMemEval-at-8GB measurement.

## Selection, storage, and resolution

Storage (decision: store the profile id, resolve the model at runtime):

- Global default: a top-level `generator_profile` key in `~/.taosmd/config.json`,
  with `config.get_generator_profile(data_dir)` and
  `config.set_generator_profile(profile_id, clear=False, data_dir)`, mirroring the
  existing `get_memory_model` / `set_memory_model` pair (same atomic `_read` /
  `_write`, same unset-returns-None contract).
- Per-agent override: `agents.get_agent_generator_profile(name, data_dir)` and
  `agents.set_agent_generator_profile(name, profile_id, data_dir)`, mirroring the
  per-agent recipe storage (`get_applied_recipe` / `set_agent_recipe_config`).

Resolution: `generator_profiles.resolve_generator(agent, data_dir) -> str`
returns the `provider:model` string (or `""` for retrieval-only) by this
precedence, most specific first:

1. Explicit user model pin (`config.get_memory_model`), when set. The advanced
   escape hatch for an exact model regardless of profiles.
2. Active generator profile, per-agent override before global default. The
   profile resolves via `models.get(detected_tier)`, where the tier comes from
   `recipes.tier_of(recipes.local_probe())`. A present key (including `""`) is
   used directly. An absent key falls through to the next level.
3. The active recipe's `generator.model` (today's default path).
4. Empty string (retrieval-only).

There is no separate tier guard: the `models` map already encodes which tiers a
profile serves, so a profile is never applied to a tier it does not list. The
model-presence preflight from PR #174 runs for the resolved local model as it
does today; a resolved `""` means retrieval-only and skips generation.

### Compatibility change: stop auto-seeding `memory_model` from the recipe

Today `apply_recipe` calls `set_memory_model(recipe.generator)` when no model is
set, to make a fresh install's generator visible. With profiles, that auto-seed
would masquerade as an explicit user pin at precedence level 1 and would shadow
any profile the user later selects. So:

- `apply_recipe` stops auto-seeding `memory_model`. `memory_model` becomes an
  explicit-user-pin-only field.
- The recipe generator is read live at precedence level 3 by `resolve_generator`,
  not copied into config.
- Every current site that reads `config.get_memory_model` to choose the answer
  model is routed through `resolve_generator` instead, so the fresh-install
  default still resolves to the recipe generator (now via level 3). The
  implementation plan must enumerate and migrate these read sites; the
  no-regression snapshot test guards the result.

## Surfacing

- Controls registry (`taosmd/controls.py`): add a `generator_profile` control,
  `type="choice"`, `category="quality"`, `scope="consumer"`, `choices` populated
  from `list_profiles()` ids at module load, `default="balanced"`, with
  `cost` / `pros` / `cons` / `benchmarks_anchor` from the profile data. It then
  appears in `GET/POST /controls` and the dashboard Settings panel with no
  dashboard code change; `validate_control` already enforces choice membership.
- CLI: `taosmd generator-profile list | show <id> | set <id> [--agent A]`,
  mirroring the recipe CLI verbs. `show` displays the per-tier model map and
  which entry applies to the current detected tier.
- Docs: a short "Generator profiles" section in README.md and a cross-reference
  from the per-workload table in benchmarks.md, marking `balanced` as the default
  and `factual-recall` as opt-in with the task-dependency caveat.

## Evidence discipline

- Every profile `evidence` entry cites a `docs/benchmarks.md` (or research
  report) provenance string, per tier where the pick is tier-specific. No
  invented numbers, same rule as `recipes.py`.
- `factual-recall` carries both its LongMemEval win and its LoCoMo / BEAM losses,
  so the caveat travels with the data.
- Adding a tier entry or a new profile (for example a measured 4 GB factual
  model, or `temporal` = `mistral-small3.2`) requires subset-then-full
  confirmation before it ships, per the benchmarks.md line-582 rule. The registry
  makes adding one a data change; the evidence bar is unchanged.

## Testing

- Registry integrity: every profile has all required fields and a non-empty
  `models` map with valid tier keys; `default_profile_id()` returns `"balanced"`.
- Backward-compat guard: `balanced.models` equals the per-tier recipe generators
  (qwen3.5:9b at 12/8 GB, llama3.1:8b at 4 GB, "" on Pi), so the default can
  never drift from today's behaviour silently.
- Resolution precedence: pin beats profile beats recipe; per-agent profile beats
  global profile.
- Tier-map resolution: `factual-recall` resolves to gemma4:12b at `gpu-12gb` and
  llama3.1:8b at both `gpu-8gb` and `gpu-4gb`; at `pi-npu` (absent key) it falls
  through to the recipe generator (retrieval-only there).
- Retrieval-only: a profile entry of `""` resolves to `""` and the generation
  path is skipped.
- No-regression snapshot: with no profile and no pin set, `resolve_generator`
  equals today's recipe generator for each tier.
- Controls round-trip: the `generator_profile` control validates a known id,
  rejects an unknown id, and surfaces in the schema.

## Decision log

- Breadth: extensible, data-driven registry, seeded with two profiles; more are
  data.
- Integration: orthogonal profile axis that overrides the recipe generator
  (Approach A), over workload-variant recipes (B) and folding into resolution (C).
- Tier coverage: a profile is tier-aware (a per-tier `models` map), not a single
  model with a floor, so low-end devices get a workload-appropriate generator or
  an explicit retrieval-only entry rather than only a fallback.
- Backends: `local` and `none` in scope now; `remote` and `cloud` deferred to a
  follow-up generator-backend-abstraction spec (provider-dispatch layer, secrets,
  privacy). The `models` map accommodates the new backend kinds without redesign.
- Storage: store the profile id and resolve the model at runtime, over writing
  the model through to `memory_model`.
