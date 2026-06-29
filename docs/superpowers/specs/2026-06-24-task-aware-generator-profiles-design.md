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

## Goals

- Let a user opt into a generator tuned for their workload (for example
  single-fact retrieval QA) without changing the safe default for everyone else.
- Keep `qwen3.5:9b` as the default generator, never silently changed.
- Make the set of profiles a data-driven, extensible registry, so a new
  workload-to-model mapping is one data row, not a code change.
- Stay backward-compatible: with no profile selected, behaviour is identical to
  today.

## Non-goals (YAGNI)

- No automatic workload detection. Selection is explicit (the user picks), in
  keeping with the "never silently enabled" principle. Auto-detection would need
  to infer the workload from the user's data and query mix, which we cannot do
  reliably and which would surprise users.
- No per-query model swapping. The generator is loaded into VRAM; swapping it per
  query is expensive and risks the 12 GB model-swap stall. Selection is per
  deployment / per agent, not per query.
- No new benchmarking for the two seed profiles. They rest on existing,
  confirmed numbers. Future profiles require fresh confirmation (see Evidence).

## Approach (chosen: orthogonal profile axis)

A new generator-profile registry sits ALONGSIDE the retrieval recipe rather than
inside it. The retrieval recipe keeps choosing retrieval-by-tier; the active
generator profile overrides which model answers. This avoids a combinatorial
explosion of recipes (5 recipes + N profiles, not 5xN), is the most literal
reading of "extensible registry", and reuses the existing per-recipe generator
field as the fallback.

Two alternatives were considered and rejected: workload-variant recipes (combin-
atorial explosion, near-duplicate recipes, muddied per-recipe score semantics),
and folding profiles into `resolve_recipe` (cleaner conceptually but changes the
resolution signatures, more invasive). See the decision log at the end.

## Data model: `taosmd/generator_profiles.py`

A dataclass plus a module-level registry, mirroring `recipes.py`.

```python
@dataclass
class GeneratorProfile:
    id: str            # "balanced", "factual-recall"
    label: str         # human label for UIs
    workload: str      # what this profile is tuned for, in plain words
    model: str         # "provider:model", e.g. "ollama:qwen3.5:9b"
    min_tier: str      # smallest tier whose VRAM fits this model
    evidence: dict     # {"scores": {...}, "source": "docs/benchmarks.md ..."}
    notes: str         # caveats (when NOT to use it)
```

Registry API, matching the recipes module:

- `get_profile(profile_id) -> GeneratorProfile | None`
- `list_profiles() -> list[GeneratorProfile]`
- `default_profile_id() -> str` returns `"balanced"`

Seed profiles (only these two ship):

- `balanced`: model `ollama:qwen3.5:9b`, min_tier `gpu-8gb`, workload
  "multi-session conversational and long-context recall". This is the default
  and reproduces today's behaviour. Evidence: the LoCoMo full-1540 tri-judge and
  BEAM numbers already in benchmarks.md.
- `factual-recall`: model `ollama:gemma4:12b`, min_tier `gpu-12gb`, workload
  "single-fact retrieval QA (LongMemEval-style)". Notes must state plainly that
  it loses on conversational and long-context workloads (LoCoMo 0.63 vs 0.68,
  BEAM 46 vs 49), so it is the right pick only for single-fact QA. Evidence: the
  full-500 LongMemEval numbers (53.8 / 61.4) and the N-017 entry in the research
  report.

Tier ordering for `min_tier` comparisons reuses the existing tier vocabulary from
`recipes.tier_of` (`gpu-12gb` > `gpu-8gb` > `gpu-4gb` > `pi-npu` > `cpu`). A
small ordered list in the profiles module ranks them; no new tier concept.

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

Resolution: a new `generator_profiles.resolve_generator(agent, data_dir) -> str`
returns the `provider:model` string by this precedence, most specific first:

1. Explicit user model pin (`config.get_memory_model`), when set. This is the
   advanced escape hatch for someone who wants an exact model regardless of
   profiles.
2. Active generator profile, per-agent override before global default. The
   profile id resolves to its `model` via `get_profile`.
3. The active recipe's `generator.model` (today's default path).
4. Empty string (the lite/no-LLM tier).

Tier guard: when a profile would be used, compare its `min_tier` to the detected
tier (`recipes.tier_of(recipes.local_probe())`). If the profile needs a larger
tier than is present, do NOT use it: fall through to the next precedence level
and surface a warning (returned to callers / shown in the CLI and dashboard).
We never silently load a model that will not fit. The model-presence preflight
from PR #174 runs for the resolved generator as it does today.

### Compatibility change: stop auto-seeding `memory_model` from the recipe

Today `apply_recipe` calls `set_memory_model(recipe.generator)` when no model is
set, to make a fresh install's generator visible. With profiles, that auto-seed
would masquerade as an explicit user pin at precedence level 1 and would shadow
any profile the user later selects. So:

- `apply_recipe` stops auto-seeding `memory_model`. `memory_model` becomes an
  explicit-user-pin-only field.
- The recipe generator is no longer copied into config; it is read live at
  precedence level 3 by `resolve_generator`.
- Every current read site of `config.get_memory_model` for the purpose of
  choosing the answer model is routed through `resolve_generator` instead, so the
  fresh-install default still resolves to the recipe generator (now via level 3,
  not via an auto-seeded level 1). The implementation plan must enumerate and
  migrate these read sites; the no-regression snapshot test (below) guards the
  result.

## Surfacing

- Controls registry (`taosmd/controls.py`): add a `generator_profile` control,
  `type="choice"`, `category="quality"`, `scope="consumer"`, `choices` populated
  from `list_profiles()` ids at module load, `default="balanced"`, with
  `cost` / `pros` / `cons` / `benchmarks_anchor` filled from the profile data.
  This makes it appear in `GET/POST /controls` and the dashboard Settings panel
  with no dashboard code change. `validate_control` already enforces choice
  membership.
- CLI: `taosmd generator-profile list | show <id> | set <id> [--agent A]`,
  mirroring the recipe CLI verbs.
- Docs: a short "Generator profiles" section in README.md and a cross-reference
  from the existing per-workload table in benchmarks.md, marking `balanced` as
  the default and `factual-recall` as opt-in with the task-dependency caveat.

## Evidence discipline

- Every profile `evidence` field cites a `docs/benchmarks.md` (or research
  report) provenance string. No invented numbers, same rule as `recipes.py`.
- `factual-recall` carries both its LongMemEval win and its LoCoMo / BEAM losses,
  so the caveat travels with the data and cannot be dropped by mistake.
- Future profiles (for example `temporal` = `mistral-small3.2`) require
  subset-then-full confirmation before being added, per the benchmarks.md
  line-582 rule. The registry makes adding one a data change, but the evidence
  bar is unchanged.

## Testing

- Registry integrity: every profile has all required fields, a non-empty model,
  and a valid `min_tier`; `default_profile_id()` returns `"balanced"`.
- Backward-compat guard: `get_profile("balanced").model` equals the 12 GB recipe
  generator string, so the default can never drift away from today's behaviour
  silently.
- Resolution precedence: pin beats profile beats recipe; per-agent profile beats
  global profile.
- Tier guard: selecting `factual-recall` on a `gpu-4gb` tier falls back to the
  recipe generator and emits a warning rather than loading gemma4:12b.
- No-regression snapshot: with no profile and no pin set, `resolve_generator`
  equals today's recipe generator for each tier.
- Controls round-trip: the `generator_profile` control validates a known id,
  rejects an unknown id, and surfaces in the schema.

## Decision log

- Breadth: extensible, data-driven registry (not just two hard-coded profiles,
  not the full four-workload set up front). Seeded with two profiles; more are
  data.
- Integration: orthogonal profile axis that overrides the recipe generator
  (Approach A), over workload-variant recipes (B) and folding into resolution (C).
- Storage: store the profile id and resolve the model at runtime, over writing
  the model through to `memory_model`. Keeps the selection semantic and keeps the
  explicit pin distinct.
