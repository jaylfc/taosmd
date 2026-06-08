# Recipes / Config Profiles: Design (SP1, the recipe core)

Status: approved design, ready for implementation plan
Date: 2026-06-08
Scope: Sub-project 1 of 6 (the recipe CORE). Later sub-projects are sketched in
"Sub-project map" so the seams line up, but only SP1 is specified here.

## Problem

taosmd has a rich set of retrieval and ingest levers (reranker, adjacent
neighbours, fusion strategy, candidate pool size, LLM extraction, librarian
fanout, generator model), and we have benchmarked which combinations win at
which compute tier. The current leader on LoCoMo-1540 is MaxSim+rerank
(`qwen3.5:9b --retrieval-top-k 50 --adjacent-turns 2 --reranker bge-v2-m3
--fusion mem0_additive`).

But two gaps make that knowledge unreachable in practice:

1. **A fresh install does not use the leader.** `retrieve()` defaults are
   `reranker=None`, `adjacent_neighbors=0`, fusion via VectorMemory's "boost",
   `limit=5`. The levers are parameters that nothing sets. The benchmarked
   recipe lives in docs and in benchmark scripts, not in the running code.
2. **There is no named, inspectable object for a configuration.** A user (or
   taOS) cannot list the available configurations, see their scores and
   trade-offs, pick one, or have one auto-selected for their hardware.

SP1 closes both gaps: a Recipe becomes a first-class named object, there is a
built-in registry of the recipes we have actually measured, and there is
plumbing that applies a resolved recipe inside `search()` and `ingest()` so a
fresh install runs a real recipe (tier-gated to the leader where affordable).

## Goals (SP1)

- A `Recipe` is a declared, typed config bundle with four sections (retrieval,
  ingest, generator, librarian/fanout) plus a metadata block (benchmark scores
  per judge, tier, pros, cons, estimated latency/footprint).
- A built-in registry of the recipes we have benchmarked, defined in code.
- Resolution + application: `search()` and `ingest()` resolve the active recipe
  (per-agent override, else global default, else a tier-safe fallback) and apply
  its fields to the existing `retrieve()` / ingest call sites. This is the change
  that makes "install uses a real recipe" true.
- The no-LLM-ingest lite path is one of the built-in recipes (decided: built
  inside SP1, not split out). It is a recipe whose ingest section disables LLM
  enrichment; everything else (archive, embed, retrieve) is unchanged.
- A declared JSON-Schema export of the bundle shape, so any consumer (the taOS
  Memory-framework manager, the dashboard) can render a recipe generically with
  no taosmd-specific UI code. This is the contract seam @taOS asked for.

## Non-goals (SP1)

- `recommend(device_info) -> ranked[recipe]` auto-pick. The schema carries the
  `tier` and footprint metadata that makes ranking possible, and the registry is
  tier-tagged, but the ranking call itself is SP2.
- The full taOS-facing API surface (list/get/set/create over HTTP/MCP/remote)
  is SP4. SP1 ships the Python API and the schema export it builds on.
- The dashboard recipe-selector page is SP5.
- New retrieval science. SP1 only wires recipes to levers that already exist and
  are already benchmarked. No new lever is introduced.

## Sub-project map (context only)

- **SP1 (this spec):** recipe core: object, registry, resolution+application,
  lite path, schema export.
- **SP2:** `recommend(device_info) -> ranked[recipe]` and tier inference.
- **SP3:** custom recipes (create/edit/delete) persisted in the config DB.
- **SP4:** taOS contract API across HTTP / MCP / CLI / remote (the framework
  manager calls these).
- **SP5:** dashboard recipe-selector page (scores, pros/cons, librarian/fanout).
- **SP6:** generator-config plumbing depth (recipe selects the answer-synthesis
  model end to end), if not fully covered by `get/set_memory_model`.

Each later sub-project gets its own spec. SP1 deliberately defines the schema in
full (SP4 and SP5 consume it) but implements only the core.

## The Recipe config-bundle schema

A recipe is a typed bundle. Every field carries a type, a default, an allowed
range or enum, and a short help string, so a generic renderer can build a form
from the schema alone. The metadata block is rendered as read-only display.

```
Recipe
  id            str            stable slug, e.g. "maxsim-rerank-9b"
  name          str            human label, e.g. "MaxSim + rerank (12 GB GPU)"
  retrieval:
    strategy            enum   thorough | fast | minimal | custom   (default thorough)
    limit               int    1..50    final result count          (default 5)
    candidate_top_k     int    5..100   pool size before rerank      (default 20)
    fusion              enum   boost | rrf | mem0_additive           (default boost)
    reranker            enum   none | bge-v2-m3                      (default none)
    adjacent_neighbors  int    0..4     positional neighbours per hit (default 0)
    llm_reranker        bool   listwise LLM second pass              (default false)
  ingest:
    extraction          bool   run LLM enrichment on ingest          (default true)
    extraction_model    str    provider:model for enrichment, or "" to inherit
    embed_verbatim      bool   archive + embed raw turns             (default true)
  generator:
    model               str    provider:model for answer synthesis, or "" to inherit
  librarian:
    fanout              enum   off | conservative | balanced | aggressive (default balanced)
    worker_aware        bool   scale fanout by worker capabilities    (default true)
  metadata:                    (read-only, display + ranking input)
    tier            enum   pi-npu | cpu | gpu-4gb | gpu-8gb | gpu-12gb | unconstrained
    scores          map    judge -> float, e.g. {"gemma4:e2b": 0.748,
                           "llama3.1:8b": 0.394, "qwen3:4b-instruct-2507": 0.659}
    pros            list[str]
    cons            list[str]
    est_latency     enum   low | medium | high
    est_footprint   enum   low | medium | high
    source          str    provenance, e.g. "docs/benchmarks.md full-1540 tri-judge"
```

The four config sections map one-to-one onto existing call sites:
`retrieval.*` -> `retrieve(...)` parameters; `ingest.extraction` -> the LLM
enrichment stage (memory_extractor / emem_event_lift / catalog_pipeline), which
is already separate from the archive+embed that `api.ingest()` does today;
`generator.model` -> `config.set_memory_model` semantics for the synthesis
model; `librarian.fanout` -> `effective_fanout` per-agent config.

`metadata.scores` is a map keyed by judge so the dashboard and recommend() can
show the honest tri-judge picture (lenient gemma high, strict near-tie) rather
than a single conflated number. Recipes we have not benchmarked under a given
judge simply omit that key.

### Schema export

`recipe_schema() -> dict` returns the JSON Schema for the bundle above
(`type: object`, typed properties, enums, ranges, descriptions, plus the
read-only metadata sub-schema). The taOS framework manager renders a settings
page from this with no taosmd-specific UI code; the dashboard SP5 consumes the
same export. This is the declared contract seam.

### Contract seam with taOS (agreed, implemented in SP4/SP2)

Recorded here so the SP1 core exposes the right shapes for later wiring. taOS
already renders memory config generically from `MemoryBackend.get_settings_schema`
(GET /api/memory/backend/settings-schema); the recipe surface follows the same
"render from declared schema" pattern.

- Recipes are **dedicated methods on the `MemoryBackend` ABC**, not folded into
  `get_settings_schema` (a recipe is a first-class object, not a flat settings
  key). The methods: `list_recipes()`, `get_recipe(id)`, `apply_recipe(id)`,
  `create_recipe(spec)` (SP3), `recommend(device_info)` (SP2). They delegate to
  the SP1 `recipes.py` core. `get_settings_schema` stays for per-field knobs;
  `apply_recipe` writes through to `update_settings` (the one-source-of-truth
  rule above). taOS proxies these as GET /api/memory/recipes,
  GET|POST .../recipes/{id}, POST .../recipes/recommend, mirroring the existing
  backend-settings route so the Memory app stays framework-generic.
- `recommend(device_info)` returns a **ranked list** (score + rationale per
  recipe), which drives the picker UI. device_info shape that taOS produces and
  recommend consumes:
  `{host: HardwareProfile, cluster: {online_workers, workers[], aggregate:
  {max_gpu_vram_mb, total_gpu_vram_mb, has_npu, total_cores, total_ram_mb}}}`
  where HardwareProfile is `{cpu{arch,cores,soc}, ram_mb, npu{type,tops,cores},
  gpu{type,vram_mb,cuda,vulkan,rocm}, disk, os, profile_id}`. taOS owns building
  this (it will add the cluster aggregate, which does not exist today). recommend
  reasons over host-alone or host+cluster.
- Convergence: `device_info` is a superset of the
  `worker_capabilities={gpu_vram_gb, turboquant}` that `effective_fanout`
  (agents.py:553) already consumes, so recipes and fanout share one device
  vocabulary. The taOS WorkerInfo.hardware -> effective_fanout bridge is not
  wired today; recipes can close that gap (SP2).

## Built-in registry

Defined in a new `taosmd/recipes.py` as code (decided: built-ins in code, custom
in the config DB in SP3). The registry holds the recipes we have actually
measured. Scores are taken verbatim from `docs/benchmarks.md`; we do not invent
numbers for combinations we have not run.

| id | tier | retrieval shape | scores (lenient / llama / qwen-inst) | role |
|---|---|---|---|---|
| `maxsim-rerank-9b` | gpu-12gb | k=50, adj=2, bge-v2-m3, mem0_additive, gen qwen3.5:9b | 0.748 / 0.394 / 0.659 | leader where reranker affordable |
| `rrf-9b` | gpu-12gb | k=20, adj=2, rrf, llm-exp, gen qwen3.5:9b | 0.723 / 0.390 / 0.634 | leader without a reranker |
| `fast-8b` | gpu-4gb | k=20, adj=2, rrf, gen llama3.1:8b | (fast-tier, -0.02 vs 9B, ~2.4x faster) | realtime / multi-agent on one GPU |
| `lite-pi` | pi-npu / cpu | adj=2, no reranker, **extraction off** | (no-LLM-ingest, Midas-class low tier) | SBC where LLM-on-ingest is too slow |

Numbers in parentheses are qualitative because they come from subset or
cross-tier runs, not the full-1540 tri-judge; the metadata records the source so
the display can label them honestly. The lite recipe's headline property is
`ingest.extraction = false`: ingest stays archive+embed only (which is what the
current `api.ingest()` already does), so a Pi 4B does not pay an LLM extraction
cost per turn. Retrieval is unchanged, so recall is unchanged; only the
ingest-time enrichment is dropped.

## Resolution and application

A new `taosmd/recipes.py` exposes:

- `get_recipe(recipe_id) -> Recipe | None`
- `list_recipes() -> list[Recipe]`
- `resolve_recipe(agent, data_dir) -> Recipe` resolution order:
  1. the agent's per-agent recipe override (agent config), if set;
  2. the global default recipe (config.json, new key alongside memory_model);
  3. a tier-safe fallback built-in (`rrf-9b` when a GPU is present and the
     reranker is not known-available, else `lite-pi`). The fallback is
     conservative on purpose: it never assumes a reranker that may not be
     installed.
- `recipe_schema() -> dict` (the export above).

Application happens at the existing call sites, not by rewriting them:

- `api.search()` calls `resolve_recipe(...)` and maps the recipe's `retrieval`
  section onto its `retrieve(...)` call (limit, candidate pool, fusion,
  reranker, adjacent_neighbors, llm_reranker). The reranker field maps to
  constructing the CrossEncoderReranker only when the recipe asks for it and the
  model is available; if asked-for but unavailable, it degrades to no reranker
  and records that in the result so the gap is visible rather than silent.
- `api.ingest()` (and the enrichment stage it feeds) consults
  `ingest.extraction`; when false, the LLM enrichment is skipped and only
  archive+embed runs.
- `generator.model` and `librarian.fanout` thread through the existing
  `config.set_memory_model` / `effective_fanout` machinery.

Resolution is explicit and overridable at the call boundary: callers can still
pass `reranker=` etc. directly, and an explicit argument wins over the recipe so
nothing existing breaks. This keeps the lever-level API intact and layers
recipes on top.

### Global default + per-agent override (decided)

- Global default recipe id lives in `config.json` under a new key (next to
  `memory_model`), read/written with the existing atomic `_read`/`_write`.
- Per-agent override is the **applied recipe id** stored on the agent. taOS
  confirms there is already a stub agent field `memory_config` (device_id +
  tier_id, `None` -> global default) that nothing wires today: this is the same
  "params nothing sets" gap from the taOS side. Recipes supersede it. We retire
  the dead `device_id` / `tier_id` stub and make `applied_recipe_id` the per-agent
  (and system) selector. One agent can run the lite recipe while another runs the
  leader on the same install.
- A fresh install ships with the global default unset; `resolve_recipe` then
  returns the tier-safe fallback, which is a real benchmarked recipe rather than
  today's unconfigured defaults. Setting the global default to `maxsim-rerank-9b`
  on a 12 GB box is then a one-line change a user or taOS can make.

### One source of truth (write-through)

Verified in code: `api.search()` today passes only `limit`, `project`, and the
agent list to `retrieve()`; it never sets `reranker`, `fusion`,
`adjacent_neighbors`, or the candidate pool, so they always fall to the
off-by-default values. There is no settings-read on the search path at all. SP1
introduces one.

The resolved active recipe is the single source of truth for the retrieval and
ingest knobs. "Applying" a recipe (the SP4 `apply_recipe(id)` method, see below)
both records `applied_recipe_id` and writes the recipe's config sections through
to the existing settings store via `update_settings`, so a recipe and any manual
per-field knob edits share one backing store. `resolve_recipe` merges: the
registry recipe identified by `applied_recipe_id`, with any per-field manual
overrides from settings layered on top. The UI can then show "you are on
MaxSim+rerank" and, if a manual knob has since diverged, "(modified)". The
recipe metadata (scores, pros, cons, tier) lives on the recipe object, never in
the flat settings store.

## Error handling

- Unknown recipe id: `get_recipe` returns None; `resolve_recipe` falls through to
  the next resolution step and logs which recipe id was requested and missing.
- Reranker requested but model not present: degrade to no reranker, attach a
  `recipe_degraded` note to the search result. Never silently pretend the
  reranker ran (silent-failure rule).
- Malformed custom recipe (SP3 territory, but the loader is built in SP1):
  validate against `recipe_schema()` on load; reject with a clear error naming
  the offending field rather than partially applying it.
- A recipe naming a generator/extraction model that is not installed: the model
  layer already surfaces a missing-model error; the recipe layer does not swallow
  it.

## Testing

- Schema export: `recipe_schema()` is valid JSON Schema; every built-in recipe
  validates against it.
- Registry integrity: every built-in has a unique id, a tier, and at least one
  score with a recorded source; scores match the values in docs/benchmarks.md
  (a test reads both so the two cannot drift).
- Resolution order: per-agent override beats global default beats tier-fallback;
  unknown id falls through; two agents can hold different recipes on one install.
- Application: a recipe with `reranker=bge-v2-m3` causes `retrieve()` to receive
  a reranker; `adjacent_neighbors`, `limit`, `candidate_top_k`, and `fusion`
  thread through; an explicit call-site argument overrides the recipe.
- Lite path: `lite-pi` makes ingest skip LLM enrichment (enrichment stage is not
  called) while archive+embed still runs, and search still returns hits.
- Degradation: reranker requested but unavailable degrades to no reranker and
  sets `recipe_degraded`; nothing raises.
- No regression: existing search/ingest tests pass unchanged when no recipe is
  configured (fallback behaves at least as well as today's defaults).

## Out of scope / future

- recommend(device_info) ranking (SP2).
- Custom recipe CRUD + DB persistence (SP3).
- taOS contract API across surfaces (SP4).
- Dashboard recipe-selector page (SP5).
- Deeper generator-config plumbing if `set_memory_model` does not fully cover
  per-recipe generator selection (SP6).
- Any new retrieval lever or new benchmark run. SP1 wires only measured recipes.
