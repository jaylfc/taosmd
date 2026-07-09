# taOSmd memory app: controls contract and recommended config (integration handoff)

This is the configuration surface for embedding taOSmd as the memory app in taOS: the API and schema the app builds its settings UI against, which controls exist and what scope they have, the per-tier model and retrieval defaults, and what not to default on. Full measurements and provenance are in [benchmarks.md](benchmarks.md) and the [research report](research-report.md); this page is the short, integration-focused view.

## The controls contract (build the settings UI against this)

Every control is defined once in `taosmd/controls.py`. The standalone dashboard, this handoff, and the taOS app all read that registry through one API, so they cannot drift. The taOS app should render its settings UI generically from the schema rather than hard-coding controls.

- `GET /controls` returns `{ "settings": {...}, "schema": { "controls": [...], "presets": [...] } }`. `settings` is the resolved current value for every control id. Each `schema.controls` entry carries `id`, `label`, `category`, `scope`, `type` (`bool` / `choice` / `int`), `config_key`, `default`, `choices`, `int_range`, `cost`, `pros`, `cons`, `description`, and `benchmarks_anchor`. Render the cost, pros, and cons next to each control; they are written for end users.
- `POST /controls` accepts one of: `{ "preset": "minimal|quality|integrity" }`, `{ "values": { "<id>": <value>, ... } }`, or a bare `{ "<id>": <value> }`. It validates each value, persists the live ones, and returns `{ "settings": {...}, "errors": { "<id>": "<message>" } }`. A bad value returns HTTP 400 with the offending ids in `errors`; the good settings are left untouched. An unknown preset or empty body returns HTTP 400 with `{ "error": ... }`.
- Scope is the contract for whether a control is a live toggle. `runtime` controls are live: a `POST` takes effect on the next search. `store` controls (the embedder, binary quantization, late-interaction) change how memories are indexed, so they need a re-index of the existing store and `POST /controls` rejects them with a clear message. `consumer` controls (`self_verify`) are applied in the app's own answer-generation, not in taOSmd core. Render `store` and `consumer` controls as informational state, never as live switches that would silently do nothing.

## Controls by scope

| Control | Scope | Config key | Default | What it does |
|---|---|---|---|---|
| `prefer_verified` | runtime (live) | `controls.prefer_verified` | `prefer_verified` (on) | Demotes claims verified as unsupported out of default recall. `off` / `prefer_verified` / `strict`. |
| `reranker` | runtime (live) | `controls.reranker` | `off` | Cross-encoder re-scoring before serving. `off` / `bge-v2-m3`. Overrides the recipe per query. |
| `fusion` | runtime (live) | `controls.fusion` | `rrf` | How dense and lexical hits combine. `rrf` / `mem0_additive` / `boost`. Overrides the recipe per query. |
| `adjacent_turns` | runtime (live) | `controls.adjacent_turns` | `2` (0 to 4) | Positional neighbours included around each hit. Overrides the recipe per query. |
| `embedder` | store (re-index) | `vector_memory.embed_model` | `arctic-embed-s` | Dense ONNX embedder, set at setup. `arctic-embed-s` / `minilm-onnx`. |
| `binary_quant` | store (re-index) | `vector_memory.binary_quant` | `off` | 1 bit per dimension, 32x smaller vectors, recall-neutral. |
| `late_interaction` | store (re-index) | `vector_memory.late_interaction` | `off` | Token-level MaxSim; per-token vectors are written at ingest. |
| `self_verify` | consumer (answer-gen) | `answer.self_verify` | `off` | CoVe-style answer self-verification, run in the app's answer-generation. |
| `generator_profile` | consumer (answer-gen) | `generator_profile` | `balanced` | Selects the answer generator by workload per hardware tier. `balanced` / `factual-recall` (gemma4:12b at 12 GB for single-fact QA; loses conversational/long-context work). |

The generator profile also has its own dedicated surface beyond `/controls`:

- `GET /generator-profile` (optional `?agent=<name>`) returns `{ "profiles": [...], "active": "<id>", "scope": "global" | "<agent>" }`; each profile carries `id`, `label`, `workload`, and the per-tier `models` map.
- `POST /generator-profile` with `{ "profile_id": "<id>", "agent": "<name>"? }` sets the global default (no `agent`) or a per-agent override; an unknown profile or agent returns HTTP 400.
- CLI: `taosmd generator-profile list` / `show <id>` / `set <id> [--agent NAME]`.
- Resolution precedence at answer time: explicit model pin > per-agent profile > global profile (default `balanced`) > recipe generator > retrieval-only (see `taosmd/generator_profiles.py::resolve_generator`).

The cost, pros, and cons for each are in the schema and mirrored in the README Configuration and controls section. Headline evidence: `prefer_verified` eliminates served-hallucination (0.040 to 0.000) at no measured accuracy cost, tri-judge confirmed (E-018); `reranker` plus `self_verify` is the F-013 end-to-end LongMemEval-S Judge configuration (the originally published 74.6% was inflated by the pre-#176 judge-parser bug, N-017; the corrected full-500 baseline is 42.8% qwen3.5:9b / 51.2% llama3.1:8b, and the gemma4:12b factual-recall combo reaches 53.8 / 61.4, parity with MemOS-lossless); `arctic-embed-s` is +0.057 judged retrieval over MiniLM (F-010).

## Presets (one-tap bundles of the live controls)

| Preset | `reranker` | `prefer_verified` | `fusion` | `adjacent_turns` | For |
|---|---|---|---|---|---|
| Minimal | `off` | `off` | `rrf` | 1 | Fastest and lightest; weak hardware or speed-first. |
| Quality | `bge-v2-m3` | `off` | `rrf` | 2 | Best accuracy where hardware affords (pair with `self_verify` in answer-gen). |
| Integrity | `bge-v2-m3` | `prefer_verified` | `rrf` | 2 | Auditable, zero-served-hallucination recall. |

The recall gate is on globally by default, so Minimal is the preset that turns it off.

## Install-time profiles vs runtime controls

These are two registries at two layers, and they share config keys so they agree:

- `taosmd/profiles.py` is the install-time registry (the `taosmd setup-prompt` flow and the smart installer): switches with consent rules and three profiles (Minimal / Quality / Integrity) that the installer writes once at setup.
- `taosmd/controls.py` is the runtime registry the memory app reads and writes live through `/controls`.

The install picks a profile; the app tunes the live controls afterward. The recall gate writes the same `controls.prefer_verified` at both layers, so an install-time choice and a later live change cannot contradict each other.

## Per-tier generator and retrieval defaults

| Tier | Generator (default) | Retrieval recipe | Measured |
|---|---|---|---|
| 12 GB GPU (RTX 3060 class) | qwen3.5:9b Q4_K_M (best quality) or llama3.1:8b (2.4x faster) | k=20, adj=2, llm-query-expansion, fusion rrf; add bge-v2-m3 MaxSim rerank where affordable | LoCoMo 0.557 strict / 0.748 lenient (tri-judge) |
| 8 GB GPU | qwen3.5:9b IQ4_XS (4.81 GB), concurrency 1 | same stack as the 12 GB tier | ~0.55 (from the quant-cliff data) |
| 16 GB Orange Pi 5 Plus (RK3588 NPU) | retrieval-only by default (the `balanced` profile maps pi-npu to no generator); Qwen3-4B via rkllama on the NPU is the measured benchmark config, opt-in | adj=2, k=20, llm-exp, RRF; reranker Qwen3-Reranker-0.6B on NPU | LoCoMo 0.490; this is the 97.0% Recall@5 reference stack |
| 4 GB GPU (GTX 1050 Ti) | llama3.1:8b (shipped default at gpu-4gb); qwen3:4b Q4 (~2.5 GB) is the measured on-GPU alternative | adj=2, k=10, RRF; skip llm-query-expansion | LoCoMo 0.530 (qwen3:4b) |
| Raspberry Pi 4 (CPU only) | retrieval-only by default (the `balanced` profile maps cpu to no generator); qwen3:1.7b or smaller is an extrapolated option | adj=1, k=10; skip every flag that adds an LLM call | better used as a storage/retrieval node, offload generation to a peer |

Embedder, reranker, and judge run as CPU ONNX on every tier; GPU or NPU VRAM is for the generator only. Do not run the judge on the same device as the generator.

## Changes from the prior handoff

- The cross-encoder reranker is `bge-v2-m3` only. The earlier note listing `ms-marco-MiniLM` was wrong; only `bge-v2-m3` is wired in the retrieval path.
- `prefer_verified` is now on by default, not opt-in. The tri-judge confirm at n=250 (E-018) reproduced zero served-hallucination under all three judges at no measured accuracy or recall cost, and Jay signed off on the flip. It is a safe no-op until the verify-pass is populated.
- Late-interaction is a store-level choice (a re-index), not a per-query toggle. The settings UI should present it as a setup option, not a live switch.

## What not to default on

multihop-decompose (regresses), session_date context format (no-op), HyDE (regresses on small generators), and bare FROM-gguf Modelfiles (they need the full TEMPLATE/RENDERER/PARSER/PARAMETER metadata cloned from `ollama show <model> --modelfile` or output runs away and looks far slower than the kernel is).

For the live measurement log and any new tier validations, see [benchmarks.md](benchmarks.md) and the [research report](research-report.md).
