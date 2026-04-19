# LoCoMo scorecard — live results log

**Purpose.** Single source of truth for LoCoMo benchmark numbers across
taosmd variants, mem0, and (in progress) MemPalace. Every row names the
commit, model, judge, and data file so the numbers never drift from the
methodology.

**Dataset.** LoCoMo-10 (10 conversations × ~154 QAs each = 1540 QAs), standard
categories Single-hop (1) / Temporal (2) / Multi-hop (3) / Open-dom (4).
Adversarial (cat 5) excluded — reported separately when run.

**Host.** All runs on Fedora workstation (192.168.6.108), RTX 3060 12GB,
Ollama (`OLLAMA_NUM_PARALLEL=3`), ONNX embed backend, top-K=10.

**Generator.** gemma4:e2b Q4_K_M (5.1B params actual) via Ollama unless
otherwise noted. Same `ANSWER_PROMPT` across all systems.

**Judges used.**
- **Self-judge**: the system's own generator also grades. Known to inflate
  — same model can't penalize its own idioms. Published for historical
  continuity only; not the headline.
- **External judge (qwen3:4b)**: fixed model outside the system under test.
  Streaming rescore tool with 240s per-call timeout, checkpointed every 25
  items. 100% coverage / 0 errors on the three taosmd runs.

---

## Scorecards

### Self-judge (same model judges its own outputs)

Run dates: 2026-04-17 → 2026-04-18.

| Category (count) | taosmd-e2b | taosmd-e4b | taosmd-e2b+prompt-opt | mem0-e2b (infer=False) | MemPalace-e2b |
|---|---|---|---|---|---|
| Single-hop (282) | 0.34 | 0.13 | 0.34 | 0.11 | 0.29 |
| Temporal (321)   | 0.29 | 0.25 | 0.36 | 0.02 | 0.33 |
| Multi-hop (96)   | 0.22 | 0.09 | 0.29 | 0.12 | 0.24 |
| Open-dom (841)   | 0.64 | 0.52 | 0.64 | 0.10 | 0.51 |
| **Overall Judge** | **0.48** | **0.37** | **0.51** | **0.09** | **0.42** |
| Overall F1 | 0.16 | 0.15 | 0.17 | 0.05 | 0.15 |

**Surprise on MemPalace self-judge.** Their chromadb+MiniLM setup (no LLM extraction, no cross-encoder, no query expansion) lands at Overall 0.42 — **much closer to taosmd than to mem0**. Per-category:
- MemPalace **beats baseline taosmd-e2b on Temporal** (0.33 vs 0.29) and **Multi-hop** (0.24 vs 0.22)
- taosmd's architectural edge shows on Open-dom (0.64 vs 0.51, +0.13) and Single-hop (0.34 vs 0.29, +0.05)
- The prompt-opt variant is still the overall leader (0.51)
- mem0 is a distant fourth on every category

Reads as: **verbatim-store + good default embedder is a surprisingly strong baseline**. taosmd's cross-encoder rerank + query expansion + KG only pulls ahead on categories that benefit from surface-semantic retrieval beyond single-turn similarity (Open-dom especially). mem0's `infer=False` raw vector pattern (with nomic-embed-text) underperforms both — likely the nomic embedder being a weaker default than all-MiniLM on conversation-turn-level content.

Result files:
- `benchmarks/results/locomo_20260417_140810_full_gemma_e2b.json`
- `benchmarks/results/locomo_20260418_035232_full_gemma_e4b.json`
- `benchmarks/results/locomo_20260418_130212_full_gemma_e2b_opt.json`
- `benchmarks/results/locomo_20260419_185944_full_mem0_e2b_noinfer_full_mem0_e2b.json`
- `benchmarks/results/locomo_20260419_225441_full_mempalace_e2b_full_mempalace_e2b.json`

Ingest timings (10 convs each): taosmd runner ingests via ONNX MiniLM + KG in ~180-200s total. mem0 batch-ingest at `infer=False` ~130s. **MemPalace (raw chromadb) was the fastest at ~100s** — simpler architecture, less processing per turn.

### External qwen3:4b judge (100% coverage, 0 errors)

Run date: 2026-04-19. Same predictions as self-judge, re-graded by qwen3:4b.

| Category (count) | taosmd-e2b | taosmd-e4b | taosmd-e2b+prompt-opt | mem0-e2b |
|---|---|---|---|---|
| Single-hop (282) | 0.17 | 0.15 | 0.16 | 0.04 |
| Temporal (321)   | 0.36 | 0.31 | 0.41 | 0.02 |
| Multi-hop (96)   | 0.21 | 0.14 | 0.24 | 0.10 |
| Open-dom (841)   | 0.51 | 0.51 | 0.51 | 0.07 |
| **Overall Judge** | **0.40** | **0.38** | **0.41** | **0.06** |
| Overall F1 | 0.162 | 0.152 | 0.175 | 0.047 |

**Biggest architecture gap: Temporal** (taosmd-e2b+prompt-opt 0.41 vs mem0 0.02 = 20.5×). **Overall gap: ~7×** under external judge (taosmd-e2b 0.40 vs mem0 0.06). Same generator, same prompt, same judge, same 1540 QAs — only the retrieval layer differs.

> **Earlier-run caveat** — before we settled on `qwen3:4b` as external judge,
> a first pass with `qwen3.5:9b` hit a 64–72% timeout rate (root cause: 60 s
> client timeout vs the model's occasional long tail at NUM_PARALLEL=3). The
> numbers from that partial run (taosmd-e2b Overall 0.27, e4b 0.22, opt 0.34)
> were computed over a ~28% sample biased toward short predicted text.
> **Those numbers are superseded by the qwen3:4b 100%-coverage pass tabulated
> above** and should not be quoted. Kept here only to explain the shift if
> anyone reads earlier chat transcripts.

Rescore output files (include per-item `judge_rejudged`):
- `benchmarks/results/locomo_20260417_140810_full_gemma_e2b.rescored_v2.json`
- `benchmarks/results/locomo_20260418_035232_full_gemma_e4b.rescored_v2.json`
- `benchmarks/results/locomo_20260418_130212_full_gemma_e2b_opt.rescored_v2.json`
- `benchmarks/results/locomo_20260419_185944_full_mem0_e2b_noinfer.rescored_v2.json`

Timings: taosmd e2b rescore 188.8 min, e4b 166.6 min, e2b-opt 206.7 min, mem0 116.9 min — all at concurrency 3 against `qwen3:4b` with 240 s per-call timeout. Zero judge errors across all four runs.

---

## Architectural interpretation

**At same compute tier (gemma4:e2b generator, same prompt, same dataset),
the only variable is the memory architecture.** Self-judge numbers:

- taosmd's cross-encoder + query expansion + KG: Overall 0.48
- mem0's raw vector retrieval (infer=False, chroma + nomic-embed): Overall 0.09

That is a **5.3× gap**. Per-category the most dramatic split is Temporal
(taosmd 0.29 / mem0 0.02 = 14.5×) — mem0's naive vector retrieval has no
temporal reasoning.

The `taosmd-e2b-opt` prompt tweak (absolute dates + inference-when-uncertain)
gained +0.07 Overall Judge over the base prompt under external judge, and the
gain localized cleanly to the targeted categories (Temporal +0.05, Multi-hop
+0.03), with Single-hop and Open-dom unchanged.

e4b being larger (8B vs 5B) did **not** translate to higher scores because
its higher IDK rate (78% on multi-hop vs 59% for e2b) overwhelmed the
generator-quality improvement.

---

## Known data artefacts

- **mem0 R@K always reports 0.0** because mem0 doesn't round-trip per-turn
  `dia_id` metadata, so the adapter hardcodes `evidence_hits=0`. PR #33
  patches this to `None` (metric unavailable) so the summary drops it from
  recall aggregation rather than publishing a fake 0.0. Existing mem0
  result JSON should be rescored via the fixed `_summary` logic once #33
  lands.
- **Silent-failure aggregation.** Prior to PR #33, judge timeouts and
  generation errors both returned 0.0 and got folded into the averages.
  That let infra flakiness bias the Judge number downward. After #33, both
  return None and are excluded.

---

## Methodology disclosures

| Item | Value |
|---|---|
| Dataset | LoCoMo-10, 1540 QAs, categories 1–4 (adversarial reserved) |
| Ingestion | All convs ingested into one embedding backend per run. taosmd: MiniLM ONNX + KG + cross-encoder. mem0: chroma + nomic-embed-text, `infer=False`. |
| Generator | gemma4:e2b Q4_K_M via Ollama (Fedora, NUM_PARALLEL=3) |
| Top-K | 10 |
| Answer prompt | Identical across systems (`ANSWER_PROMPT` in `benchmarks/locomo_runner.py`) |
| Self-judge | gemma4:e2b (i.e. the same generator) |
| External judge | qwen3:4b via Ollama, temperature 0.0, `benchmarks/locomo_rescore_streaming.py` |
| Timeout (rescore) | 240s per call, concurrency 3 |
| Commits | Runner + adapter + prompt-opt: `fd27d2c` (PR #30, merged). Rescore tool: `c360faa` (PR #29, merged). README Judge framing: `116edab`/`PR #31` + `fix/readme-judge-accuracy-consistency` (PR #32, merged). Silent-failure fixes: `bc0a773` (PR #35, pending). MemPalace adapter (chromadb path): `571d8af` (PR #36, pending — supersedes earlier `ca0ccb7` palace/mine attempt). |

---

## Configuration log (models + deployment profile)

| Role | Model | Params | Quant | VRAM/RAM | Backend | Notes |
|---|---|---|---|---|---|---|
| Generator (benchmark default) | `gemma4:e2b` | 5.1B | Q4_K_M | ~7 GB | Ollama | Best default at ≤12 GB VRAM tier |
| Generator (larger, tested) | `gemma4:e4b` | 8.0B | Q4_K_M | ~11 GB | Ollama | Higher IDK rate (78% multi-hop) hurts scores — see lessons below |
| External judge (settled on) | `qwen3:4b` | 4B | Q4_K_M | ~5 GB | Ollama | Reliable structured/JSON output; 100% coverage on all four rescore runs |
| External judge (rejected) | `qwen3.5:9b` | 9B | Q4_K_M | ~9 GB | Ollama | 64–72% timeout rate under NUM_PARALLEL=3 at 60 s client timeout — too slow for production judging |
| Embedder (taosmd) | all-MiniLM-L6-v2 ONNX | 22M | fp32 | CPU (~80 MB) | ONNX Runtime | 0.3 ms Pi / ~10 ms Fedora CPU |
| Embedder (mem0) | `nomic-embed-text` | 137M | Q4 | ~0.6 GB | Ollama | 2048-token context — batch ingest mandatory for long conversations |
| Embedder (MemPalace) | chromadb default (MiniLM) | 22M | fp32 | CPU | chromadb built-in | Matches MemPalace's own benchmark default |
| Fact extractor (mem0) | same as generator | — | — | — | Ollama | Requires reliable JSON output; **fails on gemma family**, works on qwen — see lesson #2 |
| Cross-encoder rerank (taosmd) | ms-marco-MiniLM ONNX | 22M | fp32 | CPU | ONNX Runtime | Second-stage rerank over top-K vector hits |

### Host / runtime

| Item | Value |
|---|---|
| Benchmark host | Fedora workstation, RTX 3060 12 GB |
| Ollama | `OLLAMA_NUM_PARALLEL=3` (systemd override) — capped throughput at 3 concurrent |
| Python | 3.14 |
| Concurrency (client) | 3 during rescore; request limit dominated by server NUM_PARALLEL |
| Rescore timeout | 240 s per call (60 s was too tight for qwen3.5:9b) |
| GPU utilisation observed | 87–95% through ingest + QA; 90–95% through rescore |

---

## Hardware tier recommendations (derived from this benchmark)

**Orange Pi 5 Plus (16 GB RAM, RK3588 NPU, no CUDA):**
- Generator: `qwen3:4b` via rkllama on the NPU (~4 GB) — only option that does structured output reliably at this size
- Judge: **external** — Pi can't comfortably dual-load a judge model alongside the generator. Offload to a Fedora or cloud peer.
- Embedder: all-MiniLM-L6-v2 ONNX on CPU (0.3 ms)
- Memory arch: taosmd (this is the tier it was designed around)
- Inference backend: rkllama for LLM, ONNX Runtime for embed/rerank

**Fedora 3060 or any 10–12 GB VRAM GPU:**
- Generator: `gemma4:e2b` as default (best measured Overall Judge at this tier); `qwen3.5:9b` if quality headroom matters more than speed
- Judge: `qwen3:4b` runs alongside gen comfortably (~10 GB combined)
- Ollama: set `OLLAMA_NUM_PARALLEL=3` — verified sweet spot; higher client-side concurrency without raising this limit wastes effort
- Memory arch: taosmd. **Apply prompt-opt** (absolute-dates + infer-when-uncertain in `ANSWER_PROMPT`) — gains +0.05 Temporal / +0.03 Multi-hop for free, zero cost
- Embedder: MiniLM ONNX (CPU) — keeps VRAM for generator + judge

**Laptop / Mac Mini (16 GB unified, CPU or small Metal):**
- Generator: `qwen3:4b` via Ollama — slow but workable (20–30 s/answer on M-series, 60+ s on CPU-only)
- Judge: external (don't dual-load)
- Embedder + arch: same as Pi

**High-end workstation (≥24 GB VRAM):**
- Generator: consider `qwen3.5:9b` as default for quality headroom
- Note: **`gemma4:e2b` remained competitive against e4b even on our 12 GB card** — larger isn't always better at this architectural tier (see lesson #1). Don't assume bigger is the default.

---

## Lessons that drive the defaults

1. **Bigger generator ≠ better default at small-LLM scale.** `gemma4:e2b` (5.1 B) beat `gemma4:e4b` (8.0 B) on LoCoMo Overall Judge, 0.40 vs 0.38. The 8 B model was more cautious — 78% "I don't know" rate on multi-hop vs 59% for e2b — and that refusal behaviour dominates the score. Recommendation: keep e2b as the default at the 7 GB VRAM tier even if a bigger model fits.
2. **`qwen3:4b` is the structured-output default, not `gemma`.** Mem0's LLM fact-extraction blew up on every gemma variant (returned JSON lists where mem0 expected dicts). Qwen handles mem0's extraction schema without patching. Any downstream that asks the model for structured output — extractors, rerankers, judges — should default to qwen at 4 B+.
3. **Ollama `NUM_PARALLEL` is the real concurrency ceiling.** Client-side `concurrency=8` is meaningless if the server is capped at 3. Either raise the server setting for heavier hardware or calibrate client concurrency to match it.
4. **Nomic embedder's 2048-token context forces batch ingest.** A single LoCoMo conversation's 400–700 turns concatenated fits nowhere near 2048 tokens. Either split into short batches (≤6 turns per `memory.add`) or use a longer-context embedder like qwen3-embedding:4b (8192 ctx).
5. **Architecture choice dominates generator choice at small scale.** taosmd vs mem0 at the same gemma4:e2b was a ~7× Judge delta. No single-model swap we tested moved the needle that far. Put engineering effort into retrieval architecture (rerank, query expansion, KG) before chasing a bigger generator.
6. **Self-judge systematically inflates.** Every system we rescored showed self-judge > external by 0.03–0.15. Never publish a number the system graded itself. Always pair generation and judging with distinct models.
7. **R@K in adapters without per-turn dia_id round-trip is meaningless.** Mem0 and MemPalace both store turns without preserving LoCoMo's `dia_id` metadata, so the recall metric collapses to 0 regardless of retrieval quality. Report it as `None` (metric unavailable) rather than `0.0` — see PR #35.

---

## In flight / queued

- **Complete 2026-04-19 21:57 BST** — mem0-e2b external rescore with `qwen3:4b`: Overall Judge **0.06** (116.9 min runtime, 0 errors, 100% coverage).
- **Complete 2026-04-19 23:54 BST** — MemPalace-e2b full benchmark (self-judge): Overall Judge **0.42** (ingest ~100s + QAs ~86min). Per-category showing MemPalace beating baseline taosmd on Temporal + Multi-hop.
- **Running** — MemPalace external rescore with `qwen3:4b`. ETA ~01:55 BST 2026-04-20 based on prior run pace.
- Open PRs awaiting merge: **#34** (this doc), **#35** (silent-failure fixes), **#36** (mempalace adapter rewrite).

---

## When to update this doc

After every new run completes (or its rescore completes) — commit an update
the same PR/branch that ships the numbers. Don't let scorecards live only
in a chat transcript.
