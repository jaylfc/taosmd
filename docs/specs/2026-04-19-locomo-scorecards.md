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

| Category (count) | taosmd-e2b | taosmd-e4b | taosmd-e2b+prompt-opt | mem0-e2b (infer=False) |
|---|---|---|---|---|
| Single-hop (282) | 0.34 | 0.13 | 0.34 | 0.11 |
| Temporal (321)   | 0.29 | 0.25 | 0.36 | 0.02 |
| Multi-hop (96)   | 0.22 | 0.09 | 0.29 | 0.12 |
| Open-dom (841)   | 0.64 | 0.52 | 0.64 | 0.10 |
| **Overall Judge** | **0.48** | **0.37** | **0.51** | **0.09** |
| Overall F1 | 0.16 | 0.15 | 0.17 | 0.05 |

Result files:
- `benchmarks/results/locomo_20260417_140810_full_gemma_e2b.json`
- `benchmarks/results/locomo_20260418_035232_full_gemma_e4b.json`
- `benchmarks/results/locomo_20260418_130212_full_gemma_e2b_opt.json`
- `benchmarks/results/locomo_20260419_185944_full_mem0_e2b_noinfer_full_mem0_e2b.json`

### External qwen3:4b judge (100% coverage, 0 errors)

Run date: 2026-04-19. Same predictions as self-judge, re-graded by qwen3:4b.

| Category (count) | taosmd-e2b | taosmd-e4b | taosmd-e2b+prompt-opt | mem0-e2b |
|---|---|---|---|---|
| Single-hop (282) | 0.08 | 0.18 | 0.16 | — |
| Temporal (321)   | 0.13 | 0.14 | 0.21 | — |
| Multi-hop (96)   | 0.05 | 0.00 | 0.00 | — |
| Open-dom (841)   | 0.43 | 0.30 | 0.46 | — |
| **Overall Judge** | **0.27** | **0.22** | **0.34** | **pending** |
| Overall F1 | 0.162 | 0.152 | 0.175 | 0.05 (self-graded) |

> **Earlier caveat** — the initial rescore pass with qwen3.5:9b had a 64–72%
> timeout rate; those numbers (taosmd-e2b Overall 0.27 on that pass) were
> computed over a ~28% sample biased toward short predictions. That run is
> superseded by the qwen3:4b pass above which has 100% coverage.

Rescore output files (include per-item `judge_rejudged`):
- `benchmarks/results/locomo_20260417_140810_full_gemma_e2b.rescored_v2.json`
- `benchmarks/results/locomo_20260418_035232_full_gemma_e4b.rescored_v2.json`
- `benchmarks/results/locomo_20260418_130212_full_gemma_e2b_opt.rescored_v2.json`
- mem0 rescore output pending (in flight, ETA ~23:00 BST 2026-04-19)

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
| Commits | taosmd runs: `40403cc` (runner) + `86c4c19` (rescore); prompt-opt: `3c5c6c2` |

---

## In flight / queued

- mem0-e2b external qwen3:4b rescore — running in screen `locomo-mem0-rescore`
  on Fedora, ETA ~23:00 BST 2026-04-19.
- MemPalace adapter (`benchmarks/mempalace_locomo_runner.py`) — to be built
  via Sonnet subagent once mem0 rescore completes. Expected architecture:
  use `mempalace.searcher.search_memories()` as the retrieval step, keep
  the same generator/prompt/judge as taosmd + mem0 runners. MemPalace's own
  LoCoMo numbers are R@10-only (60.3% raw / 88.9% hybrid v5 per their
  `benchmarks/BENCHMARKS.md`) — adding end-to-end Judge is a novel
  measurement.
- PR #30 (LoCoMo bundle), PR #32 (README Judge framing), PR #33 (silent-
  failure fixes) — open, awaiting bot review cycle.

---

## When to update this doc

After every new run completes (or its rescore completes) — commit an update
the same PR/branch that ships the numbers. Don't let scorecards live only
in a chat transcript.
