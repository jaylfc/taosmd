# taOSmd Research Report

**Edition 1, 2026-06-11**

---

## 0. Index

### Table of Contents

- [1. Abstract](#1-abstract)
- [2. Methodology](#2-methodology)
  - [2.1 External multi-judge protocol](#21-external-multi-judge-protocol)
  - [2.2 Datasets](#22-datasets)
  - [2.3 Metrics](#23-metrics)
  - [2.4 Hardware tiers](#24-hardware-tiers)
  - [2.5 Kill criteria and pre-registration](#25-kill-criteria-and-pre-registration)
- [3. Results](#3-results)
  - [3.1 LoCoMo leader table (full-1540, tri-judge)](#31-locomo-leader-table-full-1540-tri-judge)
  - [3.2 Late-interaction retrieval (reranker-free, tri-judge)](#32-late-interaction-retrieval-reranker-free-tri-judge)
  - [3.3 Community-standard judge column (qwen3:14b)](#33-community-standard-judge-column-qwen314b)
  - [3.4 CPU-tier retrieval-only latency probe](#34-cpu-tier-retrieval-only-latency-probe)
  - [3.5 LongMemEval-S Recall@5](#35-longmemeval-s-recall5)
  - [3.6 Shipped recipe: lateint-9b](#36-shipped-recipe-lateint-9b)
  - [3.7 Low-tier dense embedder: arctic-embed vs MiniLM](#37-low-tier-dense-embedder-arctic-embed-vs-minilm)
  - [3.8 BEAM long-context (mem0 nugget judge)](#38-beam-long-context-mem0-nugget-rubric-judge)
- [4. Negative Results](#4-negative-results)
- [5. Reproducibility](#5-reproducibility)
- [6. Ongoing Work (Pre-registered)](#6-ongoing-work-pre-registered)
- [7. Revision Log](#7-revision-log)

### Finding Index

Every row is stable. IDs are never reused. E-ids carry over when an experiment moves to Results (gaining an F row) or Negative Results (gaining an N row).

| ID | One-line summary | Section | Status | Provenance |
|---|---|---|---|---|
| F-001 | LongMemEval-S: 97.0% Recall@5 on full 500-question test set (was mislabelled "end-to-end Judge", corrected 2026-06-14) | [3.5](#35-longmemeval-s-recall5) | confirmed (Recall@5) | benchmarks/results/enhanced_20260413_133215.json (query_expand, top_k=5) |
| F-002 | LoCoMo full-1540 leader (MaxSim + rerank): 0.748 lenient / 0.394 strict-llama / 0.659 strict-instruct | [3.1](#31-locomo-leader-table-full-1540-tri-judge) | confirmed | docs/benchmarks.md "Full-1540 leader (tri-judge, Jun 2026)" |
| F-003 | Late-interaction (answerai backbone, no reranker): 0.716 / 0.388 / 0.656 full-1540 tri-judge | [3.2](#32-late-interaction-retrieval-reranker-free-tri-judge) | confirmed | docs/benchmarks.md "Late-interaction retrieval (token-level MaxSim at retrieval time, tri-judge)" |
| F-004 | CPU-tier R@K: dense 0.641 at 72ms p50 to answerai 0.854 at 110ms p50 on 16-core CPU VPS | [3.4](#34-cpu-tier-retrieval-only-latency-probe) | confirmed | docs/benchmarks.md "CPU-tier viability (retrieval-only, judge-free)" |
| F-005 | qwen3:14b community judge on late-int files: dense 0.487 / MiniLM MaxSim 0.532 / answerai 0.542 | [3.3](#33-community-standard-judge-column-qwen314b) | confirmed | docs/benchmarks.md "Community-standard judge column (qwen3:14b)" |
| F-006 | Librarian vocabulary-gap lever: +15.4% composite on long-horizon sessions (gemma4:e2b, 2026-04-15) | [3.5](#35-longmemeval-s-end-to-end) | confirmed | docs/benchmarks.md section "Librarian layer vocabulary-gap benchmark" |
| F-007 | lateint-9b shipped recipe: answerai + mem0_additive + k=10, retrieval_k=20, adj=2, no reranker | [3.6](#36-shipped-recipe-lateint-9b) | confirmed | CHANGELOG.md Unreleased, "Late-interaction retrieval lever and lateint-9b recipe" |
| F-008 | Same-model judging inflates scores 15-22 pp vs external judge (quantified on qwen3.5:9b self-judge: +0.10 on this stack) | [2.1](#21-external-multi-judge-protocol) | confirmed | docs/benchmarks.md section "Judge sensitivity what we are really measuring" |
| F-009 | Extraction-hallucination rate 18.8 percent (PARTIAL+UNSUPPORTED) over 526 claims, cross-family verified | [3.5](#35-extraction-hallucination-rate-f-009) | confirmed | bench host 20260611 e2 verifier summary |
| F-010 | Arctic-embed-s beats MiniLM as low-tier dense embedder: +0.157 R@K, and judged +0.040 subset-200 / +0.0565 full-1540 (0.7305 vs 0.6740); same 384 dim and latency | [3.7](#37-low-tier-dense-embedder-arctic-embed-vs-minilm) | confirmed (full-1540) | bench host e007full_* 20260613_131837 |
| F-011 | Claims gate (prefer_verified) eliminates served-hallucination (0.040 to 0.000) at no accuracy cost (judge +0.020 within noise, R@K -0.005) on 200 LoCoMo QAs; strict trades too much (judge/R@K -0.065), stays opt-in. Default flip pending Jay sign-off + tri-judge confirm | [6](#6-ongoing-work-pre-registered) | resolved (default-flip pending confirm) | bench host benchmarks/results/e009.json |
| F-012 | LongMemEval-S end-to-end Judge BASELINE 47.2% on oracle (qwen3.5:9b gen + Qwen3-4B-Instruct strict judge + MiniLM); retrieval solved, REASONING is the bottleneck (temporal 17.3%, multi-session 28.6%); improvement phase in progress | [6](#6-ongoing-work-pre-registered) | baseline recorded | benchmarks/results/e012_full500.log |
| F-013 | LongMemEval-S end-to-end Judge SCORE-UP to 74.6% (373/500 oracle, same strict judge), +27.4pp over F-012, via depth tuning (+9.6) + reranking (+6) + CoVe-style answer self-verification (dominant, ~+18); self-verify fixes the bottlenecks (temporal 30->56%, multi-session 40->67%); query decomposition is a NULL-to-NEGATIVE arm (not shipped). Confirmed judge-robust (llama3.1 firm-up reproduced both numbers exactly) | [6](#6-ongoing-work-pre-registered) | confirmed | benchmarks/results/e012_ablation_results.md |
| F-014 | BEAM long-context (mem0 nugget judge, local stack): 100K 47.0% (n=100), 1M 37.5% (75/200); local-9B honestly behind mem0 frontier 64-70% at 1M, the provenanced local-tier number | [3.8](#38-beam-long-context-mem0-nugget-rubric-judge) | measured | benchmarks/results/beam_campaign_resume_20260616_221030.log + beam_1M result JSON |
| N-001 | CLAG cluster pre-filter: worst variant -0.285 gemma, best variant -0.085 gemma; not shipped | [4](#4-negative-results) | confirmed negative | docs/benchmarks.md section "CLAG cluster pre-filter negative at our tier, not shipped" |
| N-002 | Chain-of-Memory: -0.16 single-hop for +0.02 overall at 1.8x latency (frontier-tuned, lossy at our tier) | [4](#4-negative-results) | confirmed negative | project memory (reference_memory_landscape_may2026.md), Chain-of-Memory TESTED May 31 |
| N-003 | Few-shot prompting: -0.017 vs leader (full-stack + 5 exemplars = 0.540 vs 0.557 leader) | [4](#4-negative-results) | confirmed negative | docs/benchmarks.md section "Negative results levers we tested that regressed at 9B + adj=2" |
| N-004 | HyDE: -0.060 standalone, -0.071 vs leader when stacked; poison-pill across every stack it joins | [4](#4-negative-results) | confirmed negative | docs/benchmarks.md section "Negative results levers we tested that regressed at 9B + adj=2" |
| N-005 | answerai + bge-v2-m3 reranker stacked: 0.720 vs 0.760 answerai alone (subset-200 gemma) | [4](#4-negative-results) | confirmed negative | docs/benchmarks.md "Late-interaction retrieval", confirmed negative sentence |
| N-006 | gemma4:12b first A/B: generation errors stored as zero-scoring predictions; methods lesson, run was invalid | [4](#4-negative-results) | retracted (methods failure, not result) | CHANGELOG.md Unreleased Fixed, "Generation failures in the LoCoMo runner were stored as zero-scoring [generation_error: ...]" |
| N-007 | gemma4:12b generator A/B: 0.630/0.580 vs qwen3.5:9b 0.680 same config; not an upgrade | [4](#4-negative-results) | confirmed negative | bench host 20260611 gemma12b_redo verdicts |
| N-008 | ColBERT projected spaces (answerai 96-dim, colbertv2 128-dim) both 0.730 vs backbone 0.760 subset-200 gemma; no upgrade, 4x footprint trade noted | [4](#4-negative-results) | confirmed negative | bench host 20260612_142049 rescored_gemma results |
| N-009 | Surprise-boundary chunking is a coverage artifact: matched-turn-budget baseline (k=120, 0.9192) beats chunks at k=20 (0.7677); methods lesson, compare chunking levers at matched turn budget | [4](#4-negative-results) | confirmed negative | VPS e1_mb_k20/k120_20260612_083159.json |
| N-010 | Write-skip floor safe (no recall harm, zero evidence skipped) but fires on 2.3 percent of LoCoMo turns, under the 5 percent floor; not shipped | [4](#4-negative-results) | confirmed negative (for shipping) | VPS e1_write_skip_20260612_143625.json |
| N-011 | Surprisal is the WORST retention-priority signal (below random) at every budget; keeping high-surprisal turns is bad for retention. Length wins. Kills the v2 surprisal pillar, triggers the provenance/claims pivot | [4](#4-negative-results) | confirmed negative | VPS e008_retention_20260613_115258.json |
| N-012 | Embedding bake-off: no small ONNX embedder beats arctic-embed-s on LoCoMo R@K (arctic 0.823; e5-small 0.808, bge-small 0.773, granite-r2 0.742, gte-small 0.727); arctic retained, F-010 reaffirmed | [6](#6-ongoing-work-pre-registered) | confirmed negative | VPS e010_*_20260614_154450.json |
| N-013 | BEAM-100K config sweep: no lever beats the 47% plateau (all 7 arms 44-49%, n=100); self-verify is BEAM-neutral (off ties top pass-rate, leads avg_score 0.455), unlike LongMemEval F-013 | [4](#4-negative-results) | confirmed negative | benchmarks/results/beam_campaign_resume_20260616_221030.log |
| E-001 | Surprisal retrieval: BOTH arms dead. Prior flat; chunking +0.1263 at equal k but baseline at matched turn budget wins by 0.15 | [6](#6-ongoing-work-pre-registered) | resolved -> N-009 | VPS e1_mb_k20/k120_20260612_083159.json |
| E-002 | Extraction-hallucination rate: gemma extracts, qwen judges; reporting threshold 3-5% PARTIAL+UNSUPPORTED | [6](#6-ongoing-work-pre-registered) | resolved -> F-009 | STATUS.md "bench/e2-claim-verification" |
| E-003 | LoCoMo-Refined full run: 1382 revised questions, official qwen3:14b judge, leader recipe | [6](#6-ongoing-work-pre-registered) | methods limitation (judge unreliable on Ollama) | STATUS.md "all 1382 predictions matched" |
| E-004 | pylate projected-space vs backbone: both projected spaces 0.730 vs backbone 0.760 subset-200 gemma; no recipe upgrade | [6](#6-ongoing-work-pre-registered) | resolved -> N-008 | bench host 20260612_142049 rescored_gemma results |
| E-005 | Temporal date-range lever: LoCoMo CANNOT measure it (temporal-cat applicability 9.3 percent, under the 10 percent gate); ships default-off | [6](#6-ongoing-work-pre-registered) | resolved (not measurable on LoCoMo) | benchmarks/temporal_applicability_scan.py, master 0535f86 |
| E-006 | Write-skip floor: SAFE (delta +0.0051, zero evidence skipped) but fires on only 2.3 percent of turns, under the 5 percent shipping floor | [6](#6-ongoing-work-pre-registered) | resolved -> N-010 | VPS e1_write_skip_20260612_143625.json |
| E-007 | Arctic-embed vs MiniLM low-tier dense: CONFIRMED full-1540 (judged +0.0565, 0.7305 vs 0.6740) + R@K +0.157; clear default-change-worthy win | [6](#6-ongoing-work-pre-registered) | resolved -> F-010 (full-1540) | bench host e007full_*_20260613_131837 |
| E-008 | Surprisal as retention-priority signal: KILLED, worst of all policies at every budget (below random); length wins. Surprisal pillar dead, v2 pivots to provenance/claims | [6](#6-ongoing-work-pre-registered) | resolved -> N-011 | VPS e008_retention_20260613_115258.json |
| E-009 | Claims gate (Provable Memory): prefer_verified PASSES (served-hallucination 0.040 to 0.000 at no accuracy cost), strict fails; 200 LoCoMo QAs, llama3.1:8b judge | [6](#6-ongoing-work-pre-registered) | resolved -> F-011 | bench host benchmarks/results/e009.json |

---

## 1. Abstract

taOSmd is a local-first memory system for AI agents. It stores conversation turns in a zero-loss append-only archive, layers hybrid retrieval on top, and runs offline on modest hardware without cloud dependencies.

The current LongMemEval-S headline is 97.0% Recall@5 (500 questions, full test set), a retrieval metric: the correct evidence session appears in the top-5 retrieved. This is the same metric MemPalace (96.6%) and agentmemory (95.2%) publish, so it is a like-for-like comparison taOSmd leads. An earlier edition of this report and the public docs labelled this number "end-to-end Judge accuracy"; that was a mislabel, corrected 2026-06-14 (see F-001 and the revision log). A genuine end-to-end Judge measurement (retrieve, generate, grade) with the current stack is pre-registered as E-012.

On LoCoMo (1540 QAs, 10 multi-session conversations), the current leader recipe is MaxSim + reranking: qwen3.5:9b with retrieval-top-k 50, adjacent-turns 2, bge-v2-m3 reranker, and mem0_additive fusion. Full-1540 tri-judge scores: 0.748 lenient (gemma4:e2b), 0.394 strict (llama3.1:8b), 0.659 strict (qwen3:4b-instruct-2507). All judges are external to the generator family.

A reranker-free late-interaction lever (token-level MaxSim via answerai-colbert-small-v1, backbone path) scores 0.716 / 0.388 / 0.656 on the same full-1540 tri-judge and runs in under 110ms per query on a CPU-only 16-core VPS, making it the recommended recipe for tiers where a cross-encoder download is not affordable. All numbers are under external judges with no same-family inflation.

On BEAM (mem0's long-context memory benchmark, ICLR 2026), graded by a port of mem0's own nugget-rubric judge, the full local stack scores 47.0% at the 100K tier (n=100) and 37.5% at 1M (75/200). mem0's frontier numbers are 64 to 70% at 1M, so local-9B lands honestly behind: BEAM rewards frontier generation, and the value here is an early, fully-provenanced local-tier number rather than a leaderboard win (F-014). A seven-arm config sweep found no setting that beats the 47% plateau beyond sampling noise, and showed that the CoVe self-verification that dominated on LongMemEval-S does not transfer to BEAM (N-013).

---

## 2. Methodology

### 2.1 External multi-judge protocol

The core rule is that every accuracy number uses a judge model from a different family than the generator. Same-model judging is measured to inflate reported scores by 15 to 22 points on the LoCoMo task (source: docs/benchmarks.md "Judge sensitivity" section). On our own stack, the qwen3.5:9b generator self-judging its own output read 0.64 overall vs the external judge reading of 0.54 on the same predictions, a +0.10 inflation. That is consistent with the historic estimate in direction, somewhat below it in magnitude.

The tri-judge panel used for all full-1540 measurements (Jun 2026):

- **Lenient judge: gemma4:e2b.** The most lenient open-source judge we measured. Calibrated closer to gpt-4o-mini behavior: accepts inferred answers, is forgiving on minor wording differences. Useful for comparing against published SOTA numbers that also use a lenient API judge.
- **Strict judge 1: llama3.1:8b.** A non-thinking model that emits clean YES/NO verdicts. Compresses the field: it grades very harshly and the strict judge penalizes any answer that isn't nearly verbatim with the reference.
- **Strict judge 2: qwen3:4b-instruct-2507.** The non-thinking HF GGUF build of qwen3:4b-instruct. Used because the thinking-enabled qwen3:4b Ollama build no longer emits clean YES/NO verdicts in its current version and was retired as a judge in Jun 2026. Legacy leaderboard figures (for example RRF 0.557, mem0 0.540) came from the older thinking-mode qwen3:4b and are kept in docs/benchmarks.md for history but are not directly comparable to new strict-instruct numbers.

A fourth column, the **community-standard judge qwen3:14b**, is used to make late-interaction numbers comparable beyond the project's own panel. qwen3:14b is the official judge of the LoCoMo-Refined benchmark.

The legacy single-judge leaderboard in docs/benchmarks.md (entries from before Jun 2026) used the thinking-mode qwen3:4b and is kept for history. New cells added from Jun 2026 onward carry all three tri-judge scores.

For retrieval-only measurements the team uses **judge-free R@K**: the share of questions whose annotated evidence lands in the retrieved set, measured by the retrieval-latency probe without any LLM generation or grading. R@K is objective and does not depend on judge calibration.

### 2.2 Datasets

- **LongMemEval-S.** 500 questions, standard test set. Single and multi-session. Benchmark for end-to-end answer quality with the full pipeline. Dataset: github.com/xiaowu0162/LongMemEval (ICLR 2025).
- **LoCoMo-10.** 10 multi-session conversations, 1540 question-answer pairs across four categories: Single-hop, Temporal, Multi-hop, Open-domain. Conversations span 50+ sessions and 400-700 turns. A harder evaluation than LongMemEval-S: more categories, more QAs, and longer conversations. Benchmark harness: benchmarks/locomo_runner.py. Source: snap-research LoCoMo dataset.
- **LoCoMo-Refined (subset-comparable).** The revised 1382-question set from the LoCoMo-Refined leaderboard, using the corrected answer key and the official qwen3:14b judge. A full run was attempted but the official qwen3:14b judge proved unreliable under local Ollama (see E-003), so no refined-set number is reported. Our qwen3:14b column is on the original 1540-question set; it is judge-comparable to the Refined leaderboard but not set-comparable.
- **BEAM (100K / 1M / 10M).** mem0's long-context memory benchmark (ICLR 2026): conversations of 100K to 10M tokens with 2000+ nugget-graded questions across ten ability categories. Graded by mem0's own nugget-rubric judge (an answer passes at score at or above 0.5), ported in benchmarks/beam_runner.py so the grading matches mem0's published methodology. The dataset is public and ungated on HuggingFace (Mohammadta/BEAM for the 100K-1M tiers, Mohammadta/BEAM-10M). Used for an honest local-tier head-to-head against mem0's frontier numbers (1M 70.1% at top-200, 10M 50.5%).

### 2.3 Metrics

**End-to-end Judge accuracy** is the primary accuracy metric: what fraction of generated answers get marked correct by the judge, graded against the gold reference. It combines retrieval, generation, and grading in one number.

**R@K** (Retrieval at K) is used for retrieval-only levers where there is no LLM generation: what share of questions have their annotated evidence passage in the top-K retrieved results. It is judge-free and directly reflects retrieval quality.

**Latency** is reported as p50 and p95 per-query wall-clock in milliseconds. **Context tokens** are estimated from context chars / 4 in the LoCoMo runner and carried in the result JSON per row.

Per the three-number reporting rule: every accuracy claim in this report is accompanied by latency and context cost where those have been measured.

### 2.4 Hardware tiers

- **Orange Pi 5 Plus, 16 GB RAM (RK3588 NPU).** Reference low-end tier. The LongMemEval-S 97.0% Recall@5 retrieval stack was measured here (embedder: all-MiniLM-L6-v2 ONNX on CPU; Recall@5 is retrieval-only so no generator is involved in that number). The tier's generator for judged LoCoMo work is qwen3-4B via rkllama on the NPU. Source: docs/benchmarks.md hardware tiers section.
- **16-core x86 CPU VPS (no GPU).** Used for the retrieval-latency probe only. No LLM inference; embedder runs on CPU with ONNX Runtime. Source: docs/benchmarks.md "CPU-tier viability" subsection.
- **Fedora host, RTX 3060 12 GB.** Primary LoCoMo benchmark host. All full-1540 tri-judge numbers were measured here. Generator: qwen3.5:9b Q4_K_M via Ollama. Source: docs/benchmarks.md LoCoMo section.
- **GTX 1050 Ti LXC, 4 GB VRAM.** Measured at 0.530 ext judge (qwen3:4b, LXC container with CUDA). Source: docs/benchmarks.md "4 GB GPU" hardware tier.

### 2.5 Kill criteria and pre-registration

Experiments enter section 6 with a declared design and kill criterion before results exist. When results land, the experiment moves to section 3 or 4 with the original kill criterion quoted verbatim. Editing a kill criterion after results exist is falsification and is not done.

---

## 3. Results

### 3.1 LoCoMo leader table (full-1540, tri-judge)

**F-002.** The current leader on full-1540 LoCoMo is MaxSim + reranking. Recipe: qwen3.5:9b with `--retrieval-top-k 50 --adjacent-turns 2 --reranker bge-v2-m3 --fusion mem0_additive`. The bge-v2-m3 cross-encoder does late-interaction (MaxSim) scoring over a wider k=50 candidate pool.

Source: docs/benchmarks.md "Full-1540 leader (tri-judge, Jun 2026)."

| Recipe (qwen3.5:9b) | Lenient gemma4:e2b | Strict llama3.1:8b | Strict qwen3:4b-instruct-2507 |
|---|---|---|---|
| MaxSim + rerank | 0.748 | 0.394 | 0.659 |
| RRF (k=20 + llm-exp) | 0.723 | 0.390 | 0.634 |
| mem0_additive (k=20 + llm-exp) | 0.684 | 0.387 | 0.624 |

MaxSim + rerank leads on all three judges. The MaxSim > RRF > mem0_additive ordering is consistent across every judge. Both RRF and mem0_additive beat the older mem0-only default, so mem0_additive alone is no longer the recommended default for tiers where the reranker is affordable.

Latency and context-token cost for these full-1540 runs are tracked per-row in the result JSON (`mean_latency_ms`, `p95_latency_ms`, `mean_context_tokens`) as of the three-number bench summary feature shipped in PR #155. Per-run aggregate numbers are in the result files at benchmarks/results/ on the Fedora host.

### 3.2 Late-interaction retrieval (reranker-free, tri-judge)

**F-003.** Token-level MaxSim (ColBERT-style) scoring of retrieval candidates without any reranker. Probed with MiniLM per-token embeddings (--late-interaction) and with answerai-colbert-small-v1 (--colbert-model). Full-1540, same tri-judge rescore as the leader table. Baseline is identical config minus late interaction (mem0_additive, k=10, retrieval k=20, adj=2, no expansion, no reranker).

Source: docs/benchmarks.md "Late-interaction retrieval (token-level MaxSim at retrieval time, tri-judge)."

| Config (qwen3.5:9b, no reranker) | Scale | Lenient gemma4:e2b | Strict llama3.1:8b | Strict qwen3:4b-instruct-2507 |
|---|---|---|---|---|
| dense baseline | full 1540 | 0.674 | 0.377 | 0.598 |
| + MiniLM MaxSim (--late-interaction) | full 1540 | 0.711 | 0.378 | 0.642 |
| + answerai-colbert-small backbone (--colbert-model) | full 1540 | 0.716 | 0.388 | 0.656 |

Honest reading: the subset-200 read (+0.060 gemma) shrank to +0.037 gemma / +0.044 qwen-instruct at full scale. The strict llama judge sees near-nothing (+0.001), consistent with how that judge compresses the field. Still a real, two-judge-confirmed win at no reranker cost: 0.711 gemma vs the leader's 0.748 without the bge-v2-m3 download or its per-query latency.

The answerai-colbert-small-v1 (33M parameters) row is the backbone token output, not the trained ColBERT projection space. sentence-transformers does not apply projection heads on the token path. A pylate loader to measure the true projected space is queued as E-004.

Important caveat on N-005: stacking bge-v2-m3 reranker on top of answerai retrieval is a confirmed negative (subset-200 gemma: 0.720 vs 0.760 for answerai alone). The reranker reorders an already token-level-matched pool and hurts.

### 3.3 Community-standard judge column (qwen3:14b)

**F-005.** The same three full-1540 prediction files rejudged with qwen3:14b, the official judge of the LoCoMo-Refined benchmark. This makes the late-interaction numbers comparable beyond the project's own panel.

Source: docs/benchmarks.md "Community-standard judge column (qwen3:14b)."

Context for comparison: LoCoMo-Refined reports Mem0 at 48.91, EverMemOS at 58.25, MemPalace at 58.68, MemOS at 63.60, and MemoraX at 82.65 under this judge on their revised question set.

| Config (qwen3.5:9b, no reranker) | qwen3:14b |
|---|---|
| dense baseline | 0.487 |
| + MiniLM MaxSim | 0.532 |
| + answerai-colbert-small backbone | 0.542 |

The late-interaction levers gain more under qwen3:14b (+0.045 and +0.055 over baseline) than under the lenient gemma judge (+0.037 and +0.042). The ordering matches every other judge: dense, then MiniLM MaxSim, then answerai.

Caveat: these scores are on the original 1540-question set. The LoCoMo-Refined board uses the revised 1382-question set with a corrected answer key. These numbers are judge-comparable to competitors but not set-comparable. A full LoCoMo-Refined run was attempted; see E-003 for why no refined-set number is reported.

### 3.4 CPU-tier retrieval-only latency probe

**F-004.** Measured with benchmarks/retrieval_latency_probe.py on branch bench/retrieval-latency-probe. Subset-200, R@K = share of questions whose annotated evidence lands in the retrieved set. Per-query wall-clock on a 16-core x86 CPU VPS (no GPU) with `--reranker off`. No LLM inference; judge-free.

Source: docs/benchmarks.md "CPU-tier viability (retrieval-only, judge-free)."

| Retrieval config | R@K | p50 / p95 latency |
|---|---|---|
| dense (MiniLM ONNX) | 0.641 | 72 / 79 ms |
| + MiniLM MaxSim (--late-interaction) | 0.813 | 85 / 93 ms |
| + answerai-colbert-small backbone (--colbert-model) | 0.854 | 110 / 122 ms |

Late interaction costs 13 to 38 ms per query on CPU and buys +0.17 to +0.21 evidence recall. The lever is viable on CPU-only tiers, not just the GPU box.

The answerai row caveat carries over from section 3.2: backbone token output only; pylate projected-space comparison is queued as E-004.

### 3.5 LongMemEval-S Recall@5

**F-001 (CORRECTED 2026-06-14).** 97.0% Recall@5 on LongMemEval-S, 500 questions, standard test set. Recall@5 is a retrieval metric: the correct evidence session appears in the top-5 retrieved (no generation, no answer grading). This is the same metric MemPalace (96.6%) and agentmemory (95.2%) publish, so it is a like-for-like comparison and taOSmd leads it. Harness: benchmarks/longmemeval_recall.py. Strategy: hybrid + query expansion. Measured on the all-MiniLM-L6-v2 ONNX embedder.

Provenance: benchmarks/results/enhanced_20260413_133215.json, the query_expand variant at top_k=5 (the source of the per-category table below).

Correction note: earlier editions of this finding, the README, and docs/benchmarks.md labelled this number "end-to-end Judge accuracy." That was a mislabel introduced when the public docs were reframed in April; the underlying result file is and always was Recall@5 (top_k=5). No end-to-end Judge run ever produced 97.0%. The basic Judge harness (benchmarks/longmemeval_runner.py) reproduces about 22.6% with its default small generator, with a completely different per-category shape (temporal 3.8% vs the 94.0% Recall@5 below), which is what surfaced the mislabel. A genuine end-to-end Judge measurement with the current stack is pre-registered as E-012 and will be reported with its own provenance; it is a separate number.

| Category | hybrid + expand (Recall@5) | raw semantic (Recall@5) |
|---|---|---|
| knowledge-update | 100.0% (78/78) | 100.0% |
| multi-session | 98.5% (131/133) | 95.5% |
| single-session-user | 97.1% (68/70) | 90.0% |
| single-session-assistant | 96.4% (54/56) | 96.4% |
| temporal-reasoning | 94.0% (125/133) | 94.0% |
| single-session-preference | 90.0% (27/30) | 93.3% |
| Overall | 97.0% (485/500) | 95.0% (475/500) |

**F-006.** The Librarian vocabulary-gap lever: +15.4% composite on a long-horizon session benchmark (60 turns, fact buried at turn 5, query and fact use different surface words). Full pipeline plus Librarian 0.810 vs full pipeline without Librarian 0.752. Measured on Fedora host with gemma4:e2b, 2026-04-15.

Source: docs/benchmarks.md, Librarian layer vocabulary-gap benchmark section.

### 3.6 Shipped recipe: lateint-9b

**F-007.** The `lateint-9b` recipe ships in the package as a declared configuration bundle. It uses answerai-colbert-small-v1 with mem0_additive fusion, top_k=10, retrieval_top_k=20, adjacent_neighbors=2, and no reranker. Measured full-1540 tri-judge scores: gemma4:e2b 0.716, llama3.1:8b 0.388, qwen3:4b-instruct-2507 0.656, qwen3:14b 0.542.

Source: CHANGELOG.md Unreleased section, "Late-interaction retrieval lever and lateint-9b recipe."

This recipe is the recommended configuration for tiers where a cross-encoder download is not affordable (CPU-only, memory-constrained, or bandwidth-limited deployments).

---

### 3.5 Extraction-Hallucination Rate (F-009)

A cross-family claim-verification pass measured how often the extraction step stores claims that are not fully supported by their source turns. gemma4:e2b extracted claims; a different family, qwen3:4b-instruct-2507, judged each claim SUPPORTED, PARTIAL, or UNSUPPORTED against the exact source text the extractor saw. Using different families on the two sides guards against a model rubber-stamping its own output.

Across 526 claims from three conversations, the PARTIAL plus UNSUPPORTED rate was 18.8 percent (verifier error rate 0.2 percent, one claim). Per conversation: conv-26 22.0 percent (168 claims), conv-30 12.6 percent (103 claims), conv-41 19.2 percent (255 claims). In plain terms, close to one in five extracted facts is not fully supported by the turn it was drawn from. This is a measurement nobody else in this space reports, and it is the quantitative case for the provenance-and-verification direction: a memory layer that keeps the source can measure and re-verify its own claims, while an extraction-only layer cannot.

Source: bench host results 20260611 e2 claim-verifier pairs and summary.

### 3.7 Low-tier dense embedder: arctic-embed vs MiniLM

The low-end and offline tier this project targets uses a dense ONNX embedder (all-MiniLM-L6-v2) rather than the leader's late-interaction path. E-007 asked whether a modern small retriever beats it. Snowflake arctic-embed-s (33M, 384 dim, a clean MiniLM drop-in) does, decisively, at the same dimension and latency.

| Embedder (dense, subset-200, same harness) | R@K (judge-free) | Judged (qwen3.5:9b gen, gemma4:e2b) |
|---|---|---|
| all-MiniLM-L6-v2 (baseline) | 0.6768 | 0.6750 subset / 0.6740 full-1540 |
| arctic-embed-s (33M) | 0.8333 (+0.1565) | 0.7150 (+0.0400) subset / **0.7305 (+0.0565) full-1540** |
| arctic-embed-xs (22M) | 0.7374 (+0.0606) | not yet judged |

Both stages are on subset-200, so expect plus or minus 0.02 judge noise; the +0.040 judged delta is twice that and consistent with the much larger retrieval signal. A methodology note that mattered: arctic-embed is asymmetric (a query-only prefix) and CLS-pooled; the ONNX path had to be taught both or arctic would have mean-pooled with no prefix and posted a false negative (enabler PR #160, default-off). The full-1540 judged confirm landed and strengthened the result: arctic-embed-s 0.7305 vs MiniLM 0.6740, +0.0565, larger than the subset-200 +0.040 and well clear of noise at 1540 questions. The win is confirmed at scale and is default-change-worthy; the low-tier dense default should move MiniLM to arctic-embed-s (shipped via a sign-off PR across recipes, config, README, and the tier tables, plus a small Pi-CPU speed sample). Provenance: bench host benchmarks/results/e007_{minilm_baseline,arctic_s,arctic_xs}_20260613_122113.json (R@K) and e007c_{minilm,arctic_s}_20260613_122948.rescored_gemma.json (judged).

### 3.8 BEAM long-context (mem0 nugget-rubric judge)

BEAM (mem0, ICLR 2026) is a long-context memory benchmark at the 100K, 1M, and 10M token scales, graded by a nugget-rubric judge where an answer passes at score 0.5 or above. It is the head-to-head where mem0 publishes frontier numbers (1M 70.1% at top-200, 10M 50.5%), so it is the cleanest place to state an honest local-tier number against a public frontier baseline. taOSmd runs its own port of the BEAM nugget judge (benchmarks/beam_runner.py) so the grading matches mem0's methodology, over the full local stack: arctic-embed-s, tight retrieval (6 chunks, FTS 3, bge-v2-m3 rerank to top 4, 2000 assemble tokens, 6000 context chars), qwen3.5:9b generator, Qwen3-4B-Instruct judge.

Measured (F-014): BEAM-100K 47.0% pass (n=100), BEAM-1M 37.5% (75/200). Against mem0's 64 to 70% at 1M, local-9B lands honestly behind, exactly as the thesis predicts. BEAM rewards frontier generation, so the value is the early, fully-provenanced local-tier number, not a leaderboard win. The intelligent-context-release lever from the earlier context-amount sweep is already in this configuration: tight 6-chunk context beats a rich 40-chunk context by +16pp on the 9B, because more context buries a small model.

A seven-arm config sweep on BEAM-100K (chunk size, chunking mode, self-verify, decomposition; n=100 per arm) found no lever that beats the 47% plateau beyond sampling noise (N-013):

| Arm (BEAM-100K, n=100) | pass% | avg_score |
|---|---|---|
| base (chunk 100, self-verify) | 47.0 | 0.436 |
| chunk 50 | 47.0 | 0.417 |
| chunk 200 | 49.0 | 0.425 |
| chunk 300 | 44.0 | 0.394 |
| sentence-mode (120 words) | 47.0 | 0.425 |
| chunk 100, self-verify OFF | 49.0 | 0.455 |
| chunk 100, decompose ON | 47.0 | 0.436 |

At n=100 the pass-rate standard error is about 5 points, so the 44 to 49 spread is one band around 47. The one signal worth carrying forward is that self-verify gives no BEAM benefit: the self-verify-off arm ties the top pass-rate and leads on avg_score (0.455). On LongMemEval-S the same CoVe self-verification was the dominant +17.8pp lever (F-013); it does not transfer to BEAM because BEAM failures are wrong answers, not abstentions, and a verify-or-abstain pass can only recover the latter. Provenance: benchmarks/results/beam_campaign_resume_20260616_221030.log (arms 3 to 7) and beam_campaign_20260616_115113.log (arms 1 to 2), branch feat/beam-runner.

## 4. Negative Results

Negative results are recorded with the same rigor as wins. A lever that failed is a finding.

**N-001. CLAG cluster pre-filter.**

A frontier idea aimed at small-model latency: cosine k-means over the candidate pool, keeping only the query's nearest cluster(s) before ranking. On the 200-QA subset every variant regressed. The worst variant (keep nearest 1 cluster) scored -0.285 gemma / -0.205 qwen3:4b vs baseline. The keep-sweep was monotonic; relaxing the pruning moved toward baseline but never beat it. Cluster pruning only ever discards relevant evidence. The failure pattern matched HyDE, Chain-of-Memory, and thinking-mode: a lever tuned for frontier models that regresses at the compute tier targeted by taOSmd. Not shipped. A benchmark-only --clag flag is retained for reproduction.

Source: docs/benchmarks.md, CLAG cluster pre-filter section.

**N-002. Chain-of-Memory.**

Tested May 31 2026. Result: -0.16 single-hop for +0.02 overall at 1.8x latency. The paper is frontier-tuned and lossy at the compute tier this project targets. Not shipped.

Source: project memory reference_memory_landscape_may2026.md, entry: "Chain-of-Memory TESTED May 31 REGRESSED on our stack."

Note on provenance: the exact numeric breakdown (-0.16 SH / +0.02 overall / 1.8x latency) comes from the session memory record rather than a committed benchmarks.md section. If a future edition adds this run to docs/benchmarks.md, the numbers here should be verified against that record.

**N-003. Few-shot prompting.**

Adding 5 exemplars to the answer prompt on top of the leader recipe: full-stack + few-shot lands at 0.540 vs the 0.557 leader (qwen3:4b strict, full-1540). Delta: -0.017. The same regression appears at the Pi NPU tier (-0.11 vs the Pi full-stack leader of 0.490). Direction-consistent across model sizes. The exemplars compete for context budget without adding retrieval signal.

Source: docs/benchmarks.md section "Negative results levers we tested that regressed at 9B + adj=2", row `few_shot_full_stack`.

**N-004. HyDE (Hypothetical Document Embeddings).**

Standalone HyDE scores 0.456 vs the 0.516 adj=2 baseline (qwen3:4b strict, full LoCoMo 1540), a -0.060 regression. When stacked on the leader recipe, HyDE pulls the leader from 0.557 down to 0.486: -0.071 vs the leader. Classified as a poison-pill: HyDE assumes the generator can imagine a good hypothetical answer. Small generators hallucinate the answer's structure, the embedding lookup matches that hallucination, and recall collapses. Direction-consistent at the Pi NPU tier (-0.057 vs Pi baseline). Do not enable HyDE on memory-recall benchmarks at this generator tier.

Source: docs/benchmarks.md section "Negative results levers we tested that regressed at 9B + adj=2", rows `hyde` and `hyde_full_stack`.

**N-005. answerai + bge-v2-m3 reranker stacked.**

Stacking the bge-v2-m3 cross-encoder reranker on top of answerai-colbert-small retrieval is a confirmed negative: subset-200 gemma 0.720 vs 0.760 for answerai alone. The reranker reorders an already token-level-matched pool and hurts. Recorded so the combination is not retried without new evidence.

Source: docs/benchmarks.md "Late-interaction retrieval" section, sentence: "Stacking the bge-v2-m3 cross-encoder reranker on top of answerai retrieval is a confirmed negative."

**N-006. gemma4:12b first A/B: generation errors stored as zero-scoring predictions (methods failure).**

This is a methods lesson rather than a substantive result. Before a fix in the LoCoMo runner (CHANGELOG.md Unreleased, Fixed section), generation failures were stored as zero-scoring `[generation_error: ...]` predictions. A run with a missing Ollama model would impersonate a catastrophically bad generator rather than failing loudly. An early gemma4:12b A/B run produced numbers under these conditions; the run was invalid. The fix makes all-failed runs abort with exit code 1. Any gemma4:12b numbers from before this fix are not included in this report.

Source: CHANGELOG.md Unreleased Fixed: "Generation failures in the LoCoMo runner were stored as zero-scoring [generation_error: ...] predictions."

---

**N-007. gemma4:12b as the generator.**

The newest in-budget generator candidate (released one week before testing, marketed as near-26B quality at half the memory) lost the A/B against the incumbent: 0.630 lenient / 0.580 strict qwen-instruct on the subset-200 dense baseline, versus 0.680 for qwen3.5:9b on the same config. The lenient judge shares the challenger's model family, which biases toward the challenger, making the loss more credible rather than less. Recipes keep qwen3.5:9b. The first run of this A/B is recorded separately as a methods lesson: an unavailable model impersonated a bad generator through silently stored generation errors (see N-006).

Source: bench host results 20260611 gemma12b_redo verdicts; docs/benchmarks.md, the generator reading list.

**N-008. ColBERT projected spaces do not beat the backbone at our tier (E-004 resolved).**

Loading the trained projection heads through pylate, both proper ColBERT spaces score 0.730 on subset-200 gemma vs 0.760 for the answerai backbone token space the lateint-9b recipe ships with: answerai-colbert-small-v1 projected (96-dim) 0.730, colbert-ir/colbertv2.0 projected (128-dim) 0.730. The pre-registered rule ("upgrades only if projected beats backbone outside noise") resolves this as no upgrade. Two honest notes. First, the projected spaces are real and working: an earlier 0.050 reading was a search-path bug (the pylate-mode query gate returned empty before MaxSim ever ran), not a property of the models, and the trained heads were verified loading correctly. Second, the answerai projected space stores 96-dim token vectors against the backbone's 384, a 4x token-matrix footprint reduction for -0.030 quality; that is a defensible swap on storage-constrained tiers, recorded here so the option is not forgotten, but it is not the default.

Source: bench host benchmarks/results/20260612_142049_{answerai_small,colbertv2}.rescored_gemma.json, run log colbert_models_20260612_142049.log; cross-referenced in docs/benchmarks.md late-interaction section.

**N-009. Surprise-boundary chunking is a coverage artifact (E-001 resolved, both arms dead).**

The surprisal pillar's retrieval-side claims are both negative. The prior arm (weighting candidate scores by turn surprisal) was flat at +0.0000 across every weight tested. The chunking arm (merging predictable turns into chunks at surprise boundaries) showed +0.1263 at equal k, but a matched-turn-budget control kills it: baseline handed the same raw turn count directly (k=120, 0.9192) beats chunking at k=20 (0.7677, roughly 120 turns via 6-turn chunks) by 0.15. The apparent gain was retrieved-turn coverage, not surprisal signal. Methods lesson worth keeping: R@K favors any variant that packs more turns per retrieved item, so chunking-style levers must always be compared at matched turn budget, not matched k. What survives of the surprisal idea: the write-skip floor (E-006, running) and surprisal-seeded trace strength for consolidation priority (a v2 question, not a retrieval lever).

Source: VPS benchmarks/results/e1_mb_k20_20260612_083159.json and e1_mb_k120_20260612_083159.json, git 1427f54.

**N-010. Write-skip floor: safe but fires too rarely to ship (E-006 resolved).**

Skipping highly predictable short turns at ingest (surprisal z at or below -1.0 and under 12 tokens) left R@K within noise (+0.0051) and skipped zero evidence turns, so the classifier causes no measurable recall harm at these thresholds. But it only fired on 18 of 788 turns (2.3 percent), under the pre-registered 5 percent shipping floor: on LoCoMo there are simply not enough bare greetings and acknowledgements for the floor to save meaningful index weight. Not shipped. The thresholds were deliberately conservative; loosening them to hit 5 percent would change the safety result and would need a fresh pre-registration. Worth revisiting only on a corpus with denser conversational filler.

Source: VPS benchmarks/results/e1_write_skip_20260612_143625.json, variant code on branch bench/e1-write-skip (46793e0).

**N-011. Surprisal is a bad retention-priority signal (the v2 consolidation kill-shot fails).**

The last surviving role for the v2 surprisal pillar was consolidation priority: when an agent must forget down to a budget, is surprisal a good "keep" signal? It is the worst one tested. Under retention budgets of 25/50/75 percent on subset-200 (judge-free R@K, evidence on a dropped turn counts as a miss), keeping the highest-surprisal turns scored 0.313/0.444/0.591, below random (0.258/0.424/0.571) at every budget, and far below the winner. The winner is length: keep the longest turns, 0.525/0.641/0.702. Recency was weakest. The reading: high token-surprisal marks unpredictable content, not evidence-bearing content, and short surprising turns (the ones the write-skip floor also targeted) carry little retrievable evidence, so prioritising them throws away the longer, denser turns that answers actually need. With its retrieval role already dead (N-009, N-010), the surprisal pillar of v2 is conclusively killed; the spine pivots to the provenance and claims layer (F-009). The length result is a genuine positive side-finding, a cheap, model-free retention heuristic, recorded for its own future pre-registration rather than claimed here.

Source: VPS benchmarks/results/e008_retention_20260613_115258.json, branch bench/e1-retention.

**N-013. BEAM-100K config sweep: no lever beats the 47% plateau; self-verify is BEAM-neutral.**

Seven arms (chunk size 50/100/200/300, sentence-mode, self-verify on and off, decomposition on) on BEAM-100K at n=100 each, over the full local stack (arctic-embed-s, tight retrieval, qwen3.5:9b generator, Qwen3-4B-Instruct judge). Every arm landed in the 44 to 49% band, one standard error around the 47% tight-retrieval baseline. chunk200 and self-verify-off are nominally highest at 49% but within noise, so no configuration lever moves the local-9B BEAM-100K number. The transferable sub-finding: CoVe answer self-verification, the dominant +17.8pp lever on LongMemEval-S (F-013), gives no BEAM benefit; the self-verify-off arm ties the top pass-rate and leads avg_score (0.455), because BEAM failures are wrong answers rather than abstentions and a verify-or-abstain pass can only recover abstentions. The 47% plateau confirms local-9B sits structurally behind frontier generation on BEAM (mem0 64 to 70% at 1M); the product story is the provenanced local-tier number, not a tuned score. The full per-arm table is in section 3.8.

Source: benchmarks/results/beam_campaign_resume_20260616_221030.log and beam_campaign_20260616_115113.log, branch feat/beam-runner.

## 5. Reproducibility

**Install.**

```
pip install taosmd
```

**LoCoMo harness.**

```
python benchmarks/locomo_runner.py --model qwen3.5:9b \
    --retrieval-top-k 20 --adjacent-turns 2 \
    --llm-query-expansion --fusion mem0_additive
```

Late-interaction and ColBERT flags:

```
python benchmarks/locomo_runner.py --model qwen3.5:9b \
    --late-interaction
python benchmarks/locomo_runner.py --model qwen3.5:9b \
    --colbert-model answerdotai/answerai-colbert-small-v1
```

**Retrieval-latency probe (no LLM, judge-free).**

```
python benchmarks/retrieval_latency_probe.py
```

Branch: bench/retrieval-latency-probe.

**LongMemEval-S harness.**

```
python benchmarks/longmemeval_runner.py  # end-to-end Judge
python benchmarks/longmemeval_recall.py  # Recall@5 only
```

**Model versions as used in measurements.**

- Generator (LoCoMo leader): qwen3.5:9b Q4_K_M via Ollama
- Generator (Orange Pi LongMemEval-S): qwen3-4B via rkllama on RK3588 NPU
- Embedder: all-MiniLM-L6-v2 ONNX (onnx-models/all-MiniLM-L6-v2-onnx)
- Late-interaction model: answerdotai/answerai-colbert-small-v1 (33M, sentence-transformers backbone path)
- Reranker (leader recipe): cross-encoder/ms-marco-MiniLM-L-6-v2 ONNX; leader recipe uses bge-v2-m3
- Lenient judge: gemma4:e2b via Ollama
- Strict judge 1: llama3.1:8b via Ollama
- Strict judge 2: qwen3:4b-instruct-2507 (HF GGUF)
- Community-standard judge: qwen3:14b via Ollama

**Datasets.**

- LoCoMo: snap-research LoCoMo dataset (10 conversations, 1540 QAs). Run via benchmarks/locomo_runner.py.
- LongMemEval-S: github.com/xiaowu0162/LongMemEval (ICLR 2025), standard test set, 500 questions.

Every result JSON under benchmarks/results/ on the bench host carries a `meta` block with commit SHA, model, and dataset path. Prediction files for the full-1540 late-interaction runs are available on request.

---

## 6. Ongoing Work (Pre-registered)

These experiments have declared designs and kill criteria. Results will be recorded verbatim whatever they are.

**E-001. Surprisal retrieval prior and surprise-boundary chunking.**

Hypothesis: turns that carry high surprisal relative to the running conversation are more likely to contain important facts worth preferential retrieval. The lever would weight candidate scoring by a surprisal signal estimated from the embedding shift between turns, and use high-surprisal points as chunk boundaries.

Kill criterion (verbatim): "no variant beats baseline R@K by more than 0.02 on subset-200."

Result (2026-06-11), recorded honestly: the surprisal PRIOR variant was flat, +0.0000 R@K at every weight (0.25, 0.5, 1.0) over a 0.641 baseline on subset-200. The prior path is dead. The surprise-boundary CHUNKING variant reported +0.0859 (0.727), which clears the kill threshold on paper, BUT the result is confounded and is NOT being claimed as a win: the chunker produced a mean chunk length of 22.54 turns against a stated maximum of 6, which means its size cap was not being enforced, and evidence was credited per-chunk rather than per-turn, so a single giant chunk trivially counted as a hit for every turn it spanned. A corrected re-run is in flight on the CPU VPS (size cap enforced, per-turn evidence credit, prior variant dropped); only its numbers will count. Until then E-001 is unresolved.

Resolved (2026-06-12), recorded as negative result N-009. The corrected runs (chunk cap enforced, per-turn evidence credit, git 1427f54) measured both variants at two retrieval budgets on subset-200. At equal k=20 chunking still looked good: 0.7677 vs 0.6414 baseline, +0.1263. The matched-turn-budget control exposes it: a 6-turn chunk at k=20 delivers about 120 raw turns, and plain baseline given the same budget directly (k=120) scores 0.9192, beating chunking at k=20 by 0.15. Chunking at k=120 (0.9899) repeats the same confound at six times the payload. Quoting the kill criterion: "E1 is killed if no variant beats baseline R@K by more than 0.02": at matched turn budget no variant does, so the chunking arm is killed alongside the already-flat prior arm. Surprise-boundary chunking adds no retrieval signal beyond raw turn count, and R@K alone flatters anything that stuffs more turns per retrieved item, which is exactly why the matched-budget control was pre-committed. Provenance: VPS benchmarks/results/e1_mb_k20_20260612_083159.json and e1_mb_k120_20260612_083159.json.

**E-002. Extraction-hallucination rate via cross-family claim verification.**

Hypothesis: the extraction step (where facts are pulled from turns and stored as memories) may introduce hallucinated or unsupported claims that compound over time. Measuring this rate would tell us whether extraction quality is a ceiling for the overall pipeline.

Design: gemma4:e2b extracts claims from stored memories; qwen3:4b-instruct judges whether each claim is SUPPORTED, PARTIAL, or UNSUPPORTED against the source turns.

Reporting threshold (verbatim): "a PARTIAL plus UNSUPPORTED rate above 3 to 5 percent is a finding in itself."

Result (2026-06-11): RESOLVED, recorded as finding F-009. The PARTIAL plus UNSUPPORTED rate was 18.8 percent across 526 claims from three conversations, far above the 3 to 5 percent reporting threshold, so it is a finding. See F-009 in Results.

**E-003. LoCoMo-Refined full run.**

ATTEMPTED, RESOLVED AS A METHODS LIMITATION (2026-06-11/12). We ran our leader recipe against the refined 1382-question set (all predictions generated and matched) and attempted to score them with the benchmark's official judge, qwen3:14b, on local Ollama. The judge stage could not be made reliable: across five mitigations (a configurable client timeout, the Qwen3 /no_think soft-switch since Ollama ignores the harness's vLLM-style enable_thinking flag, a max-token cap, terminal JSON-only output enforcement, and tolerant per-question scoring) qwen3:14b kept returning prose instead of the required JSON object on a large fraction of questions, and crucially those failures concentrated on TEMPORAL questions, the category the refined judge is strictest on. Tolerant scoring (count an unjudgeable response as wrong) would therefore deflate the score non-representatively, by judge mechanics rather than by our system's quality, so we do not report a number: a temporally-biased lower bound would mislead more than it informs.

The honest takeaway is itself a finding: the official qwen3:14b judge is not reliably steerable to clean structured output under local Ollama for these reasoning-heavy prompts. The closest comparable we can stand behind is the qwen3:14b community-judge column on the ORIGINAL 1540-question set (F-005: dense 0.487, MiniLM MaxSim 0.532, answerai 0.542), which uses the same judge model but our original questions, not the revised refined set, so it is judge-comparable to the refined board's model choice but not set-comparable. The five harness fixes are a candidate upstream contribution to the LoCoMo-Refined project (configurable timeout/retries and an Ollama-compatible thinking-disable).

**E-004. pylate projected-space vs backbone comparison.**

The current answerai numbers (F-003, F-004, F-007) use the model's 384-dim backbone token output. sentence-transformers does not apply projection heads on the token path, so these are not the trained ColBERT projection space. Branch feat/pylate-loader (221 tests green, pushed) loads ColBERT models through pylate so projection heads apply; it auto-detects output dimension per instance.

Decision rule (verbatim): "the recipe model upgrades only if projected beats backbone outside noise on subset-200."

First attempt (2026-06-11) was INCONCLUSIVE due to a probable loader fault: answerai loaded through the pylate projected path scored 0.050 on subset-200, against 0.760 for the same model's backbone. A trained projection head cannot make a model fifteen times worse than its own backbone, so this is read as a pylate-path bug, not a real projected-space result. The pylate probe is held until the loader is root-caused; no projected-space number is claimed.

Root cause found (2026-06-12), and the pylate loader is exonerated: a direct test on the bench host (pylate 1.5.1) loads answerai-colbert-small-v1 with its trained Stanford-format projection applied correctly, Dense (96, 384) with real weights, 96-dim token output matching the checkpoint's artifact.metadata dim of 96. The 0.050 result (and a colbertv2.0 run that scored 0.0 retrieval recall across the board) came from the search gate bug: search() called embed(), which returns empty when only a pylate model is loaded, so every query exited before MaxSim ever ran. The bench host was also still running a pre-fix checkout. With the fix (commit b3b83e4) deployed, the full comparison (answerai projected 96-dim and colbertv2.0 projected 128-dim, each in a fresh store, subset-200 with the standard gemma rescore) is re-running. The earlier "96 vs 128" blocker note was a misdiagnosis.

Models to compare: answerai-colbert-small-v1, colbertv2.0 (both load via pylate's Stanford-format conversion), LateOn, ColBERT-Zero (later).

Resolved (2026-06-12), recorded as negative result N-008. Quoting the decision rule: "the recipe model upgrades only if projected beats backbone outside noise on subset-200." Both projected spaces landed at 0.730 subset-200 gemma against the answerai backbone's 0.760: answerai projected (96-dim) 0.730, colbertv2.0 projected (128-dim) 0.730. Neither beats backbone; the recipe stays on the backbone path. The pylate path was confirmed in the run logs (no fallback warnings; the loader was separately proven to apply the trained 96-dim Stanford head). LateOn and ColBERT-Zero are not worth a GPU window unless something changes the prior. See N-008 for the full reading including the footprint trade.

**E-005. Temporal date-range lever.**

Hypothesis: when a query contains a natural-language temporal expression ("last week", "in May 2023", "on 8 May, 2023"), parsing it into a date range and filtering or boosting candidates whose stored timestamps fall inside that range improves temporal-category retrieval. The lever is a deterministic parser, no model in the loop, so it is nearly free at query time.

Origin, stated honestly: this is a port of the temporal parser from an earlier internal project (engram), which scored 94.6 percent on LoCoMo temporal in one configuration. That number is single-conversation, small-n, and judged by a model protocol we do not use, so it is treated as motivation only, not evidence. Implementation merged to master in 0535f86 (PR #157), default off.

Design, two stages. Stage 1, applicability scan: run the expression extractor over all 1540 LoCoMo questions and count how many contain a parseable date-range expression, overall and within the temporal category. This is the honest first question, because most LoCoMo temporal questions ASK for a date ("When did X happen?") rather than constrain by one, and a lever that fires on too few questions cannot move the category no matter how good it is. Stage 2, only if applicability clears the floor: subset-200 judge-free R@K on the applicable questions, lever off (baseline) vs boost mode (0.25) vs filter mode, with the reference time set to each conversation's final session date so relative expressions resolve against the conversation, not the wall clock.

Kill criterion (verbatim): "if applicability is at least 10 percent of temporal-category questions, the lever is killed for LoCoMo unless boost or filter beats baseline R@K on the applicable subset by more than 0.02; if applicability is under 10 percent, LoCoMo is recorded as unable to measure this lever, no LoCoMo claim is made in either direction, and the lever ships default-off as a product feature with its applicability number documented."

Stage 1 result (2026-06-12), RESOLVED BY THE PRE-REGISTERED GATE: temporal-category applicability is 30 of 321 questions, 9.3 percent, under the 10 percent floor. Quoting the criterion: "if applicability is under 10 percent, LoCoMo is recorded as unable to measure this lever, no LoCoMo claim is made in either direction, and the lever ships default-off as a product feature with its applicability number documented." So recorded: LoCoMo cannot measure the temporal date-range lever, because LoCoMo temporal questions overwhelmingly ask FOR a date rather than constrain BY one. Full scan: single-hop 2.1 percent, temporal 9.3 percent, multi-hop 2.1 percent, open-domain 15.8 percent, overall 171/1540 (11.1 percent). Provenance: benchmarks/temporal_applicability_scan.py (master 0535f86) against data/locomo/data/locomo10.json. An honest observation, not a criterion change: the open-domain category clears 15.8 percent, so a separately pre-registered experiment on the applicable 171-question subset remains possible if the lever ever needs a LoCoMo number. The lever ships default-off; its natural habitat is live agent queries ("what did we decide last week"), which LoCoMo does not contain.

**E-006. Write-skip floor: do near-zero-surprise short turns earn their index entry?**

Hypothesis: turns that are both highly predictable (surprisal z at or below -1.0 against the conversation) and short (under 12 whitespace tokens) are greetings and acknowledgements that add index weight without adding recall. Skipping them at ingest should leave R@K unchanged while shrinking the corpus. Nothing is lost either way: the archive keeps every turn; skipping only means no index entry.

Design: the surprisal probe (benchmarks/surprisal_probe.py, branch bench/e1-write-skip) gains a write_skip variant, identical ingest to baseline except skippable turns are not added. Per-turn surprisal reuses the probe's cached scores, no re-scoring. Evidence credit stays per-turn: a skipped turn that was evidence for a question counts as a miss, which is exactly the cost being measured. Subset-200, judge-free R@K, queued on the CPU VPS behind the E-001 matched-budget re-run.

Kill criterion (verbatim): "write-skip is killed if R@K drops by more than 0.005; it ships only if R@K is within noise AND turns_skipped >= 5 percent of corpus (otherwise it saves nothing)."

Resolved (2026-06-12), recorded as N-010. Verdict line: delta=+0.0051, turns_skipped=18/788 (2.3 percent), evidence_turns_skipped=0. Quoting the criterion: "write-skip is killed if R@K drops by more than 0.005; it ships only if R@K is within noise AND turns_skipped >= 5 percent of corpus (otherwise it saves nothing)." The first half passes (no harm: R@K within noise, marginally up, and none of the skipped turns carried evidence, so the classifier is safe at these thresholds). The second half fails: 2.3 percent skipped is under the 5 percent floor, so the floor saves nothing worth its complexity on this corpus and does not ship. Honest caveat for later: LoCoMo turns are long dialogue contributions with few bare acknowledgements; a real agent transcript corpus with denser greetings might clear the floor, and re-running this probe on such a corpus is the right precondition for revisiting. Provenance: VPS benchmarks/results/e1_write_skip_20260612_143625.json, branch bench/e1-write-skip (46793e0).

**E-007. Arctic-embed as the low-power-tier dense embedder.**

Hypothesis: Snowflake arctic-embed-s (33M) and arctic-embed-xs (22M), strong small MTEB retrieval models, beat the current all-MiniLM ONNX dense baseline (subset-200 R@K 0.641 on the CPU probe) as the dense retriever at comparable size and latency, which would make one of them a better default for the low-end and offline tier this project targets. Both are 384-dim, a clean MiniLM drop-in.

Methodology note that gates validity: arctic-embed is asymmetric and CLS-pooled. It needs the query-only prefix "Represent this sentence for searching relevant passages: " (no document prefix) and CLS-token pooling, not mean pooling. Pointing the existing ONNX path at it without those silently cripples it and would produce a false negative. Support added in feat/arctic-embed-onnx (query-only prefix + CLS pooling, selected by model name, with tests) before any run; the experiment is invalid without it.

Design: the judge-free retrieval probe (benchmarks/retrieval_latency_probe.py), subset-200, --embed-mode onnx, --reranker off, dense only (no late interaction), R@K and per-query latency. Three arms at matched harness: all-MiniLM baseline, arctic-embed-s, arctic-embed-xs, plus the int8 ONNX of the winner if one emerges (the low-power tier cares about size). On the Fedora GPU host for throughput; the retrieval logic is identical to the CPU probe, only embedding is accelerated.

Kill criterion (verbatim): "arctic-embed replaces all-MiniLM as the low-tier dense default only if it beats MiniLM R@K by more than 0.02 on subset-200 at the matched harness; a tie or loss keeps MiniLM (smaller, proven). A winner is then confirmed with a judged LoCoMo run before any recipe or default changes."

Stage 1 result (2026-06-13), R@K gate PASSED decisively: judge-free evidence recall on subset-200 at matched harness (dense, reranker off, adj=2, identical flags, only the embedder swapped), MiniLM 0.6768, arctic-embed-s 0.8333 (+0.1565), arctic-embed-xs 0.7374 (+0.0606). Both clear the 0.02 gate; per-query latency is identical (~705ms p50, same harness, the embedding cost difference is dwarfed by the rest of the pipeline). This is judge-free recall, so no judge inflation to discount; the gain is consistent with MiniLM-L6 being an old, weak retriever against a modern small one. Per the criterion a judged LoCoMo confirm is required before any default change: that run (arctic-s vs MiniLM, dense, qwen3.5:9b generator, gemma4:e2b rescore) is in flight on the GPU host. The enabler (arctic-embed ONNX support) merged in PR #160; the MiniLM default is unchanged pending the judged confirm. Provenance: bench host benchmarks/results/e007_{minilm_baseline,arctic_s,arctic_xs}_20260613_122113.json.

Stage 2 result (2026-06-13), judged confirm PASSED: subset-200 with qwen3.5:9b generation and gemma4:e2b rescore, MiniLM 0.6750 vs arctic-embed-s 0.7150, +0.0400 judged. The large retrieval gain (+0.157 R@K) compresses to +0.040 at the answer level, the expected pattern (generation recovers from some retrieval misses, and the lenient judge has a ceiling), but it is real and directionally consistent across both stages. Recorded as F-010. A full-1540 judged run (both arms, same recipe, checkpointed) is in flight on the GPU host to firm the magnitude before any user-facing default change; the published number and the default flip wait on it (and ideally the tri-judge). Provenance: bench host benchmarks/results/e007c_{minilm,arctic_s}_20260613_122948.rescored_gemma.json.

Status: resolved to F-010, CONFIRMED at full-1540 (judged +0.0565). Default-change-worthy; ship-the-win PR (low-tier dense default MiniLM -> arctic-embed-s) is the next action, riding the recipe-aware store seam from PR #161.

**E-008. Surprisal as a forgetting/retention-priority signal under a memory budget.**

The consolidation kill-shot for the v2 surprisal pillar, designed to test a claim today's negatives do NOT touch. N-009 and N-010 tested surprisal at retrieval time over the full corpus (which turns to chunk, skip, or boost when answering). E-008 tests it as the offline retention signal, the core sleep-consolidation decision: when an agent must forget down to a working-set budget B (the archive keeps everything; this is only what stays in the live compilation), is surprisal a better "keep" signal than the cheap baselines?

Design: per conversation, score per-turn surprisal (reuse the E1 probe's cached scores). Apply a retention budget B in {25, 50, 75} percent, keeping only that fraction of turns under four policies: keep-highest-surprisal, keep-most-recent (recency), keep-longest (length, a content proxy), and keep-random (uniform, deterministic by conversation id, the floor). Re-index only the retained turns and measure judge-free R@K: evidence in a dropped turn counts as a miss. Judge-free and CPU-only, so it runs on the VPS while the GPU host stays free. This is distinct from write-skip (N-010): that was a fixed low-surprisal-short threshold firing on 2.3 percent of turns; this is a retention curve where surprisal selects across all turns at several budgets against recency, the baseline it must beat to justify the scorer's cost.

Kill criterion (verbatim): "surprisal-priority retention must beat the best non-random baseline (recency or length) by more than 0.02 R@K at one or more matched budgets and must not trail it at any budget, for the consolidation-strength bet to survive. If surprisal does not clearly beat recency, the surprisal pillar of v2 is dead and the spine pivots to the provenance and claims layer."

Result (2026-06-13), KILLED, recorded as N-011. Judge-free R@K under retention budgets on subset-200, keep-by-policy vs the baselines:

| Budget | surprisal | recency | length | random | surprisal vs best non-random |
|---|---|---|---|---|---|
| 25 percent | 0.313 | 0.232 | 0.525 | 0.258 | -0.212 |
| 50 percent | 0.444 | 0.348 | 0.641 | 0.424 | -0.197 |
| 75 percent | 0.591 | 0.450 | 0.702 | 0.571 | -0.111 |

Quoting the criterion: "surprisal-priority retention must beat the best non-random baseline by more than 0.02 at >=1 budget and must not trail it at any budget. If surprisal does not clearly beat recency, the surprisal pillar of v2 is dead and the spine pivots to the provenance and claims layer." Surprisal not only misses, it is the WORST signal at every budget, below random: keeping the highest-surprisal turns is actively bad for what to retain. The surprisal pillar is dead across both its retrieval role (N-009, N-010) and now its consolidation role. Two findings come out of this: (1) the v2 spine pivots to the provenance and claims layer (anchored by F-009, the 18.8 percent extraction-hallucination rate, which no competitor measures); (2) length is a strong retention signal (best at every budget, +0.11 to +0.18 over random), worth its own pre-registration as a cheap forgetting heuristic for memory-budgeted agents. Provenance: VPS benchmarks/results/e008_retention_20260613_115258.json, branch bench/e1-retention.

Status: resolved to N-011 (surprisal retention killed); v2 spine pivots to provenance/claims.

**E-009. Does the claims gate improve answers without tanking recall?**

The validation gate for the v2 claims layer (Provable Memory), the additive default-off layer that verifies extracted claims against their archive spans (cross-family local entailment, async) and demotes unverified/unsupported claims from recall. This experiment decides whether the gate ships default-on.

Design: on a LoCoMo slice, ingest with claims extracted and a verify-pass run (real cross-family local judge), then answer the QAs three ways, gate off (baseline), prefer_verified (demote unsupported, prefer supported), and strict (also exclude unverified), judged externally. Report, per mode: judged accuracy, R@K (to confirm demotion does not discard needed evidence), and the served-hallucination rate (share of answers citing an unsupported claim). Harness: benchmarks/claims_gate_probe.py (to build), checkpointable.

Kill criterion (verbatim): "the claims gate ships default-on only if it reduces the served-hallucination rate by a meaningful margin AND does not drop judged accuracy or R@K by more than 0.02. If it trades accuracy for purity, it ships default-OFF as an opt-in integrity mode with its measured trade documented; it is never silently enabled."

Status (2026-06-14): the claims layer is MERGED to master (PR #163, taosmd/claims/, default-off). The harness benchmarks/claims_gate_probe.py is built and code-reviewed (it drives the shipped ClaimStore, claims_from_text provenance, LocalEntailmentVerifier, verify_pass, and pure apply_claims_gate; two faithfulness fixes applied, micro-vs-macro R@K matched to the runner, and the gate sees the production-shaped top-k input). The offline self-test passes; a smoke run validated the path end to end (one conversation, 168 claims, all verified, live extraction-hallucination 32.7 percent on that slice). The full LoCoMo sweep (10 conversations, off vs prefer_verified vs strict, qwen3:4b generator, unsloth Qwen3-4B-Instruct-2507 verifier, llama3.1:8b judge) is running on the Fedora GPU host; its verdict, recorded verbatim against the kill criterion, decides the flag. Branch bench/e009-claims-gate-probe. Harness note (2026-06-14): a first launch using qwen3:14b as judge hung mid-run because the qwen3:4b generator (2.9 GB) and qwen3:14b judge (9.3 GB) together exceed the 3060's 12 GB, so Ollama swapped models on every generate/judge alternation and a request eventually deadlocked (no checkpoint, no output). It was relaunched with llama3.1:8b as judge: that model (5.0 GB) coexists with the generator and the verifier all resident (about 10.8 GB total, no swapping). llama3.1:8b is the strict cross-family judge already used in the tri-judge protocol, and the gate comparison is internally consistent (the same judge scores all three modes), so the change is an efficiency fix, not a methodology shift.

Resolved (2026-06-14), recorded as finding F-011: the prefer_verified gate PASSES its pre-registered criterion; strict fails. 200 QAs across 10 LoCoMo conversations, 1853 claims extracted and verified, single external judge (llama3.1:8b), gate applied to the production-shaped top-k input.

| Mode | Judged accuracy | R@K (recall) | Served-hallucination rate |
|---|---|---|---|
| off (baseline) | 0.425 | 0.615 | 0.040 |
| prefer_verified | 0.445 | 0.610 | 0.000 |
| strict | 0.360 | 0.550 | 0.000 |

Quoting the criterion: "the claims gate ships default-on only if it reduces the served-hallucination rate by a meaningful margin AND does not drop judged accuracy or R@K by more than 0.02." prefer_verified satisfies every clause: served-hallucination falls 0.040 to 0.000 (every exposed answer removed, 8 of 200 at baseline), judged accuracy does NOT drop (it is +0.020, off 0.425 to 0.445), and R@K falls only 0.005, inside the 0.02 bound. strict fails both no-drop clauses: it discards unverified-backed evidence, dropping judged accuracy by 0.065 and R@K by 0.065, the "trades accuracy for purity" case the criterion names, so it stays an opt-in integrity mode.

Honest reading, because the absolute deltas are small at n=200 under a single judge: the robust, defensible claim is that prefer_verified ELIMINATES served-hallucination exposure at NO measured accuracy or recall cost. The +0.020 judged "improvement" is four questions and is within noise; it should be read as "no cost," not "a gain." The served-hallucination elimination is the real effect and is mechanical (the gate drops exactly the unsupported-backed hits), and it costs nothing on the headline answer quality, which is the result the claims layer was built to demonstrate: provable memory can refuse to serve unsupported facts without hurting answers.

Per the criterion prefer_verified qualifies to ship default-on, but per the same criterion's "it is never silently enabled" and the project ship-the-win discipline, the default flip is NOT automatic: it needs Jay sign-off and a confirming run at larger n under the full tri-judge (the arctic two-stage pattern) before the shipped default changes. Until then the layer stays default-off with this measured, validated trade documented. Provenance: bench host benchmarks/results/e009.json, harness benchmarks/claims_gate_probe.py, branch bench/e009-claims-gate-probe.

Status: resolved to F-011 (prefer_verified validated, strict opt-in). Default flip pending Jay sign-off + a tri-judge larger-n confirm.

**E-010. Embedding-model bake-off for the low-tier dense default.**

Hypothesis: the MiniLM to arctic-embed-s switch was +0.0565 judged (F-010), but the small-embedder field has strong newer entrants. A judge-free R@K bake-off across the current small, fast, ONNX-deployable models may find a better low-tier dense default than arctic-embed-s, or confirm arctic is already near the small-model ceiling. The whole class is small and CPU-fast, so screening all of them is cheap.

Methodology gate (same lesson as E-007, this invalidates the run if skipped): every candidate must be embedded with its correct pooling (CLS vs mean vs last-token) and its correct query/passage prefix (asymmetric where required), or its score is a false negative. The drop-in subset that the current loader already handles (CLS + the arctic query prefix; mean + no prefix) runs first with zero code; E5-style two-sided query:/passage: prefixes, Nomic search_query:/search_document:, the arctic-v2 query: prefix (NOT the v1 string), and EmbeddingGemma's structured prompts + Dense layers each need loader handling before those models count. CLS + no-prefix (granite, gte-modernbert) needs the loader to treat pooling and prefix as independent flags.

Candidate shortlist (ONNX-ready, permissive, <=384 dim native or MRL-truncatable, ranked by prior MTEB retrieval), all screened on subset-200 judge-free R@K at a matched harness with only the embedder swapped: granite-embedding-small-english-r2 (47M, 384, CLS, no prefix, Apache), gte-modernbert-base (149M, 768 truncate-to-384, CLS, no prefix, Apache), arctic-embed-m-v1.5 (768 to 256 MRL, CLS, arctic prefix, drop-in), arctic-embed-xs (23M, 384, drop-in), bge-small-en-v1.5 (33M, 384, CLS, MIT, drop-in prefix), granite-embedding-english-r2 (149M, 768, CLS, no prefix, Apache), e5-small-v2 (33M, 384, mean, query:/passage:, MIT), nomic modernbert-embed-base (149M, 768 to 256, mean, search_query:/document:, Apache), gte-small (33M, 384, mean, no prefix, MIT), mxbai-embed-xsmall-v1 (24M, 384, mean, Apache), and two static extremes for the speed-vs-accuracy floor, static-retrieval-mrl-en-v1 and potion-retrieval-32M (ONNX feasibility is a pre-flight gate for these, they are not transformer graphs). arctic-embed-s is the incumbent baseline arm. Excluded and why: Jina v3+ and EmbeddingGemma carry non-Apache or Gemma licenses (license blocker for shipped defaults); gte-tiny has no declared license; 0.5B-7B LLM-embedders are outside the CPU/SBC envelope.

Kill criterion (verbatim): "a new embedder replaces arctic-embed-s as the low-tier dense default only if it beats arctic-embed-s R@K by more than 0.02 on subset-200 at the matched harness, with its correct pooling and prefix verified for that model (else the run is invalid), AND is permissively licensed (Apache-2.0 or MIT) AND deployable as ONNX at 384 dimensions or fewer (native or MRL-truncated). A winner is then confirmed with a judged full-1540 run (tri-judge) before any recipe or default change. A tie or loss keeps arctic-embed-s. At an R@K tie within 0.02, the smaller and faster model wins (footprint tiebreak)."

Result (2026-06-14), recorded as negative result N-012: no candidate beats arctic-embed-s; it is retained. Judge-free R@K on LoCoMo (conv=10) on the VPS, each embedder run with its correct pooling + prefix via the new env-configurable loader:

| Embedder | pooling / prefix | R@K |
|---|---|---|
| **arctic-embed-s (shipped default)** | CLS / query-prefix | **0.823** |
| e5-small-v2 | mean / two-sided query:+passage: | 0.808 |
| bge-small-en-v1.5 | CLS / query-prefix | 0.773 |
| granite-embedding-small-english-r2 | CLS / no prefix | 0.742 |
| gte-small | mean / no prefix | 0.727 |

Quoting the kill criterion: "a new embedder replaces arctic-embed-s only if it beats arctic-s R@K by more than 0.02." None does: the closest is e5-small-v2 at -0.015 (below, within noise), and granite-small-r2, the highest-MTEB candidate, lands at -0.081 (its MTEB ranking did not transfer to this LoCoMo retrieval). So arctic-embed-s is retained as the low-tier dense default and F-010 is reaffirmed. The loader build (env-configurable pooling + prefix, TAOSMD_ONNX_POOLING / _QUERY_PREFIX / _DOC_PREFIX) and the bake-off runner (benchmarks/e010_bakeoff.sh) are reusable for future candidates. Methods lesson: onnx-community models store weights in a separate model.onnx_data file; the fetch must include it or the probe loads a weightless graph. Provenance: VPS benchmarks/results/e010_*_20260614_154450.json, branch bench/e010-embedding-bakeoff.

Status: resolved to N-012 (arctic-embed-s retained, no challenger). The 768-dim / upper-tier candidates (granite-r2-english, gte-modernbert) remain an optional future probe if a footprint increase is ever justified, but the small-tier verdict is clear.

**E-011. Does arctic-embed-s move the LongMemEval-S headline?**

Hypothesis: arctic-embed-s beat MiniLM on LoCoMo retrieval (+0.0565 judged full-1540, F-010), but the 97.0 percent LongMemEval-S headline (F-006) was measured on MiniLM and arctic has never been run on it. Swapping the dense embedder MiniLM to arctic-embed-s on the LongMemEval-S harness measures whether the flagship end-to-end number moves. This is the honest follow-up that closes the gap between "better on LoCoMo retrieval" and "better on the headline."

Design: benchmarks/longmemeval_runner.py on the full 500-question LongMemEval-S, two arms identical except the embedder (MiniLM vs arctic-embed-s, both 384-dim ONNX, arctic with its query prefix and CLS pooling), same generator and recipe otherwise, same external judge as F-006. Report end-to-end judge accuracy for each arm and the delta.

Kill criterion (verbatim): "if arctic-embed-s improves LongMemEval-S end-to-end judge accuracy outside noise (delta > +0.005), the 97.0 percent headline is re-measured and the README/headline updated to the arctic stack with provenance. If it is within noise (delta in -0.005..+0.005), arctic is recorded as neutral on LongMemEval-S and MiniLM stays the documented LongMemEval default, with the honest note that arctic's win is LoCoMo-retrieval-specific. If it regresses (delta < -0.005), MiniLM stays the LongMemEval default and the regression is documented; the LoCoMo-vs-LongMemEval divergence is itself a finding. The number is recorded verbatim whatever it is."

Status: pre-registered, queued. REFRAMED 2026-06-14: the "97.0% headline" E-011 targeted is Recall@5, not Judge (see F-001 correction, rev 1.19). E-011 is therefore now "does arctic-embed-s improve LongMemEval-S Recall@5" (a judge-free retrieval question, cheap, CPU-tractable), and the separate end-to-end Judge question moves to E-012.

**E-012. A genuine end-to-end Judge number on LongMemEval-S with the current stack.**

Motivation: the 97.0% headline is Recall@5 (retrieval); the report has never had a real end-to-end Judge number on LongMemEval-S (retrieve, generate an answer, grade it against the reference). The April Judge harness is weak (a 3B generator used as both generator and its own judge, no thinking-suppression) and reproduces about 22.6% with a different per-category shape, which is not a fair test of the system. This experiment establishes an honest Judge number.

Design: a corrected Judge harness (separate EXTERNAL judge, not the generator; thinking-suppression for thinking generators; a real generator, qwen3.5:9b, with the arctic-embed-s embedder and the hybrid + query-expansion retrieval that produces the 97.0% Recall@5). Run the full 500 on the oracle set (headline-comparable, evidence sessions present) and, separately, on the full-distractor set (the harder real test). Validate the harness on a 50-question slice before the full run. Report per-category and overall Judge accuracy, with provenance.

Kill criterion / reporting rule (verbatim): "the end-to-end Judge number is recorded verbatim whatever it is, on both the oracle and full sets, with its generator, judge, retrieval config, dataset, and commit pinned. It is published as a SEPARATE number from the Recall@5 headline and is never conflated with it. A public end-to-end-Judge claim is made only if it is backed by this measurement."

Baseline result (2026-06-14), recorded verbatim as finding F-012: 47.2% end-to-end Judge on LongMemEval-S oracle (236/500). Generator qwen3.5:9b with `/no_think`, EXTERNAL strict judge Qwen3-4B-Instruct-2507 (non-thinking, cross-size from the generator), MiniLM embedder, basic retrieval. Per-category: single-session-assistant 89.3% (50/56), knowledge-update 75.6% (59/78), single-session-user 71.4% (50/70), single-session-preference 53.3% (16/30), multi-session 28.6% (38/133), temporal-reasoning 17.3% (23/133). Provenance: benchmarks/results/e012_full500.log, branch bench/e012-judge-harness (84bb1f3).

Reading: this is far below the 97.0% Recall@5 exactly as the metric distinction predicts. Recall@5 measures whether retrieval surfaced the evidence (solved, 97%); end-to-end Judge measures whether a local 9B then composes the correct answer and a STRICT judge agrees (hard). The per-category shape is the useful finding: retrieval is not the bottleneck, REASONING is. The system nails single-session and knowledge-update questions (75 to 89%) and craters on multi-hop temporal and multi-session (17 to 29%). 47.2% is also conservative because the judge is strict; a lenient judge (what competitors publish against) scores higher.

The harness fix corrected three real bugs that had made a real generator score ~0%: a same-model judge with no think-suppression (emitted CoT, never cleanly returned CORRECT), a `/nothink` typo, and a 3000-char context cap. Recorded as a methods lesson.

Score-up result (2026-06-14), recorded verbatim as finding F-013: the honest end-to-end Judge rose from the F-012 47.2% baseline to 74.6% (373/500) on the oracle set, same external strict Qwen3-4B-Instruct judge, qwen3.5:9b generator, MiniLM. A pre-registered ablation isolated three levers (1a depth knobs, 1b cross-encoder reranking bge-v2-m3, 2 query decomposition, 3 CoVe-style answer self-verification): first a 9-arm representative n=100 screen (seed 42, all six question types), then a full-500 confirm of the baseline and the winner. Per-arm screen (n=100): starved-anchor 24, tuned-depth baseline 54, +rerank 60, +decompose 54, +self-verify 74, rerank+decompose 57, rerank+self-verify 76 (winner), decompose+self-verify 70, all-three 74. Full-500 confirm: tuned-depth baseline 56.8% (284/500), winner rerank+self-verify 74.6% (373/500). Attribution: depth tuning is +9.6pp over the starved F-012 baseline (47.2 to 56.8); answer self-verification is the dominant lever (about +18pp), reranking adds about +6pp and stacks with it; decomposition is a NULL-to-NEGATIVE arm that drags down every combination it joins (recorded as the negative half of this finding, NOT shipped). Self-verification fixes exactly the diagnosed reasoning bottlenecks: at full-500 it lifts temporal-reasoning 30.1 to 56.4%, multi-session 39.8 to 66.9%, single-session-preference 53.3 to 76.7%, while the already-strong categories stay near ceiling. Mechanism: one extra generator pass that checks the draft answer against the retrieved context and rewrites unsupported parts catches the multi-hop drift that is the failure mode in temporal and multi-session questions. Judge: strict Qwen3-4B-Instruct, with the delta CONFIRMED judge-robust by a cross-family llama3.1:8b firm-up that reproduced both numbers exactly (baseline 56.8%, winner 74.6%, identical per-category counts: temperature-0 generation gives byte-identical answers across the two runs, and the two cross-family strict judges agree on every grade). The lenient gemma4:e2b judge does not fit co-resident with the 9B generator (the 12 GB deadlock budget), so a lenient cross-check is deferred to a prediction-cache re-judge. Provenance: benchmarks/results/e012_ablation_results.md (curated), branch bench/e012-judge-harness, runner benchmarks/e012_ablation.sh.

Reading: this is the honest generation-side number and it moved a long way (+27.4pp over the published F-012 baseline) without touching the model, only the evidence pipeline and a self-check. It stays clearly distinct from the 97.0% Recall@5 retrieval headline. Self-verification is a general, model-agnostic lever that any local-tier consumer can apply, which is the shippable insight.

Status: SCORE-UP CONFIRMED (F-013), strict-judge full-500. REMAINING: the llama3.1 cross-family firm-up (running) and, before any default flip, Jay sign-off (per "never silently enabled") plus a product decision on HOW to ship self-verification (taOSmd serves memory; answer generation is the consumer's path, so the likely shape is an opt-in verified-answer mode and reranking enabled in the recommended retrieve config, not a silent core default). The full-distractor-set run remains pre-registered. GPU runs are coordinated with @taOS via the shared-hardware lease protocol.

---

## 7. Revision Log

This log is append-only. History is never rewritten.

| Date | Edition | Changes |
|---|---|---|
| 2026-06-17 | 1.23 | Recorded the BEAM long-context results: new section 3.8, a BEAM dataset entry in 2.2, and index rows F-014 plus N-013. F-014: local-stack numbers under a port of mem0's own nugget judge, BEAM-100K 47.0% (n=100) and BEAM-1M 37.5% (75/200), honestly behind mem0's frontier 64 to 70% at 1M. N-013: a seven-arm BEAM-100K config sweep (chunk size, chunking mode, self-verify, decompose) finds no lever beats the 47% plateau beyond n=100 noise; the transferable sub-finding is that CoVe self-verification, the dominant +17.8pp lever on LongMemEval-S (F-013), gives no BEAM benefit because BEAM failures are wrong answers, not abstentions. Provenance benchmarks/results/beam_campaign_resume_20260616_221030.log plus beam_campaign_20260616_115113.log (branch feat/beam-runner). Exploratory sweep under the standing score-up directive, not pre-registered. |
| 2026-06-11 | 1 | First edition. Sections 0-7 drafted. Index rows F-001 through F-008, N-001 through N-006, E-001 through E-004. All numbers sourced from docs/benchmarks.md, CHANGELOG.md, and STATUS.md. |
| 2026-06-11 | 1.1 | Added N-007 (gemma4:12b generator A/B, confirmed negative) with its index row. E-003 noted as still in flight after a judge-stage client-timeout crash on the harness side; a timeout-patched judge-only rerun is queued. |
| 2026-06-12 | 1.3 | E-003 (LoCoMo-Refined full run) resolved as a methods limitation: the official qwen3:14b judge could not be made to emit reliable JSON under local Ollama across five mitigations, with failures concentrated on temporal questions, so no number is reported (a temporally-biased lower bound would mislead). The qwen3:14b column on the original 1540 set (F-005) is the closest comparable. |
| 2026-06-11 | 1.2 | E-002 resolved to finding F-009 (extraction-hallucination rate 18.8 percent, cross-family). E-001 recorded honestly: prior flat, chunking result confounded by an unenforced size cap and per-chunk evidence credit, corrected re-run in flight, unresolved. E-004 marked inconclusive after the pylate projected path scored 0.050 vs 0.760 backbone, read as a loader fault. |
| 2026-06-12 | 1.4 | Pre-registered E-005 (temporal date-range lever, ported from the engram project) with a two-stage design: applicability scan first, then R@K on the applicable subset only if the lever can fire often enough to matter. Kill criterion written before any run. Index row added. |
| 2026-06-12 | 1.5 | E-005 Stage 1 resolved by its own gate: temporal-category applicability 9.3 percent, under the pre-registered 10 percent floor, so LoCoMo is recorded as unable to measure the lever and no LoCoMo claim is made. Scan script committed for reproducibility. Index status flipped. |
| 2026-06-12 | 1.6 | Pre-registered E-006 (write-skip floor) with its kill criterion, before the queued VPS run produces any number. Index row added. |
| 2026-06-12 | 1.7 | E-004 root cause found: the pylate loader is exonerated (direct load test shows the trained 96-dim Stanford projection applied correctly); the 0.050 and 0.0 results came from the search gate bug plus a pre-fix checkout on the bench host. The earlier "96 vs 128" blocker note was a misdiagnosis. Full projected-space comparison re-running. Index status updated. |
| 2026-06-12 | 1.8 | E-004 resolved to N-008: both ColBERT projected spaces 0.730 vs answerai backbone 0.760 on subset-200 gemma; decision rule quoted, no recipe upgrade; the 4x token-matrix footprint trade of the 96-dim space recorded. benchmarks.md late-interaction section updated with the projected-space table. |
| 2026-06-12 | 1.9 | E-001 resolved to N-009: the matched-turn-budget control shows surprise-boundary chunking was a coverage artifact (baseline at k=120 beats chunks at k=20 by 0.15); kill criterion quoted; both surprisal retrieval arms now dead. Methods lesson recorded: compare chunking levers at matched turn budget, never matched k. |
| 2026-06-14 | 1.21 | E-010 embedding bake-off RESOLVED to N-012: no small ONNX embedder beats arctic-embed-s on judge-free LoCoMo R@K (arctic 0.823 vs e5-small 0.808 / bge-small 0.773 / granite-r2 0.742 / gte-small 0.727); none clears the +0.02 kill margin, arctic retained, F-010 reaffirmed. Shipped a reusable env-configurable onnx loader (pooling + prefix independent) + bake-off runner; ran on the VPS (CPU). Methods lesson: onnx-community models need the separate model.onnx_data weights file fetched. |
| 2026-06-14 | 1.20 | E-012 baseline recorded as F-012: 47.2% end-to-end Judge on LongMemEval-S oracle (236/500), qwen3.5:9b generator + external strict Qwen3-4B-Instruct judge + MiniLM. The honest Judge counterpart to the 97.0% Recall@5; far lower exactly as the metric distinction predicts. Per-category shows retrieval is solved and REASONING is the bottleneck (temporal 17.3%, multi-session 28.6%; single-session/knowledge 71-89%). Harness fix corrected three bugs (same-model judge with no think-suppression, /nothink typo, 3000-char context cap). Improvement phase started per Jay: evidence-depth knobs made tunable (commit a23ea26) since the baseline starved the generator; richer-evidence + arctic + query-expansion + lenient/tri-judge + full-distractor sweep to follow, each measured vs 47.2%. |
| 2026-06-14 | 1.22 | E-012 improvement phase resolved to F-013: the end-to-end Judge rose from 47.2% (F-012, starved) to 74.6% (373/500 oracle, same strict Qwen3-4B-Instruct judge) via depth tuning (+9.6), reranking (+6), and CoVe-style answer self-verification (dominant, about +18); rerank+self-verify is the winner. Method: pre-registered 9-arm ablation, representative n=100 screen (seed 42, all six types, fixing an earlier type-ordered head-slice bias) then a full-500 confirm of baseline + winner. Self-verification fixes the diagnosed bottlenecks (temporal 30->56%, multi-session 40->67%). Query decomposition is the negative half of F-013: null-to-negative, drags every combination down, NOT shipped. Strict-judge full-500; a cross-family llama3.1:8b firm-up of the delta is running (gemma4:e2b lenient does not fit co-resident with the 9B generator). Provenance benchmarks/results/e012_confirm.log + e012_screen2.log. Ship pending Jay sign-off + a product decision on the shape of self-verification. |
| 2026-06-14 | 1.19 | INTEGRITY CORRECTION. F-001 (the LongMemEval-S 97.0% headline) was mislabelled "end-to-end Judge accuracy" across the report, README, benchmarks.md, AGENTS.md, and CHANGELOG. The source result file (benchmarks/results/enhanced_20260413_133215.json, query_expand, top_k=5) shows it is and always was Recall@5. The mislabel was introduced when the public docs were reframed in April; no end-to-end Judge run ever produced 97.0% (the Judge harness reproduces about 22.6% with its default generator, with a different per-category shape, which surfaced the error). Corrected every surface to "97.0% Recall@5" with provenance. Pre-registered E-012 to measure a genuine end-to-end Judge number with the current stack. Recall@5 still leads MemPalace 96.6% head-to-head, so the corrected claim remains strong. |
| 2026-06-14 | 1.18 | E-009 RESOLVED to F-011: the prefer_verified claims gate PASSES its kill criterion (served-hallucination 0.040 to 0.000 eliminated, judged accuracy +0.020 i.e. no drop, R@K -0.005 within the 0.02 bound) on 200 LoCoMo QAs / 1853 claims under a single llama3.1:8b judge; strict fails (judge -0.065, R@K -0.065, the trades-accuracy-for-purity case) and stays opt-in. Honest read: the robust claim is "eliminates served-hallucination at no accuracy cost," the +0.02 is within noise at n=200. Per "never silently enabled" + ship-the-win, the default flip waits on Jay sign-off + a tri-judge larger-n confirm. First launch with qwen3:14b judge hung on a 12GB VRAM model-swap deadlock; relaunched with llama3.1:8b co-resident (efficiency fix, comparison internally consistent). |
| 2026-06-14 | 1.17 | Claims layer MERGED (PR #163) and arctic-embed-s SHIPPED as the low-tier dense default (PR #162, F-010). E-009 status updated: harness built, code-reviewed, smoke-validated, full LoCoMo sweep running on the Fedora host. Pre-registered two new experiments with kill criteria before any run: E-010 (embedding-model bake-off, judge-free R@K screen of the small/fast/ONNX field against arctic-embed-s, footprint tiebreak, license + ONNX gates) and E-011 (does arctic-embed-s move the 97.0 percent LongMemEval-S headline, the honest LoCoMo-to-flagship follow-up). Both queued. |
| 2026-06-13 | 1.16 | Pre-registered E-009, the validation gate for the v2 claims layer (Provable Memory): does verified-preferred recall cut the served-hallucination rate without dropping judged accuracy or R@K? Kill criterion written before any run; the default-off claims layer code is on branch feat/claims-layer. |
| 2026-06-13 | 1.15 | E-007 (F-010) confirmed at full-1540: arctic-embed-s 0.7305 vs MiniLM 0.6740 judged, +0.0565, larger than the subset-200 delta. Default-change-worthy; ship-the-win PR is the next action. |
| 2026-06-13 | 1.14 | E-008 resolved to N-011: surprisal is the worst retention-priority signal (below random at every budget); length wins. The v2 surprisal pillar is conclusively dead across retrieval (N-009/N-010) and consolidation, so the spine pivots to the provenance/claims layer (F-009) per the pre-registered criterion. Length flagged as a positive side-finding for future pre-registration. |
| 2026-06-13 | 1.13 | E-007 resolved to F-010 on subset-200: arctic-embed-s beats MiniLM as the low-tier dense embedder, +0.157 R@K and +0.040 judged at the same dimension and latency, both stages passed. Added results subsection 3.7. Full-1540 judged confirm in flight before the user-facing default is switched. First positive finding after N-008/N-009/N-010. |
| 2026-06-13 | 1.12 | Pre-registered E-008, the v2 consolidation kill-shot: surprisal as a forgetting/retention-priority signal under a memory budget, judge-free R@K vs recency/length/random across budgets. Designed explicitly to test the consolidation claim that N-009/N-010 (retrieval-time) do not, with a kill criterion that pivots v2 to the provenance/claims layer if surprisal does not beat recency. |
| 2026-06-13 | 1.11 | Pre-registered E-007 (arctic-embed s/xs vs all-MiniLM as the low-tier dense embedder) with its kill criterion before any run. Recorded the validity gate: arctic-embed needs a query-only prefix and CLS pooling, implemented in feat/arctic-embed-onnx first, or the comparison is a false negative. |
| 2026-06-12 | 1.10 | E-006 resolved to N-010: the write-skip floor is safe (R@K within noise, zero evidence turns skipped) but fires on only 2.3 percent of turns, under the pre-registered 5 percent shipping floor; not shipped, criterion quoted. With E-001, E-004, and E-006 all resolved today, no pre-registered experiment remains open. |
