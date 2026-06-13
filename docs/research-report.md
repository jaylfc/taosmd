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
  - [3.5 LongMemEval-S end-to-end](#35-longmemeval-s-end-to-end)
  - [3.6 Shipped recipe: lateint-9b](#36-shipped-recipe-lateint-9b)
  - [3.7 Low-tier dense embedder: arctic-embed vs MiniLM](#37-low-tier-dense-embedder-arctic-embed-vs-minilm)
- [4. Negative Results](#4-negative-results)
- [5. Reproducibility](#5-reproducibility)
- [6. Ongoing Work (Pre-registered)](#6-ongoing-work-pre-registered)
- [7. Revision Log](#7-revision-log)

### Finding Index

Every row is stable. IDs are never reused. E-ids carry over when an experiment moves to Results (gaining an F row) or Negative Results (gaining an N row).

| ID | One-line summary | Section | Status | Provenance |
|---|---|---|---|---|
| F-001 | LongMemEval-S: 97.0% end-to-end Judge on full 500-question test set | [3.5](#35-longmemeval-s-end-to-end) | confirmed | docs/benchmarks.md section "LongMemEval-S 97.0% end-to-end Judge" |
| F-002 | LoCoMo full-1540 leader (MaxSim + rerank): 0.748 lenient / 0.394 strict-llama / 0.659 strict-instruct | [3.1](#31-locomo-leader-table-full-1540-tri-judge) | confirmed | docs/benchmarks.md "Full-1540 leader (tri-judge, Jun 2026)" |
| F-003 | Late-interaction (answerai backbone, no reranker): 0.716 / 0.388 / 0.656 full-1540 tri-judge | [3.2](#32-late-interaction-retrieval-reranker-free-tri-judge) | confirmed | docs/benchmarks.md "Late-interaction retrieval (token-level MaxSim at retrieval time, tri-judge)" |
| F-004 | CPU-tier R@K: dense 0.641 at 72ms p50 to answerai 0.854 at 110ms p50 on 16-core CPU VPS | [3.4](#34-cpu-tier-retrieval-only-latency-probe) | confirmed | docs/benchmarks.md "CPU-tier viability (retrieval-only, judge-free)" |
| F-005 | qwen3:14b community judge on late-int files: dense 0.487 / MiniLM MaxSim 0.532 / answerai 0.542 | [3.3](#33-community-standard-judge-column-qwen314b) | confirmed | docs/benchmarks.md "Community-standard judge column (qwen3:14b)" |
| F-006 | Librarian vocabulary-gap lever: +15.4% composite on long-horizon sessions (gemma4:e2b, 2026-04-15) | [3.5](#35-longmemeval-s-end-to-end) | confirmed | docs/benchmarks.md section "Librarian layer vocabulary-gap benchmark" |
| F-007 | lateint-9b shipped recipe: answerai + mem0_additive + k=10, retrieval_k=20, adj=2, no reranker | [3.6](#36-shipped-recipe-lateint-9b) | confirmed | CHANGELOG.md Unreleased, "Late-interaction retrieval lever and lateint-9b recipe" |
| F-008 | Same-model judging inflates scores 15-22 pp vs external judge (quantified on qwen3.5:9b self-judge: +0.10 on this stack) | [2.1](#21-external-multi-judge-protocol) | confirmed | docs/benchmarks.md section "Judge sensitivity what we are really measuring" |
| F-009 | Extraction-hallucination rate 18.8 percent (PARTIAL+UNSUPPORTED) over 526 claims, cross-family verified | [3.5](#35-extraction-hallucination-rate-f-009) | confirmed | bench host 20260611 e2 verifier summary |
| F-010 | Arctic-embed-s beats MiniLM as low-tier dense embedder: +0.157 R@K, and judged +0.040 subset-200 / +0.0565 full-1540 (0.7305 vs 0.6740); same 384 dim and latency | [3.7](#37-low-tier-dense-embedder-arctic-embed-vs-minilm) | confirmed (full-1540) | bench host e007full_* 20260613_131837 |
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
| E-001 | Surprisal retrieval: BOTH arms dead. Prior flat; chunking +0.1263 at equal k but baseline at matched turn budget wins by 0.15 | [6](#6-ongoing-work-pre-registered) | resolved -> N-009 | VPS e1_mb_k20/k120_20260612_083159.json |
| E-002 | Extraction-hallucination rate: gemma extracts, qwen judges; reporting threshold 3-5% PARTIAL+UNSUPPORTED | [6](#6-ongoing-work-pre-registered) | resolved -> F-009 | STATUS.md "bench/e2-claim-verification" |
| E-003 | LoCoMo-Refined full run: 1382 revised questions, official qwen3:14b judge, leader recipe | [6](#6-ongoing-work-pre-registered) | methods limitation (judge unreliable on Ollama) | STATUS.md "all 1382 predictions matched" |
| E-004 | pylate projected-space vs backbone: both projected spaces 0.730 vs backbone 0.760 subset-200 gemma; no recipe upgrade | [6](#6-ongoing-work-pre-registered) | resolved -> N-008 | bench host 20260612_142049 rescored_gemma results |
| E-005 | Temporal date-range lever: LoCoMo CANNOT measure it (temporal-cat applicability 9.3 percent, under the 10 percent gate); ships default-off | [6](#6-ongoing-work-pre-registered) | resolved (not measurable on LoCoMo) | benchmarks/temporal_applicability_scan.py, master 0535f86 |
| E-006 | Write-skip floor: SAFE (delta +0.0051, zero evidence skipped) but fires on only 2.3 percent of turns, under the 5 percent shipping floor | [6](#6-ongoing-work-pre-registered) | resolved -> N-010 | VPS e1_write_skip_20260612_143625.json |
| E-007 | Arctic-embed vs MiniLM low-tier dense: CONFIRMED full-1540 (judged +0.0565, 0.7305 vs 0.6740) + R@K +0.157; clear default-change-worthy win | [6](#6-ongoing-work-pre-registered) | resolved -> F-010 (full-1540) | bench host e007full_*_20260613_131837 |
| E-008 | Surprisal as retention-priority signal: KILLED, worst of all policies at every budget (below random); length wins. Surprisal pillar dead, v2 pivots to provenance/claims | [6](#6-ongoing-work-pre-registered) | resolved -> N-011 | VPS e008_retention_20260613_115258.json |

---

## 1. Abstract

taOSmd is a local-first memory system for AI agents. It stores conversation turns in a zero-loss append-only archive, layers hybrid retrieval on top, and runs offline on modest hardware without cloud dependencies.

The current headline result is 97.0% end-to-end Judge accuracy on LongMemEval-S (500 questions, full test set), measured on a 16 GB Orange Pi 5 Plus under a strict local judge. End-to-end Judge means retrieve, generate an answer, and grade it against the reference with an LLM; it is a harder bar than retrieval-only Recall@5.

On LoCoMo (1540 QAs, 10 multi-session conversations), the current leader recipe is MaxSim + reranking: qwen3.5:9b with retrieval-top-k 50, adjacent-turns 2, bge-v2-m3 reranker, and mem0_additive fusion. Full-1540 tri-judge scores: 0.748 lenient (gemma4:e2b), 0.394 strict (llama3.1:8b), 0.659 strict (qwen3:4b-instruct-2507). All judges are external to the generator family.

A reranker-free late-interaction lever (token-level MaxSim via answerai-colbert-small-v1, backbone path) scores 0.716 / 0.388 / 0.656 on the same full-1540 tri-judge and runs in under 110ms per query on a CPU-only 16-core VPS, making it the recommended recipe for tiers where a cross-encoder download is not affordable. All numbers are under external judges with no same-family inflation.

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

### 2.3 Metrics

**End-to-end Judge accuracy** is the primary accuracy metric: what fraction of generated answers get marked correct by the judge, graded against the gold reference. It combines retrieval, generation, and grading in one number.

**R@K** (Retrieval at K) is used for retrieval-only levers where there is no LLM generation: what share of questions have their annotated evidence passage in the top-K retrieved results. It is judge-free and directly reflects retrieval quality.

**Latency** is reported as p50 and p95 per-query wall-clock in milliseconds. **Context tokens** are estimated from context chars / 4 in the LoCoMo runner and carried in the result JSON per row.

Per the three-number reporting rule: every accuracy claim in this report is accompanied by latency and context cost where those have been measured.

### 2.4 Hardware tiers

- **Orange Pi 5 Plus, 16 GB RAM (RK3588 NPU).** Reference low-end tier. LongMemEval-S 97.0% was measured here. Generator: qwen3-4B via rkllama on the NPU. Embedder: all-MiniLM-L6-v2 ONNX on CPU. Source: docs/benchmarks.md hardware tiers section.
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

### 3.5 LongMemEval-S end-to-end

**F-001.** 97.0% end-to-end Judge accuracy on LongMemEval-S, 500 questions, standard test set. Harness: benchmarks/longmemeval_runner.py. Measured on the Orange Pi 5 Plus reference stack (generator: qwen3-4B via rkllama on RK3588 NPU; embedder: all-MiniLM-L6-v2 ONNX on CPU). Strategy: hybrid + query expansion.

Source: docs/benchmarks.md, LongMemEval-S 97.0% end-to-end Judge section.

| Category | hybrid + expand | raw semantic |
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

---

## 7. Revision Log

This log is append-only. History is never rewritten.

| Date | Edition | Changes |
|---|---|---|
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
| 2026-06-13 | 1.15 | E-007 (F-010) confirmed at full-1540: arctic-embed-s 0.7305 vs MiniLM 0.6740 judged, +0.0565, larger than the subset-200 delta. Default-change-worthy; ship-the-win PR is the next action. |
| 2026-06-13 | 1.14 | E-008 resolved to N-011: surprisal is the worst retention-priority signal (below random at every budget); length wins. The v2 surprisal pillar is conclusively dead across retrieval (N-009/N-010) and consolidation, so the spine pivots to the provenance/claims layer (F-009) per the pre-registered criterion. Length flagged as a positive side-finding for future pre-registration. |
| 2026-06-13 | 1.13 | E-007 resolved to F-010 on subset-200: arctic-embed-s beats MiniLM as the low-tier dense embedder, +0.157 R@K and +0.040 judged at the same dimension and latency, both stages passed. Added results subsection 3.7. Full-1540 judged confirm in flight before the user-facing default is switched. First positive finding after N-008/N-009/N-010. |
| 2026-06-13 | 1.12 | Pre-registered E-008, the v2 consolidation kill-shot: surprisal as a forgetting/retention-priority signal under a memory budget, judge-free R@K vs recency/length/random across budgets. Designed explicitly to test the consolidation claim that N-009/N-010 (retrieval-time) do not, with a kill criterion that pivots v2 to the provenance/claims layer if surprisal does not beat recency. |
| 2026-06-13 | 1.11 | Pre-registered E-007 (arctic-embed s/xs vs all-MiniLM as the low-tier dense embedder) with its kill criterion before any run. Recorded the validity gate: arctic-embed needs a query-only prefix and CLS pooling, implemented in feat/arctic-embed-onnx first, or the comparison is a false negative. |
| 2026-06-12 | 1.10 | E-006 resolved to N-010: the write-skip floor is safe (R@K within noise, zero evidence turns skipped) but fires on only 2.3 percent of turns, under the pre-registered 5 percent shipping floor; not shipped, criterion quoted. With E-001, E-004, and E-006 all resolved today, no pre-registered experiment remains open. |
