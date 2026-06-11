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
| N-001 | CLAG cluster pre-filter: worst variant -0.285 gemma, best variant -0.085 gemma; not shipped | [4](#4-negative-results) | confirmed negative | docs/benchmarks.md section "CLAG cluster pre-filter negative at our tier, not shipped" |
| N-002 | Chain-of-Memory: -0.16 single-hop for +0.02 overall at 1.8x latency (frontier-tuned, lossy at our tier) | [4](#4-negative-results) | confirmed negative | project memory (reference_memory_landscape_may2026.md), Chain-of-Memory TESTED May 31 |
| N-003 | Few-shot prompting: -0.017 vs leader (full-stack + 5 exemplars = 0.540 vs 0.557 leader) | [4](#4-negative-results) | confirmed negative | docs/benchmarks.md section "Negative results levers we tested that regressed at 9B + adj=2" |
| N-004 | HyDE: -0.060 standalone, -0.071 vs leader when stacked; poison-pill across every stack it joins | [4](#4-negative-results) | confirmed negative | docs/benchmarks.md section "Negative results levers we tested that regressed at 9B + adj=2" |
| N-005 | answerai + bge-v2-m3 reranker stacked: 0.720 vs 0.760 answerai alone (subset-200 gemma) | [4](#4-negative-results) | confirmed negative | docs/benchmarks.md "Late-interaction retrieval", confirmed negative sentence |
| N-006 | gemma4:12b first A/B: generation errors stored as zero-scoring predictions; methods lesson, run was invalid | [4](#4-negative-results) | retracted (methods failure, not result) | CHANGELOG.md Unreleased Fixed, "Generation failures in the LoCoMo runner were stored as zero-scoring [generation_error: ...]" |
| E-001 | Surprisal retrieval prior + surprise-boundary chunking; kill criterion: no variant beats baseline R@K by more than 0.02 on subset-200 | [6](#6-ongoing-work-pre-registered) | pre-registered | STATUS.md "bench/e1-surprisal finishing" |
| E-002 | Extraction-hallucination rate: gemma extracts, qwen judges; reporting threshold 3-5% PARTIAL+UNSUPPORTED | [6](#6-ongoing-work-pre-registered) | pre-registered | STATUS.md "bench/e2-claim-verification" |
| E-003 | LoCoMo-Refined full run: 1382 revised questions, official qwen3:14b judge, leader recipe | [6](#6-ongoing-work-pre-registered) | in flight | STATUS.md "all 1382 predictions matched" |
| E-004 | pylate projected-space vs backbone comparison: answerai, colbertv2, LateOn, ColBERT-Zero | [6](#6-ongoing-work-pre-registered) | pre-registered | STATUS.md "feat/pylate-loader pushed, 221 tests green" |

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
- **LoCoMo-Refined (subset-comparable).** The revised 1382-question set from the LoCoMo-Refined leaderboard, using the corrected answer key and the official qwen3:14b judge. A full run is in flight at time of writing (see E-003). Our current qwen3:14b column is on the original 1540-question set; it is judge-comparable to the Refined leaderboard but not set-comparable.

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

Caveat: these scores are on the original 1540-question set. The LoCoMo-Refined board uses the revised 1382-question set with a corrected answer key. These numbers are judge-comparable to competitors but not set-comparable. A full LoCoMo-Refined run is in flight (E-003).

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

Status: branch bench/e1-surprisal is finishing as of 2026-06-11.

**E-002. Extraction-hallucination rate via cross-family claim verification.**

Hypothesis: the extraction step (where facts are pulled from turns and stored as memories) may introduce hallucinated or unsupported claims that compound over time. Measuring this rate would tell us whether extraction quality is a ceiling for the overall pipeline.

Design: gemma4:e2b extracts claims from stored memories; qwen3:4b-instruct judges whether each claim is SUPPORTED, PARTIAL, or UNSUPPORTED against the source turns.

Reporting threshold (verbatim): "a PARTIAL plus UNSUPPORTED rate above 3 to 5 percent is a finding in itself."

Status: branch bench/e2-claim-verification is pushed, 41 tests green, as of 2026-06-11.

**E-003. LoCoMo-Refined full run.**

In flight at time of writing. All 1382 predictions have been matched. The run uses the leader recipe with qwen3.5:9b on the revised 1382-question set and the official qwen3:14b judge pointed at local Ollama.

The result will be recorded verbatim whatever it is. For context, the LoCoMo-Refined published competitor board under qwen3:14b: Mem0 48.91, EverMemOS 58.25, MemPalace 58.68, MemOS 63.60, MemoraX 82.65.

Note: the CC BY-NC license on the LoCoMo-Refined dataset means we publish scores but do not vendor the dataset into this repo.

**E-004. pylate projected-space vs backbone comparison.**

The current answerai numbers (F-003, F-004, F-007) use the model's 384-dim backbone token output. sentence-transformers does not apply projection heads on the token path, so these are not the trained ColBERT projection space. Branch feat/pylate-loader (221 tests green, pushed) loads ColBERT models through pylate so projection heads apply; it auto-detects output dimension per instance.

Decision rule (verbatim): "the recipe model upgrades only if projected beats backbone outside noise on subset-200."

Models to compare: answerai-colbert-small-v1, colbertv2.0 (needs pylate-style loader; plain sentence-transformers drops its linear.weight projection head), LateOn, ColBERT-Zero.

Status: ready for the next GPU window.

---

## 7. Revision Log

This log is append-only. History is never rewritten.

| Date | Edition | Changes |
|---|---|---|
| 2026-06-11 | 1 | First edition. Sections 0-7 drafted. Index rows F-001 through F-008, N-001 through N-006, E-001 through E-004. All numbers sourced from docs/benchmarks.md, CHANGELOG.md, and STATUS.md. |
