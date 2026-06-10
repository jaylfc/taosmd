# Benchmarks

## LongMemEval-S — 97.0% end-to-end Judge

500 questions, standard test set, same embedding model (all-MiniLM-L6-v2, 384-dim). End-to-end Judge means the full pipeline runs: retrieve → generate an answer with an LLM → judge the generated answer against the reference answer with an LLM grader.

Harness: [`benchmarks/longmemeval_runner.py`](../benchmarks/longmemeval_runner.py).

### Per-category breakdown

| Category | hybrid + expand | raw semantic |
|----------|----------------|--------------|
| knowledge-update | **100.0%** (78/78) | 100.0% |
| multi-session | **98.5%** (131/133) | 95.5% |
| single-session-user | **97.1%** (68/70) | 90.0% |
| single-session-assistant | **96.4%** (54/56) | 96.4% |
| temporal-reasoning | 94.0% (125/133) | 94.0% |
| single-session-preference | 90.0% (27/30) | 93.3% |
| **Overall** | **97.0%** (485/500) | 95.0% (475/500) |

### Fusion strategy comparison

| Strategy | Judge accuracy | Delta |
|----------|---------------|-------|
| Raw cosine | 95.0% | — |
| Additive keyword boost | 96.6% | +1.6 |
| **Hybrid + query expansion (default)** | **97.0%** | **+2.0** |
| All-turns hybrid (harder test) | 93.2% | -1.8 |

### Other published systems on LongMemEval-S

| System | Score | Metric | Notes |
|--------|-------|--------|-------|
| **taosmd** | **97.0%** | end-to-end Judge | this repo |
| MemPalace | 96.6% | Recall@5 | retrieval-only |
| agentmemory | 95.2% | Recall@5 | retrieval-only |
| SuperMemory | 81.6% | Recall@5 | retrieval-only, cloud embeddings |

The two metrics measure different things:

- **End-to-end Judge** (ours): retrieve → generate an answer → LLM grades the answer against the reference. Scores how often the final answer is correct.
- **Recall@5**: did the correct session appear in the top-5 retrieved? No generation, no grading of the answer. Scores only whether retrieval surfaced the right chunk.

A system can have excellent Recall@5 and still fail end-to-end Judge if the generator can't compose a correct answer from the retrieved evidence. Both reproductions are in this repo — [`benchmarks/longmemeval_runner.py`](../benchmarks/longmemeval_runner.py) for Judge, [`benchmarks/longmemeval_recall.py`](../benchmarks/longmemeval_recall.py) for Recall@5.

## LoCoMo — multi-session conversational memory (1540 QAs)

LoCoMo-10: 10 multi-session conversations, 1540 question-answer pairs across Single-hop / Temporal / Multi-hop / Open-dom categories. Harder dataset than LongMemEval-S — longer conversations (50+ sessions, 400–700 turns), more categories, more QAs. We run it on smaller generators (5B–9B quants on a 12 GB GPU) to measure how the taosmd architecture behaves at the compute tier we target, not at gpt-4o-mini scale.

End-to-end Judge here is graded by an **external** LLM (`qwen3:4b`) — distinct from the generator — with 100 % coverage, 0 errors across all runs. Self-judge numbers (the generator grading its own output) are reported for reference but aren't the headline.

Harness: [`benchmarks/locomo_runner.py`](../benchmarks/locomo_runner.py). Rescore tool: [`benchmarks/locomo_rescore_streaming.py`](../benchmarks/locomo_rescore_streaming.py).

### Full-1540 leader (tri-judge, Jun 2026)

The current leader is **MaxSim + reranking**: `qwen3.5:9b --retrieval-top-k 50 --adjacent-turns 2 --reranker bge-v2-m3 --fusion mem0_additive`, a bge-v2-m3 cross-encoder doing late-interaction (MaxSim) scoring over a wider k=50 candidate pool. Scored on the full 1540 under three judges:

| Recipe (qwen3.5:9b) | Lenient `gemma4:e2b` | Strict `llama3.1:8b` | Strict `qwen3:4b-instruct-2507` |
|---|---|---|---|
| **MaxSim + rerank** | **0.748** | **0.394** | **0.659** |
| RRF (k=20 + llm-exp) | 0.723 | 0.390 | 0.634 |
| mem0_additive (k=20 + llm-exp) | 0.684 | 0.387 | 0.624 |

MaxSim+rerank is first on every judge (clearly under gemma and qwen3-instruct at about +0.025, by a near-tie margin under llama). The MaxSim > RRF > mem0_additive ordering is identical across all three judges. Both RRF and mem0_additive beat the older mem0-only default on every judge, so mem0_additive is no longer the recommended default (the default is tier-gated: MaxSim+rerank where the reranker is affordable, a lighter recipe on constrained tiers).

**Judge methodology change (Jun 2026).** The original external strict judge `qwen3:4b` is a thinking model whose current ollama build no longer emits clean YES/NO verdicts (it preambles, or does not commit within a reasonable token budget), so it is retired as a judge. The strict column now uses two non-thinking judges: `llama3.1:8b` and the non-thinking `qwen3:4b-instruct-2507` (HF GGUF). The lenient judge is unchanged (`gemma4:e2b`). The legacy `qwen3:4b` figures in the leaderboard below (for example RRF 0.557, mem0 0.540) came from an older `qwen3:4b` build and are not directly comparable to the new strict judges; they are kept for history.

### Lighter recipes and a larger generator (tri-judge)

The same three judges, applied to the lighter recipes we ship and to a larger-generator probe. The two full-1540 rows are directly comparable to the leader table above. The 35B row is a 200-QA subset and is **not** comparable to the full-1540 numbers (different sample, roughly ±0.02 noise); it is a generator-size probe, not a leaderboard entry. All five rows use the same `judge_rejudged` rescore as the leader table.

| Recipe | Generator | Scale | Lenient `gemma4:e2b` | Strict `llama3.1:8b` | Strict `qwen3:4b-instruct-2507` |
|---|---|---|---|---|---|
| fast-8b (RRF, k=20 + adj=2, no reranker) | llama3.1:8b | full 1540 | 0.636 | 0.433 | 0.608 |
| lite (no-LLM ingest, boost fusion, k=5 + adj=2) | qwen3.5:9b | full 1540 | 0.607 | 0.353 | 0.568 |
| leader recipe + larger generator | qwen3.6:35b-a3b Q4_K_M | subset 200 | 0.715 | 0.465 | 0.675 |

Reading these honestly:

- **fast-8b trades roughly 0.04 to 0.05 for a 4 GB footprint and about 2.4x throughput.** It stays within 0.05 of the leader on the lenient and qwen-instruct judges, and it actually edges the leader on the strict llama judge (0.433 vs 0.394). That judge is harsh and compresses the field, so an 8B generator can come out ahead inside the noise band. That is not a reason to prefer it on a 12 GB box, but it is a fair reason to run it on a 4 GB one.
- **lite (no-LLM ingest) keeps most of the recall with zero per-turn extraction cost.** It lands about 0.04 to 0.07 below the leader across judges, the price of dropping fact and event enrichment and relying on raw embedding recall. The intended tier is a Pi 4B or any CPU-only box where an LLM call per turn is not affordable.
- **The 35B generator probe is encouraging but unproven at full scale.** On the 200-QA subset it leads every judge, but subset-200 over-claims have bitten us before (the llama+RRF cell collapsed by 0.16 SH when validated at full 1540). A matched full-1540 run is needed before any promotion. See the TurboQuant spec (`docs/specs/2026-04-29-qwen36-turboquant-benchmark-design.md`) for the planned full run.

### Late-interaction retrieval (token-level MaxSim at retrieval time, tri-judge)

Distinct from the leader's cross-encoder rerank: this lever scores retrieval candidates with token-level MaxSim (ColBERT-style late interaction) instead of a single dense cosine, with **no reranker in the loop**. Probed with MiniLM per-token embeddings (`--late-interaction`), then with a proper ColBERT-trained model (`--colbert-model`). Full-1540, same tri-judge rescore as the tables above; the baseline row is the identical config minus late interaction (mem0_additive, k=10, retrieval k=20, adj=2, no expansion, no reranker).

| Config (qwen3.5:9b, no reranker) | Scale | Lenient `gemma4:e2b` | Strict `llama3.1:8b` | Strict `qwen3:4b-instruct-2507` |
|---|---|---|---|---|
| dense baseline | full 1540 | 0.674 | 0.377 | 0.598 |
| + MiniLM MaxSim (`--late-interaction`) | full 1540 | 0.711 | 0.378 | 0.642 |
| + answerai-colbert-small backbone (`--colbert-model`) | full 1540 | 0.716 | 0.388 | 0.656 |

Reading it honestly: the subset-200 read (+0.060 gemma) shrank to **+0.037 gemma / +0.044 qwen-instruct at full scale, and the strict llama judge sees nothing** (+0.001 — that judge compresses the field; the same pattern as fast-8b above). Still a real, two-judge-confirmed win for a lever that costs no reranker model: it lands 0.711 gemma vs the full leader's 0.748 without the bge-v2-m3 download or its per-query latency.

answerai-colbert-small-v1 (33M) is confirmed at full-1540: 0.716 lenient gemma4:e2b, 0.388 strict llama3.1:8b, 0.656 strict qwen3:4b-instruct-2507. It beats MiniLM MaxSim on all three judges, but the subset-200 gap (+0.020 gemma, 0.760 vs 0.740) shrank to +0.005 / +0.010 / +0.014 at full scale: a real but small edge, consistent in sign across judges.

Stacking the bge-v2-m3 cross-encoder reranker on top of answerai retrieval is a confirmed negative: subset-200 gemma 0.720 vs 0.760 for answerai alone. The reranker reorders an already token-level-matched pool and hurts. Recorded so nobody retries it blind.

Known issue: `colbert-ir/colbertv2.0` cannot be loaded via plain sentence-transformers — ST drops its `linear.weight` projection head and every query errors on a dimension mismatch (the run produced zero results); it needs a pylate-style loader before its number means anything. The pylate loader now exists on branch `feat/pylate-loader`; a projected-space vs backbone comparison is queued for the next GPU window.

**CPU-tier viability (retrieval-only, judge-free).** Measured with the
retrieval-latency probe (`benchmarks/retrieval_latency_probe.py`, branch
`bench/retrieval-latency-probe`): subset-200, R@K = share of questions whose
annotated evidence lands in the retrieved set, per-query wall-clock on a
16-core x86 CPU VPS (no GPU), `--reranker off`:

| Retrieval config | R@K | p50 / p95 latency |
|---|---|---|
| dense (MiniLM ONNX) | 0.641 | 72 / 79 ms |
| + MiniLM MaxSim (`--late-interaction`) | 0.813 | 85 / 93 ms |
| + answerai-colbert-small backbone (`--colbert-model`) | 0.854 | 110 / 122 ms |

Late interaction costs 13-38 ms per query on CPU and buys +0.17 to +0.21
evidence recall: the lever is viable on CPU-only tiers, not just the GPU box.
(Note: the answerai row is the model's 384-dim backbone token output, not its
trained ColBERT projection space — sentence-transformers does not apply
projection heads on the token path; a pylate loader is queued to measure the
true projected space.)

### Leaderboard (legacy external `qwen3:4b` judge, same dataset + same prompt)

| System | Generator | Retrieval config | Ext Judge | Notes |
|---|---|---|---|---|
| **taosmd** | qwen3.5:9b | k=20 + adj=2 + llm-exp + RRF | **0.557** | **leader — full stack with RRF fusion** |
| **taosmd** | qwen3.5:9b | k=20 + adj=2 + llm-exp + RRF + multi-level | 0.552 | adding multi-level on top of leader regresses by -0.005 |
| **taosmd** | qwen3.5:9b | k=20 + adj=2 + llm-exp | 0.545 | full stack without RRF |
| **taosmd** | qwen3.5:9b | k=20 + adj=2 + llm-exp + RRF + few-shot | 0.540 | few-shot on top of the leader regresses by -0.017 |
| **taosmd** | qwen3.5:9b | adj=3 | 0.532 | broader context window |
| **taosmd** | qwen3.5:9b | adj=2 + multi-level retrieval | 0.524 | turns + summaries + events |
| **taosmd** | qwen3.5:9b | adj=2 + RRF only | 0.500 | RRF without scaffolding regresses at 9B |
| **taosmd** | qwen3.5:9b | adj=2 + multi-level + RRF | 0.495 | negative interaction without k=20 scaffolding |
| **taosmd** | qwen3.5:9b | adj=2 + BGE-v2-m3 reranker | 0.522 | stronger cross-encoder, marginal lift at this tier |
| **taosmd** | qwen3.5:9b | adj=2 | 0.516 | simplest 9B leader |
| **taosmd** | qwen3.5:9b | k=20 + adj=1 + llm-exp | 0.509 | full stack at adj=1 |
| **taosmd** | gemma4:e2b | adj=2 | 0.499 | 5B best — same architecture, smaller generator |
| **taosmd** | qwen3.5:9b | adj=2 + HyDE + full stack | 0.486 | HyDE drags the leader recipe down — see negative results |
| **taosmd** | qwen3.5:9b | adj=1 | 0.481 | |
| **taosmd** | gemma4:e2b | k=20 + adj=1 + llm-exp | 0.482 | 5B stack |
| **taosmd** | gemma4:e2b | adj=1 (C3) | 0.465 | baseline adjacent-turns win |
| **taosmd** | gemma4:e2b | baseline (prompt-opt) | 0.410 | reference point |
| MemPalace | gemma4:e2b | chromadb + MiniLM | 0.336 | same generator + same dataset, different architecture |
| **taosmd** | qwen3.5:9b | full-context (no retrieval) | 0.090 | retrieval ablation — see section below |
| mem0 | gemma4:e2b | chromadb + nomic-embed, infer=False | 0.060 | same generator + same dataset, different architecture |

All taosmd rows on the same commit series (`feat/locomo-param-configs` merged to master). Every row pins its config, generator, judge, and dataset.

### Key architectural findings

- **`adjacent_turns` is the dominant lever at every model size we measured.** 5B: adj=1 → 0.465, adj=2 → 0.499. 9B: adj=1 → 0.481, adj=2 → 0.516. Going from adj=1 to adj=2 adds more than the entire stack of `k=20 + llm-exp` adds. Available in core via `retrieve(adjacent_neighbors=N, position_key=..., group_key=...)` — see `taosmd/retrieval.py`. Default off; consumers opt in by populating an integer position field on each item's metadata at ingest time.
- **At 9B, individual retrieval levers need the full-stack scaffolding to express their value.** Adding RRF *alone* on top of adj=2 regresses (0.500 vs 0.516); adding multi-level retrieval *alone* on top of adj=2 helps modestly (0.524); combining the two without k=20 + llm-exp scaffolding is *worse* than either alone (0.495). But the same components inside the full stack — k=20 + adj=2 + llm-exp + RRF — give the leader at 0.557. The wider candidate pool from k=20 is what gives the fusion something useful to merge; without it RRF over-smooths a narrow ranked list.
- **Multihop decomposition is a footgun across model sizes.** 5B: 0.317 (-0.093 vs baseline). 9B: **0.306** (worse than 5B). Not a sizing issue — sub-query retrieval inherently surfaces lower-quality chunks. **Don't enable `--multihop-decompose` in production.**
- **Generator size alone is a weak lever.** qwen3.5:9b + k=20 (0.458) is *worse* than gemma4:e2b + adj=1 (0.465). Doubling parameters gives ≤ +0.005 unless architecture scales with it.
- **Stacking is adj-dependent at 5B.** At adj=1 + 5B, adding k=20 compounds cleanly (+0.014). At adj=2 + 5B the same k=20 addition *regresses* (-0.022). Context token budget has a sweet spot at this tier.
- **Date-format swap and LLM query expansion** are marginal in the presence of adj=1.

### Negative results — levers we tested that regressed at 9B + adj=2

Running these as cells against the same external `qwen3:4b` judge gives the lever-effectiveness map below. Useful as guardrails: not every "smarter retrieval" idea helps, and stacking levers indiscriminately makes things worse.

| Cell | Ext Judge | Δ vs 9B+adj=2 baseline (0.516) | Verdict |
|---|---|---|---|
| `kitchen_sink` (adj=2 + k=20 + llm-exp + RRF + multi-level + BGE + temporal-boost + HyDE) | 0.499 | -0.017 | piling levers on regresses |
| `few_shot` (adj=2 + 5 exemplars in answer prompt) | 0.501 | -0.015 | exemplars don't help at 9B |
| `temporal` (recency boost +0.005) | 0.500 | -0.016 | boost has no effect on memory recall |
| `inverse_temporal` (recency boost -0.005) | 0.503 | -0.013 | small regression |
| `hyde` (adj=2 + HyDE only) | 0.456 | -0.060 | **HyDE regresses standalone** |
| `hyde_full_stack` (leader recipe + HyDE) | 0.486 | -0.030 vs baseline / -0.071 vs 0.557 leader | **HyDE poison-pill — drags any stack it joins** |
| `thinking_on` (adj=2 + qwen3 thinking-mode, 200-QA subset) | 0.20 | **-0.32** | **chain-of-thought regresses catastrophically on memory-recall** |

Lesson: HyDE assumes the generator can imagine a good hypothetical answer; on memory-recall datasets a small generator hallucinates the answer's structure, the embedding lookup matches that hallucination, and recall collapses. Few-shot exemplars compete for context budget without adding retrieval signal — and the same effect persists when stacked: `few_shot_full_stack` (leader recipe + 5 exemplars) lands at 0.540, -0.017 vs the 0.557 leader. The same few-shot regression appears at the Pi NPU tier (-0.11 vs the Pi full-stack leader of 0.49) — direction-consistent across model sizes. Thinking-mode (qwen3 chain-of-thought ON, 200-QA subset) collapses to 0.20 — the model emits CoT that drifts off the retrieved context or fails to commit within the budget; **don't enable `--thinking-mode` on memory-recall benchmarks for this generator**. Generator-side bumps (35B-A3B via TurboQuant — `docs/specs/2026-04-29-qwen36-turboquant-benchmark-design.md`) are the next planned lever, since retrieval-side levers have plateaued.

### Binary embedding quantization — recall-neutral, ships as an SBC footprint option

Binary (sign-bit / Hamming) quantization replaces full-precision cosine with a 1-bit-per-dimension score: binarise the query and each stored vector to ±1 and rank by the fraction of matching bits. We A/B'd it against the leader vector-only recipe on the **full 1540-QA** LoCoMo set, dual-judge (`qwen3:4b` strict + `gemma4:e2b` lenient):

| Scoring | qwen3:4b strict | gemma4:e2b lenient |
|---|---|---|
| full-precision cosine (baseline) | 0.520 | 0.694 |
| **binary-quant** | 0.519 (**−0.001**) | 0.699 (**+0.005**) |

Per-category deltas stay inside ±0.04 on both judges (Multi-hop / Temporal / Open / Single-hop) — within run-to-run noise. **Verdict: recall-neutral.** The value isn't quality — it's footprint and speed: **32× smaller vectors** (1 bit vs 32-bit float per dimension) and integer-friendly distance, which matters on memory- and CPU-constrained SBC deployments.

Shipped as an **opt-in, default-off** option — `VectorMemory(binary_quant=True)` (benchmark flag: `--binary-quant`). Standalone behaviour is unchanged unless you turn it on; recommended for the SBC / low-memory tier where the vector-store footprint or CPU distance cost is the binding constraint, not recall.

### CLAG cluster pre-filter — negative at our tier, not shipped

CLAG (cluster-then-retrieve: cosine k-means over the candidate pool, keep only the query's nearest cluster(s) before ranking) is a frontier idea aimed at small-model latency. On the 200-QA subset, every variant regressed:

| Variant | qwen3:4b Δ | gemma4:e2b Δ |
|---|---|---|
| keep nearest 1 cluster | −0.205 | −0.285 |
| keep nearest 2 clusters | −0.085 | −0.090 |
| 1-of-6 clusters | −0.150 | −0.245 |

The keep-sweep is monotonic — relaxing the pruning moves *toward* baseline but never beats it, so cluster pruning only ever discards relevant evidence. Worst on Multi-hop + Temporal, where the evidence for one answer is spread across clusters. Same pattern as HyDE / Chain-of-Memory / thinking-mode: a lever tuned for frontier models that regresses at our compute tier. **Not shipped.** (Benchmark-only `--clag` flag retained for reproduction.)

### Architecture matters more at smaller compute tiers

The same retrieval improvement, applied to two generators on identical hardware-class infrastructure, can lift the smaller-model tier by an order of magnitude more than the larger-model tier:

| Improvement | At qwen3.5:9b (Fedora 12 GB GPU) | At qwen3-4b-chat RKLLM (Orange Pi NPU) | Ratio |
|---|---|---|---|
| **Full leader stack (k=20 + adj=2 + llm-exp + RRF)** | +0.041 (0.516 → 0.557) | **+0.108** (0.382 → 0.490) | **~2.6×** |
| **BGE-reranker-v2-m3 swap** | +0.006 (0.516 → 0.522) | **+0.074** (0.382 → 0.456) | **~12×** |
| **Multi-level retrieval + RRF** | -0.021 (0.516 → 0.495, *regresses*) | +0.043 (0.382 → 0.425) | direction-flip |
| **HyDE** | -0.060 (0.516 → 0.456) | -0.058 (0.382 → 0.325) | direction-consistent regression |

The interpretation: smaller in-weight knowledge means the generator depends more on retrieval quality. A stronger cross-encoder gives the 4 B model on the Pi NPU much more useful candidates than its baseline already retrieves; the 9 B model on the 12 GB GPU was already doing fine on the simpler reranker. This is the architectural argument for why a "memory system designed for SBC-class hardware" is worth building, not just a port of the cloud-tier system at lower precision.

Same dataset (LoCoMo-10), same external `qwen3:4b` judge, same answer prompt across both tiers.

HyDE is an interesting opposite signal: small generators write *worse* hypothetical passages than the question itself contains, so embedding-match against the corpus degrades. HyDE's "imagine the answer" trick assumes a generator strong enough to imagine well — small models don't qualify.

### Retrieval is essential — even at long-context scale

`qwen35_9b_full_context`: feed every turn of the conversation (no retrieval) to qwen3.5:9b's 128K window. LoCoMo conversations fit (~30–60K tokens). Result: **0.090 ext judge** — *worse than mem0 with retrieval* (0.060 floor). Self-judge collapses too (0.221).

| Config (qwen3.5:9b generator) | Ext Judge | Notes |
|---|---|---|
| adj=2 retrieval (`adj2_qwen9b`) | **0.516** | leader |
| full conversation in context, no retrieval | **0.090** | -0.426 absolute, **5.7× degradation** |

**This is the defining empirical answer to "does a memory system earn its keep when context is long enough to hold the whole conversation?"** Yes, by an enormous margin. Stuffing 30–60K tokens of dialogue into a capable 9B model with 128K window collapses Judge accuracy to slightly above mem0's floor (which uses retrieval). Retrieval is essential — context window size is not the bottleneck a memory system solves.

### Cross-tier context (mixed metrics, clearly labelled)

Other memory systems publish LoCoMo Judge numbers using `gpt-4o-mini` as the generator — a much larger hosted model than our 9B quant. Not apples-to-apples with our same-tier numbers above.

| System | Score | Generator | Notes |
|---|---|---|---|
| **taosmd** (this repo) | **0.509** | **qwen3.5:9b (local, 12 GB GPU)** | stack on 9B |
| OpenAI memory / Letta / LangMem | 0.50–0.52 | gpt-4o-mini (hosted) | mem0 paper baseline |
| Mem0 | 0.660 | gpt-4o-mini | arxiv 2504.19413 |
| Mem0^g (graph variant) | 0.684 | gpt-4o-mini | mem0 paper |
| Zep (self-reported) | 0.751 | gpt-4o-mini | audited 0.584 by `dial481/locomo-audit` |
| Genesys (self-reported) | 0.899 | ? | unverified |

At 0.509 on our local 12 GB GPU, we reach the Letta / LangMem / OpenAI-memory band *without* a cloud round-trip or a hosted frontier model. Mem0 paper (0.66) and Zep audited (0.584) remain ahead at cross-tier scale.

### Generator candidates at the 12 GB GPU tier — same retrieval stack, different generators

We took the leader recipe (`k=20 + adj=2 + llm-exp + RRF`) — proven on qwen3.5:9b — and ran six current open-source generators against the same 200-QA subset of LoCoMo, same external `qwen3:4b` rescore. The question being asked: at the 12 GB tier, is qwen3.5:9b actually the right default?

| Generator | VRAM | Bench (s, 200 QAs) | Ext rejudge | F1 | Notes |
|---|---|---|---|---|---|
| **qwen3.5:9b** Q4_K_M | 5.3 GB | 1749 | **0.56** | 0.226 | reference / production |
| mistral-small3.2 | ~5 GB | 4932 | **0.56** | 0.263 | ties qwen but 2.8× slower per QA |
| **llama3.1:8b** | 4.9 GB | 743 | 0.54 | 0.294 | -0.02, **2.4× faster**, fast-tier rec |
| gemma4:e4b | 9.6 GB | 1148 | 0.51 | 0.210 | -0.05 |
| granite4:tiny-h | 4.2 GB | 593 | 0.41 | 0.227 | -0.15, smaller model shows it |
| phi4-reasoning | — | timeout | — | — | reasoning tokens kill throughput at 200 QAs |

Headlines:

- **qwen3.5:9b stays as production default.** Best-in-class rejudge, no challenger meaningfully exceeds it, and it's faster than the only model that ties (mistral-small3.2 at 2.8× the wall-clock).
- **llama3.1:8b is a strong fast-tier recommendation.** Only -0.02 from the leader, 2.4× faster bench, 0.4 GB smaller. Trading -0.02 quality for ~2× throughput is sensible for users who care about realtime turn latency or run multiple agents on one GPU.
- **F1 vs rejudge divergence is real.** mistral-small3.2 has the highest F1 (0.263) of any 12 GB model — answers contain more lexical overlap with gold. Rejudge ties qwen at 0.56, so the extra wordiness doesn't translate to more correct answers — it just sounds more like the gold answer's surface text. Useful guardrail: F1 alone misranks generators.
- **phi4-reasoning is unfit for memory-recall at this tier.** Generates thousands of CoT tokens per call; projected 10–15 h for 200 QAs and didn't finish a single conversation in the budget. Same failure pattern as `--thinking-mode` on qwen3 (see negative results).

200-QA subset is a noisier estimate than the 1540-QA full leaderboard. The qwen3.5:9b row at Q4_K_M reproduces 0.56 here vs 0.557 on the full set — within noise. Subset numbers across generators on this table are directly comparable to each other (same 200 QAs) but should not be cross-compared to the full-1540 leaderboard rows without a ±0.01 caveat in mind.

Measured on Fedora 12 GB 3060 host, May 5 2026.

### 9B quant cliff — how much quality does each quant tier actually cost?

Same 200-QA subset, same leader recipe, same rescore. This time the generator family is fixed (qwen3.5:9b architecture) and we vary the quantisation:

| Quant | File size | Bench (s) | Ext rejudge | F1 | Δ vs Q4_K_M (0.56) |
|---|---|---|---|---|---|
| Q6_K | 7.6 GB | 2057 | 0.52 | 0.215 | -0.04 (counter-intuitive — see note) |
| Q5_K_M | 6.6 GB | 1900 | **0.56** | 0.212 | 0 |
| **Q4_K_M (production)** | 5.3 GB | 1749 | **0.56** | 0.226 | reference |
| IQ4_XS | 4.81 GB | 1622 | **0.55** | 0.233 | -0.01 |
| Q3_K_M | 4.7 GB | 1886 | 0.52 | 0.232 | -0.04 |
| UD-Q2_K_XL | 3.84 GB | 1739 | 0.51 | 0.275 | -0.05 |
| UD-IQ2_M | 3.4 GB | 1590 | 0.51 | 0.315 | -0.05 |
| Q3_K_S | 4.4 GB | 1999 | 0.49 | 0.255 | -0.07 |

Headlines:

- **The cliff is shallow above Q3.** Q4_K_M, Q5_K_M and IQ4_XS all sit within ±0.01. Q6_K drops 0.04 — at 200 QAs that's within subset-noise but worth noting as a non-monotonicity.
- **IQ4_XS is the 8 GB tier candidate.** 4.81 GB fits an 8 GB GPU with KV-cache headroom; -0.01 from production. IQ kernels run at K-quant speed on Ampere when the Modelfile is correct (see methodology note below) — 1622 s vs 1749 s for Q4_K_M.
- **UD-IQ2_M is the smallest 9B-family quant we tested.** 3.4 GB at -0.05 from production. It can fit a 4 GB GPU with KV-cache headroom, but `qwen3:4b` (a different architecture, measured at 0.530 on the 1050 Ti tier) outperforms it on quality and runs faster — for 4 GB GPUs, the qwen3:4b path is the recommendation, not UD-IQ2_M. See the 4 GB GPU hardware-tier section.
- **Q6_K's 0.52 is real but unexplained.** F1 (0.215) is also slightly lower than Q4_K_M's. Most likely subset-200 noise; the lower-bit Q5_K_M ties Q4_K_M at 0.56 in the same run. Reported as measured, not corrected for.
- **Below Q3 the quality gap widens.** Q3_K_S at 0.49 (-0.07) is the floor where the 9B becomes meaningfully worse than the leader.

Methodology note on custom quants: all V3 quants in this table were built via `ollama create` with full Modelfile metadata cloned from `ollama show qwen3.5:9b --modelfile` (TEMPLATE / RENDERER / PARSER / PARAMETER). A bare `FROM <gguf>` Modelfile silently applies the default Ollama chat template, which doesn't match the model's tuning, produces verbose non-terminating output, and looks indistinguishable from the kernel being slow. We measured ~130× per-QA latency on the first sweep before identifying the Modelfile root cause; the IQ-kernel-is-slow theory was incorrect.

Measured on Fedora 12 GB 3060 host, May 5 2026.

### Answer-prompt variants — does prompt engineering raise Single-hop?

Both candidate generators (qwen3.5:9b, llama3.1:8b) underperform on Single-hop relative to other LoCoMo categories. We swapped the production `ANSWER_PROMPT` for four alternative answer-prompt templates, holding everything else (retrieval stack, generator, dataset, external `qwen3:4b` rescore) constant. Goal: see whether prompt engineering — at the answer step, not the retrieval step — could raise Single-hop without architectural change.

| Variant | What it instructs | Length target |
|---|---|---|
| `concise` | "answer in 5-6 words; just the fact, no explanation" | ≤ 5–6 words |
| `refusal` | "only use facts directly stated in context; do NOT infer beyond what is written" | open |
| `citation` | "answer in 5-10 words, then cite the supporting turn `[Speaker, date]`" | 5–10 words + citation |
| `memobase` | brevity + "prioritise most recent on contradictions" + "don't confuse character names with users" | ≤ 5–6 words |

| Generator | Variant | Ext rejudge | Δ vs default | Single-hop rejudge | F1 |
|---|---|---|---|---|---|
| qwen3.5:9b | **default** | **0.56** | — | ~0.34 | 0.226 |
| qwen3.5:9b | concise | 0.39 | -0.17 | 0.23 | 0.433 |
| qwen3.5:9b | refusal | 0.33 | -0.23 | 0.14 | 0.320 |
| qwen3.5:9b | citation | 0.39 | -0.17 | 0.19 | 0.251 |
| qwen3.5:9b | memobase | 0.41 | -0.15 | 0.23 | 0.446 |
| llama3.1:8b | **default** | **0.54** | — | — | 0.294 |
| llama3.1:8b | concise | 0.38 | -0.16 | 0.19 | 0.383 |
| llama3.1:8b | refusal | 0.38 | -0.16 | 0.16 | 0.292 |
| llama3.1:8b | citation | **0.53** | **-0.01** | 0.28 | 0.185 |
| llama3.1:8b | memobase | 0.39 | -0.15 | 0.28 | 0.404 |

Headlines:

- **Single-hop hypothesis refuted.** No variant raised Single-hop rejudge on either model. The production `ANSWER_PROMPT` is locally optimal at this tier; the ceiling on Single-hop sits somewhere other than the answer prompt.
- **Prompt × model interaction is real.** `citation` costs qwen3.5:9b -0.17 but only -0.01 on llama3.1:8b. Plausible reading: llama's default-prompt output is wordier than qwen's; forcing the `[Speaker, date]` citation forces brevity that the judge reads as more committed/correct. F1 collapses to 0.185 (the lowest of any variant) — the surface form is very different — but rejudge holds. Useful guardrail: a "good prompt" for one open generator can be a poison pill for another at the same tier.
- **F1 ↑ / rejudge ↓ is a recurring pattern.** `concise` and `memobase` have the highest F1 (more lexical overlap with gold) but their rejudge regresses by 0.15–0.17 — short surface-aligned answers fool F1 but get marked wrong by the LLM judge. F1 alone misranks generators *and* prompts.

Measured on Fedora 12 GB 3060 host, May 5 2026. Variants and the `--answer-style {default,concise,refusal,citation,memobase}` runner flag live on branch `feat/locomo-prompt-variants` for reproduction.

### Embedder swap — does a stronger embedder raise Single-hop?

Same setup as the prompt-variant sweep but varying the *embedder* instead of the answer prompt. The motivation: Single-hop queries usually contain a specific noun phrase that needs precise retrieval, and our production embedder (MiniLM-L6, 22 M params, 384-dim) is small by 2026 standards. The hypothesis was that a larger 2025/2026-era retrieval-tuned embedder would lift Single-hop where prompt-engineering had failed.

We picked four current candidates with available ONNX exports and asymmetric query/document prefixes plumbed in:

| Candidate | Params | Dim | License | Released | Prefix scheme |
|---|---|---|---|---|---|
| **MiniLM-L6 v2** (production baseline) | 22 M | 384 | Apache 2.0 | 2021 | symmetric (no prefix) |
| EmbeddingGemma-300m | 308 M | 768 (MRL) | Apache 2.0 | Sep 2025 | `task: search result \| query: …` / `title: none \| text: …` |
| Snowflake Arctic Embed L v2.0 | 568 M | 1024 (MRL → 256) | Apache 2.0 | Dec 2024 | `query: …` for queries; no prefix for documents |
| Nomic Embed v1.5 | 137 M | 768 (MRL) | Apache 2.0 | Feb 2024 | `search_query: …` / `search_document: …` |

| Embedder | Overall rejudge | Δ vs leader | Single-hop | Temporal | Multi-hop | Open-dom | F1 |
|---|---|---|---|---|---|---|---|
| **MiniLM-L6 v2 (leader)** | **0.56** | — | ~0.34 | ~0.55 | ~0.4–0.5 | ~0.70 | 0.226 |
| Arctic Embed L v2.0 | 0.55 | -0.01 | 0.26 | **0.65** | **0.62** | 0.62 | 0.243 |
| EmbeddingGemma-300m | 0.54 | -0.02 | 0.26 | 0.63 | 0.31 | 0.64 | 0.238 |
| Nomic Embed v1.5 | 0.54 | -0.02 | 0.26 | 0.59 | **0.62** | 0.63 | 0.237 |

Headlines:

- **No candidate lifted Single-hop.** Every embedder swap dropped Single-hop to 0.26 (-0.08 absolute from the typical baseline). Same direction across three independently-trained models — that's a strong null on this lever.
- **Net overall change is small (-0.01 to -0.02).** Lifts on Temporal (Arctic +~0.10, EmbeddingGemma +~0.08, Nomic +~0.04) and Multi-hop (Arctic and Nomic both +~0.15) wash out against losses on Single-hop (-0.08) and Open-dom (-0.06 to -0.08).
- **Category mix shifts but the ceiling holds.** Stronger embedders trade Single-hop / Open-dom recall for Temporal / Multi-hop quality — they rerank the failure modes rather than reducing them.
- **EmbeddingGemma's Multi-hop collapse (0.31)** is interesting: half the Multi-hop score of the other two strong candidates despite being from the same generation. Something about how the asymmetric `title: none | text: …` document prefix interacts with multi-hop conversational turns appears to lose information that Arctic / Nomic preserve. Worth a follow-up.
- **F1 ↑ / rejudge ≈ pattern repeats.** All three candidates produce slightly higher F1 (~0.24 vs 0.23 baseline) but rejudge holds flat or drops — same trap as the prompt-variant sweep. F1 alone misranks the embedder choice.

Combined with the prompt-variant sweep, the takeaway across both negative results is: **the ceiling on Single-hop sits in the architecture, not in the answer prompt or the embedder.** Architectural levers (typed memory routing per ENGRAM, multi-vector retrieval, hybrid lexical+vector with stronger BM25) are the next slot.

Pending: Qwen3-Embedding-0.6B and Qwen3-Embedding-4B — the `onnx-community` ONNX exports keep causal-LM KV-cache inputs that don't fit our embedder loader, so they're queued via `--embed-mode local` (sentence-transformers + PyTorch CPU) on the same recipe. Results will be appended to this section once the chain finishes.

Measured on Fedora 12 GB 3060 host, May 6 2026. The asymmetric query/document prefix detection lives on branch `feat/embedder-asymmetric-prefixes-may5` for reproduction; full sweep summary at `/tmp/embedder_sweep_summary.tsv` on the bench host.

### Judge sensitivity — what we are really measuring

This is the most important finding in this document. Most existing public LoCoMo numbers (Mem0, Zep, EMem, HippoRAG, ENGRAM, LiCoMemory) are reported with **`gpt-4o-mini` as the LLM-judge**. We have used **`qwen3:4b` as the LLM-judge** since the start of this project — a deliberately strict, locally-runnable evaluator chosen so reproductions don't require API keys.

After hitting a stubborn ~0.28 Single-hop ceiling across four independent architectural levers (prompt, embedder, ENGRAM-typed retrieval, generator size), we asked whether the ceiling was the *judge's strictness* rather than the *system's quality*. We took **the same `qwen3.5:9b` leader-recipe predictions** (200-QA subset, no re-generation) and rescored them under five different LLM judges:

| Judge | Overall rejudge | **Single-hop** | Temporal | Multi-hop | Open-dom |
|---|---|---|---|---|---|
| **gemma4:e2b** | **0.71** | **0.53** | 0.62 | 0.85 | 0.86 |
| qwen3.5:9b (self-judge) | 0.64 | 0.35 | 0.62 | 0.77 | 0.78 |
| gemma4:e4b | 0.56 | 0.26 | 0.56 | 0.62 | 0.72 |
| **qwen3:4b** (our published baseline) | **0.54** | **0.28** | 0.59 | 0.77 | 0.60 |
| llama3.1:8b ⚠ | 0.35 | 0.21 | 0.27 | 0.46 | 0.48 |

⚠ The `llama3.1:8b` rescore returned in 26 seconds for 200 QAs (vs ~30-60 minutes for the other judges). Almost certainly a parsing failure or refusal pattern; treated as outlier and excluded from comparisons below.

The same pattern reproduces under a **second generator** (`qwen3.6:35b-a3b`, MoE):

| Judge | Overall rejudge | Single-hop | Temporal | Multi-hop | Open-dom |
|---|---|---|---|---|---|
| **gemma4:e2b** | **0.69** | **0.51** | 0.68 | 0.46 | 0.83 |
| qwen3.5:9b | 0.63 | 0.26 | 0.71 | 0.46 | 0.79 |
| gemma4:e4b | 0.56 | 0.26 | 0.63 | 0.31 | 0.72 |
| **qwen3:4b** (baseline) | **0.56** | **0.21** | 0.70 | 0.38 | 0.67 |

Headlines:

- **Same predictions, four different judges, Single-hop ranges 0.26 → 0.53 — a 2× variation.** The 0.28 floor we attributed to "the architecture is unmovable" was largely "the judge is strict." Architectural levers cannot lift Single-hop above the judge's ceiling.
- **gemma4:e2b is the most lenient open-source judge we tested.** It is closer to gpt-4o-mini's behaviour on subjective grading (lenient on minor wording differences, accepts "yes inferred from context" answers more readily). It is also coincidentally the **weakest generator** we tested at this tier (0.46 overall when used as a generator at the leader recipe) — the asymmetry is real: a model can be a useful evaluator without being a useful generator.
- **qwen3:4b is unusually strict** — flags inferences that gemma4:e2b accepts and penalises wording differences gemma4:e2b waves through. This is *not* a bug; it is just a different point on the strict ↔ lenient axis. We chose it for reproducibility (it's tiny, free, and never refuses) but its strictness costs us 0.15-0.25 on the Single-hop metric vs published-SOTA judging.
- **Self-judge inflation is +0.10** on this stack (qwen3.5:9b judging its own qwen3.5:9b output: 0.54 → 0.64). Smaller than the +15-22 pp historic estimate from the methodology memo, but consistent in direction.

Comparison to published systems on the same dataset:

| System | Reported Single-hop | Reported judge | Our equivalent (gemma4:e2b judge) |
|---|---|---|---|
| EMem | 0.83 (gpt-4o-mini) | gpt-4o-mini | unknown — would need to bench |
| Nemori | ~0.78 (gpt-4o-mini) | gpt-4o-mini | unknown |
| **taosmd qwen3.5:9b leader** | **0.28 (qwen3:4b)** | qwen3:4b | **0.53 (gemma4:e2b)** |
| **taosmd qwen3.6-MoE leader** | **0.21 (qwen3:4b)** | qwen3:4b | **0.51 (gemma4:e2b)** |

Our system at gemma4:e2b judge is **0.53 Single-hop** versus EMem's published 0.83 — a real gap of ~0.30, not the ~0.55 our qwen3:4b numbers suggested. The remaining gap is closer to "their generator (gpt-4o-mini) is much stronger than ours (qwen3.5:9b)" plus "they have EDU-level retrieval and LLM filtering" — both real, addressable architectural deltas, not unmovable ceilings.

#### Methodology change going forward

We will not retroactively re-rescore every leaderboard cell at gemma4:e2b — that breaks comparability with prior cells and looks like number-massaging. Instead:

1. **`qwen3:4b` remains the *headline* external judge** for the leaderboard above. It is the strictest judge we have access to, and reporting under a strict judge is the honest way to report. The 0.557 leader number stands.
2. **`gemma4:e2b` becomes the *secondary* judge**, reported alongside `qwen3:4b` for any new cell, so future readers can pick the comparison axis (strict-but-local-judge for self-comparison; lenient-frontier-equivalent for comparison with paper SOTA).
3. **New cells run from May 7 onward will be dual-rescored** under both judges. Every published number will carry both judge attributions explicitly (e.g. "0.557 / 0.71" or "Single-hop 0.28 (qwen3:4b) / 0.53 (gemma4:e2b)").
4. **Hardware-tier guidance**: 4-6 GB VRAM systems should keep `qwen3:4b` as their judge (it fits, runs fast, never refuses). 8 GB+ systems can run `gemma4:e2b` (7.2 GB) for matched-with-SOTA comparison numbers. Both produce useful but different signals.

The takeaway for anyone reading this doc: a single judge number is not a universal score. The right way to think about LoCoMo (and any LLM-judged benchmark) is as a *spread across reasonable judges*, not a single point. Our qwen3:4b number is at the strict end of that spread; gpt-4o-mini-equivalent numbers like EMem's 0.83 are at the lenient end. They measure overlapping but different things.

Measured on Fedora 12 GB 3060 host, May 6-7 2026. Multi-judge sweep scripts at `/tmp/multijudge_overnight.sh` and `/tmp/multijudge_phase2.sh` on the bench host; rescored JSONs at `benchmarks/results/locomo_*_engram3cell_leader_baseline_repro_*.rescored_judge_*.json` and equivalents for the qwen3.6-MoE Phase 1 output.

#### Dual-judge results across May 7 architectural levers

After the multi-judge experiment confirmed the strict-judge ceiling, we re-rescored every May 7 architectural-lever JSON under `gemma4:e2b` (lenient/published-SOTA-equivalent) alongside the existing `qwen3:4b` numbers. Same predictions, no re-generation — pure judge-strictness measurement on architectural choices.

| Cell | Overall (q3:4b → g4:e2b) | Single-hop | Temporal | Multi-hop | Open-dom |
|---|---|---|---|---|---|
| rrf_baseline_heuristic (production) | 0.55 → **0.69** | 0.30 → 0.53 | 0.59 → 0.60 | 0.46 → 0.69 | 0.65 → 0.85 |
| bm25_rrf_proper | 0.52 → **0.70** | 0.23 → 0.44 | 0.56 → 0.65 | 0.62 → 0.62 | 0.62 → 0.89 |
| bm25_lemma_rrf | 0.52 → 0.68 | 0.21 → 0.49 | 0.60 → 0.60 | 0.31 → 0.38 | 0.65 → 0.88 |
| **mem0_additive** | 0.54 → **0.70** | 0.26 → **0.56** | 0.54 → 0.59 | 0.69 → **0.77** | 0.65 → 0.85 |
| cove_4step | 0.50 → 0.65 | 0.26 → 0.56 | 0.51 → 0.54 | 0.69 → 0.69 | 0.59 → 0.78 |

Headlines:

- **Every lever lifts +0.14 to +0.18 overall under gemma4:e2b judge.** Uniform shift confirms strict-judge was the ceiling, not the architecture.
- **`mem0_additive` is the lever that wins under matched-judge.** 0.70 overall, 0.56 Single-hop, 0.77 Multi-hop — ties proper-BM25 on overall, beats it on the two categories that matter most for LoCoMo. The Mem0-style threshold-gated additive scoring (sigmoid-normalised BM25 + cosine, max_possible adapter) is doing real work that RRF flattens away.
- **Proper BM25 vs the legacy substring heuristic is a wash overall.** 0.70 vs 0.69. The "real BM25 is what production memory systems use" story we ported was right *for principle* (lemmatised, IDF-weighted, length-normalised) but the lift over the heuristic is below subset-200 noise (~±0.02). Lemmatisation slightly hurt at our chat-turn granularity (over-stemmed proper nouns).
- **CoVe at 4× wall-clock buys parity, not advantage.** Same 0.56 Single-hop as mem0_additive but 0.05 lower overall. The verifications surface answer-edits that gemma4:e2b waves through but don't compound into a win at our 9B tier. Don't ship.

#### Dual-judge results across May 5 generator candidates

Re-rescored the May 5 generator-candidate sweep JSONs under both judges so the README hardware-tier table can carry both attributions:

| Generator | Overall (q3:4b → g4:e2b) | Single-hop | Temporal | Multi-hop | Open-dom | F1 |
|---|---|---|---|---|---|---|
| **qwen3.5:9b leader** | 0.54 → **0.71** | 0.28 → 0.53 | 0.59 → 0.62 | 0.77 → **0.85** | 0.60 → **0.86** | 0.231 |
| mistral-small3.2 | 0.56 → 0.70 | 0.21 → 0.53 | 0.70 → **0.71** | 0.38 → 0.54 | 0.67 → 0.81 | 0.263 |
| **llama3.1:8b** | 0.54 → 0.67 | — → **0.65** | — → 0.51 | — → 0.69 | — → 0.80 | **0.294** |
| gemma4:e4b | 0.51 → 0.60 | — → 0.33 | — → 0.51 | — → 0.38 | — → 0.85 | 0.210 |
| granite4:tiny-h | 0.41 → 0.56 | — → 0.42 | — → 0.35 | — → 0.54 | — → 0.79 | 0.227 |
| gemma4:e2b (as generator) | 0.46 → 0.58 | 0.16 → 0.37 | 0.54 → 0.49 | 0.31 → 0.31 | 0.58 → 0.81 | 0.217 |

Per-category headlines (gemma4:e2b judge):

- **qwen3.5:9b is best on Overall (0.71)** and ties for best on Multi-hop (0.85) and Open-dom (0.86). Production generator stays.
- **llama3.1:8b is best on Single-hop (0.65)** by a wide margin (+0.12 over qwen3.5:9b's 0.53). Closest we have to the published-SOTA band — only 0.18 below EMem's 0.83 (gpt-4o-mini judge). Single-hop is the most common factual-QA pattern, so for **factual recall workloads, llama3.1:8b is now the better pick**.
- **mistral-small3.2 is best on Temporal (0.71)**. Time-anchored queries lean heavily on lexical-temporal cues that mistral handles well. The 2.8× wall-clock cost still rules it out for production, but for time-heavy specialty workloads it's a contender.
- **gemma4:e4b and granite4:tiny-h** stay middle-of-pack regardless of judge — useful for tighter-VRAM tiers but not promotion candidates.

The "qwen3.5:9b production / llama3.1:8b fast alternative" framing from the May 5 PR was right but underspecified. The matched-judge breakout shows it's actually a **per-workload pick at the 12 GB tier**:

> **12 GB GPU production guidance (matched-judge methodology):**
> - **Best overall quality** → `qwen3.5:9b` Q4_K_M (0.71 overall under gemma4:e2b)
> - **Best factual recall (Single-hop)** → `llama3.1:8b` (0.65 Single-hop, 2.4× faster than qwen)
> - **Best temporal reasoning** → `mistral-small3.2` (0.71 Temporal, but 2.8× slower than qwen)
> - **Production default**: still `qwen3.5:9b` because Single-hop is one of four categories; broad workloads should optimise for Overall.
> - **Specialty deployments**: pick the workload-matched generator above.

Measured on Fedora 12 GB 3060 host, May 7 2026. Dual-rescore script at `/home/jay/dual_judge_rescore.sh`; full summary at `/tmp/dual_judge_rescore_summary.tsv`. The full leaderboard above (line 54+) was scored exclusively under `qwen3:4b` and is *not* retroactively rescored — every cell from May 7 onward carries both judge attributions explicitly.

#### Generator-temperature sweep

Until May 9 every LoCoMo cell ran at the runner's hardcoded `temperature=0.2`. We added a `--gen-temp` flag and swept four generators × three temperatures (0.0, 0.2, 0.5) at the leader recipe + `mem0_additive` fusion, subset 200, dual-rescored.

| Generator | temp 0.0 | temp 0.2 (existing) | temp 0.5 | Sweet spot |
|---|---|---|---|---|
| **gemma4:e2b** | 0.59 / **0.49** | 0.57 / 0.37 | 0.60 / 0.35 | 0.0 for SH; 0.5 for Overall |
| **llama3.1:8b** | 0.65 / **0.60** | 0.62 / 0.47 | 0.63 / 0.51 | **temp 0.0 across the board** |
| **qwen3.5:9b** | 0.67 / 0.53 | **0.70** / **0.56** | 0.65 / 0.58 | **temp 0.2 sweet spot** |
| **gemma4:e4b** | 0.62 / 0.53 | 0.61 / 0.42 | **0.65** / 0.51 | 0.5 for Overall, 0.0 for SH |

Numbers are `Overall / Single-hop` under `gemma4:e2b` judge. Bold = peak per generator per metric.

Headlines:

- **Sampling temperature is per-generator, not universal.** qwen3.5:9b's training distribution prefers low-but-nonzero temp (0.2); llama3.1:8b prefers fully-greedy (0.0); gemma4:e4b prefers 0.5 for Overall but 0.0 for Single-hop; gemma4:e2b is similarly split. There's no "always use temp X" rule for our local-tier stack.
- **llama3.1:8b at temp 0.0 + mem0_additive = 0.65 / 0.60 Single-hop.** The +0.13 Single-hop lift (0.47 → 0.60) just from temperature is the largest single-lever effect we've measured since the judge-strictness pivot. Greedy decoding lets llama commit to the most-likely token rather than sampling around the answer's exact form — which the judge then accepts more often.
- **qwen3.5:9b + mem0_additive + temp 0.2 (0.70 / 0.56) is still the Overall leader.** No temp-sweep cell beat it on overall. Production default holds.
- **Best Single-hop overall is still `llama3.1:8b` at RRF heuristic + temp 0.2 (0.65 SH, May 5 sweep).** mem0_additive *hurts* llama's Single-hop (-0.05 vs RRF heuristic) — the lever that helps qwen3.5:9b doesn't help llama. Different generators interact with different fusion modes differently. Per-generator + per-fusion + per-temp tuning is genuinely a thing here.

Updated 12 GB tier guidance (matched-judge methodology, gemma4:e2b):

> - **Best Overall** → `qwen3.5:9b` + `--fusion mem0_additive` + `--gen-temp 0.2` (0.70 / 0.56 SH)
> - **Best Single-hop / factual recall** → `llama3.1:8b` + `--fusion rrf` + `--gen-temp 0.2` (0.67 / **0.65 SH**)
> - **Best Single-hop within mem0_additive** → `llama3.1:8b` + `--fusion mem0_additive` + `--gen-temp 0.0` (0.65 / 0.60 SH)
> - **Multi-hop / Temporal** → `qwen3.5:9b` + `mem0_additive` + temp 0.2 (0.77 / 0.59)
> - **Production default**: `qwen3.5:9b` + `mem0_additive` + temp 0.2 — best Overall and within 0.01 of best Multi-hop / Open-dom.

Measured on Fedora 12 GB 3060 host, May 8-9 2026. Bench script at `/home/jay/temp_sweep_bench.sh`; full summary at `/tmp/temp_sweep_bench_summary.tsv`. `--gen-temp` flag lives on branch `feat/gen-temp-flag` (commit `2bd1b21`) for reproduction.

#### Subset 200 → full 1540 validation

The May 7-9 temp-sweep cells were all measured at the 200-QA subset for chain throughput. On May 10-11 we re-ran the two cells that were headline picks in the README at full 1540 QAs so a public-facing ranking isn't sitting on subset noise.

| Recipe | Subset 200 (g4e2b) Overall / SH | Full 1540 (g4e2b) Overall / SH | Δ Overall | Δ SH | Verdict |
|---|---|---|---|---|---|
| **qwen3.5:9b + mem0_additive + temp 0.2** (leader) | 0.71 / 0.56 | **0.68 / 0.55** | −0.03 | −0.01 | **Generalises.** Production default holds. |
| llama3.1:8b + RRF + temp 0.2 (subset SH pick) | 0.67 / 0.65 | 0.64 / 0.49 | −0.03 | **−0.16** | **Regression.** Subset 200 over-represented Single-hop questions where llama+RRF won. Removed from README. |

Both cells now dual-judge complete:

| Cell | qwen3:4b Overall / SH | gemma4:e2b Overall / SH |
|---|---|---|
| qwen3.5:9b + mem0_additive + temp 0.2 | 0.54 / 0.28 | 0.68 / 0.55 |
| llama3.1:8b + RRF + temp 0.2 | 0.53 / 0.27 | 0.64 / 0.49 |

The qwen3:4b rescore confirms the gemma4:e2b finding: llama+RRF Overall trails qwen+mem0 by 0.01 under the strict judge too (0.53 vs 0.54), and Single-hop trails by 0.01 (0.27 vs 0.28). Both judges agree the leader recipe wins both metrics at full 1540.

Methodology takeaways:

- **Overall drift is uniformly small (−0.03)** across both recipes — subset 200 is a fair Overall sampler.
- **Single-hop drift is recipe-dependent.** qwen+mem0 holds within noise (−0.01). llama+RRF collapses (−0.16) under gemma4:e2b. Subset 200 happened to over-represent the Single-hop questions where llama+RRF won at our recipe / generator pairing.
- **Subset 200 is no longer sufficient for promoting a workload-specific recommendation** — particularly per-category picks. Subset → full validation is now required before any README claim that names a specific recipe as "best at X."

Measured on Fedora 12 GB 3060 host, May 10-11 2026. Bench script at `/tmp/llama_rrf_gapfill_bench.sh`; full summary at `/tmp/llama_rrf_gapfill_summary.tsv` on the bench host.

#### May 12-16 architectural levers — EMem-EDU and HippoRAG-PPR (preliminary, subset 200)

Two architectural lever ports tested alongside the May 7-9 fusion + temp sweeps. Both run with the full leader stack (`qwen3.5:9b + --retrieval-top-k 20 --adjacent-turns 2 --llm-query-expansion --fusion mem0_additive --gen-temp 0.2`) so the only varying axis is the new lever. Full-1540 validation for the promotion candidate is in flight at time of writing.

**EMem-EDU (vector-only variant)** — port of arXiv:2511.17208. One LLM call per session at ingest decomposes turns into atomic Elementary Discourse Units (EDUs) with normalized entities and turn attribution. Optional per-QA LLM filter step then narrows the candidate pool.

| Cell | qwen3:4b Overall / SH | gemma4:e2b Overall / SH | Δ vs baseline (g4e2b) |
|---|---|---|---|
| subset200_baseline (leader, no EMem) | 0.50 / 0.21 | 0.67 / 0.53 | — |
| subset200_emem_filter (EMem + LLM filter) | 0.49 / 0.23 | 0.65 / 0.53 | −0.02 / 0.00 |
| **subset200_emem_nofilter (EMem storage only)** | **0.51 / 0.35** | **0.68 / 0.58** | **+0.01 / +0.05** |

Headlines:

- **EMem-EDU storage alone is the first new architectural lever to beat baseline since `adjacent_turns` landed.** +0.05 SH under gemma4:e2b judge, **+0.14 SH under qwen3:4b** (0.21 → 0.35). Strict judge especially rewards the cleaner EDU-form answers — atomic propositions with normalized entities resolve fewer wording-mismatch penalties than raw turn quotes.
- **The LLM filter step is net-negative at our model tier.** Two regressions visible: −0.05 SH g4e2b vs no-filter, and even with the May 14 `difflib.get_close_matches(cutoff=0.6)` fuzz-match fix (parses now succeed), filter step ranks below ingest-only. `mem0_additive` already handles candidate selection well; adding an LLM filter on top costs latency and quality.
- **First May 11 chain reported regressions vs leader** because the bench script stripped `--adjacent-turns 2 --llm-query-expansion --fusion mem0_additive` — the actual leader levers. Real signal is the May 14 rerun with fair flags, shown above.

**HippoRAG-PPR (vector-only variant)** — port of arXiv:2405.14831. One LLM call per session at ingest extracts OpenIE (s, p, o, turn_ids) triples. In-memory igraph with entity-nodes + turn-passage-nodes + fact-edges. At query time: dot-product query embedding against pre-embedded facts → top-k facts seed entity weights → `igraph.personalized_pagerank` ranks turns.

| Cell | qwen3:4b Overall / SH | gemma4:e2b Overall / SH | Δ vs baseline (g4e2b) |
|---|---|---|---|
| subset200_fair (linking-top-k=5, fair flags) | 0.33 / 0.23 | 0.47 / 0.47 | −0.20 / −0.06 |
| subset200_fair_top10 (linking-top-k=10) | 0.38 / 0.26 | 0.53 / 0.53 | **−0.14 / 0.00** |

Headlines:

- **HippoRAG-PPR ties baseline on Single-hop at `--hipporag-linking-top-k 10`** (0.53 g4e2b, both) but regresses Overall by −0.14. PPR graph traversal helps factual single-fact lookups (which the SH category measures) but underperforms on multi-evidence aggregation tasks (Temporal / Multi-hop / Open-dom).
- **Top-k seed count matters.** Going from 5 → 10 seeds adds +0.06 g4e2b Overall and +0.06 SH. Suggests our small-corpus per-conversation KG is sparse enough that broader seeding meaningfully changes the propagation landscape.
- **EMem-EDU subsumes HippoRAG's SH win cheaply.** Both lift SH similarly under matched flags (EMem +0.05, Hippo 0.00) but EMem doesn't pay the per-conversation igraph-build cost and isn't a regression on Overall. HippoRAG-PPR is not promotion-worthy as a default at our tier; possibly a future `--retrieval-mode hipporag` opt-in for factual-recall-only workloads.

**Promotion gating:** EMem-EDU (no filter) full-1540 validation queued (`ememfull` chain — runs after hippo2 completes). Per the May 10-11 subset→full lesson, no README promotion until the full-1540 numbers land. Production default on master remains `qwen3.5:9b + mem0_additive + temp 0.2` baseline until then.

**`mem0_additive` + `--gen-temp` are now defaults on master** as of PR #69 (e1e3fc2). Previously these flags lived only on feature branches (`feat/bm25-hybrid`, `feat/gen-temp-flag`) and could not be reproduced from master without checking out the feature branch — a footgun for any external user trying to reproduce the leader recipe. Three benches in this period crashed at argparse for this reason; consolidated to master to close that footgun permanently.

Measured on Fedora 12 GB 3060 host, May 12-16 2026. Bench scripts at `/tmp/emem_edu_bench2.sh` (EMem) and `/tmp/hipporag_bench2.sh` (Hippo) on the bench host; summaries at `/tmp/emem_edu_bench2_summary.tsv` and `/tmp/hipporag_bench2_summary.tsv`. Branches: `feat/emem-edu-filter` (commit `24fcf6f` — fuzzy-match fix in `filter_edus` + merge of `gen-temp-flag`) and `feat/hipporag-ppr` (commit `e7745f9`).

#### EMem-EDU full-1540 validation (May 17)

The May 14-15 subset-200 results promoted EMem-EDU (no filter) as the first new architectural lever to clear baseline on Single-hop since `adjacent_turns`. Per the May 10-11 subset→full discipline (`llama+RRF` over-claimed at subset 200, regressed at full 1540), we re-ran both the baseline and the EMem-EDU candidate at the full 1540 QAs before any production promotion.

| Cell | qwen3:4b Overall / SH | gemma4:e2b Overall / SH |
|---|---|---|
| baseline (qwen3.5:9b + mem0_additive + temp 0.2) | 0.52 / 0.25 | **0.69 / 0.53** |
| **EMem-EDU no filter (same recipe)** | **0.52 / 0.35** | **0.67 / 0.60** |
| Δ vs baseline | **+0.10 SH (q3:4b)**, 0.00 Overall | **+0.07 SH (g4:e2b)**, −0.02 Overall |

Headlines:

- **The subset-200 win generalises.** Subset 200 said +0.05 SH g4:e2b / +0.14 SH q3:4b; full 1540 lands at **+0.07 SH g4:e2b / +0.10 SH q3:4b**. Same direction, similar magnitude. The May 10-11 subset→full discipline catches over-claims (the llama+RRF cell collapsed −0.16 SH at full scale); this one passed.
- **EMem-EDU is a Single-hop specialist, not a new universal default.** Trades −0.02 g4:e2b Overall for +0.07 SH (∼2× the SH gain vs the Overall cost). On q3:4b it ties Overall (0.52 vs 0.52) while gaining +0.10 SH — a clean win under the strict judge.
- **The strict judge especially rewards EDU-form answers.** qwen3:4b's stricter accept policy resolves more EDU-style atomic-proposition answers as correct than full-turn quotes (0.25 → 0.35 SH = +0.10 absolute, a 40 % relative lift). The lenient g4:e2b judge sees a smaller but still real lift.
- **Filter step stays net-negative at our model tier.** The May 14 difflib fuzz-match fix made filter calls parse correctly, but the LLM-filter cell still trailed the no-filter cell on subset 200. We did not re-run the LLM-filter cell at full 1540 since the subset result was clear; mem0_additive already does candidate selection that the LLM filter then over-cuts.
- **Recipe is one extra ingest flag away from production.** Append `--emem-edu --emem-edu-extract-model llama3.1:8b --emem-edu-no-filter` to the leader-recipe bench call (or, programmatically, pass `emem_edu=True` to the equivalent runner / store init path on `feat/emem-edu-filter`). One LLM call per session at ingest (~5-10 s/session via llama3.1:8b on a 12 GB GPU). No per-query LLM cost at retrieval — filter step disabled.

Promotion decision: **EMem-EDU (no filter) is the recommended retrieval mode for Single-hop-heavy workloads on the 12 GB GPU tier**; production default for general workloads stays `mem0_additive + temp 0.2` baseline pending more bench evidence on whether the −0.02 Overall trade-off matters for the user's task mix. Both options now have full-1540 dual-judge measurements.

Measured on Fedora 12 GB 3060 host, May 17 2026. Bench script at `/tmp/emem_full1540_bench.sh`; full summary at `/tmp/emem_full1540_bench_summary.tsv`. Branch `feat/emem-edu-filter` (commit `24fcf6f`). Total wall time: 2 cells × (~5 h bench + ~9 h dual-rescore) ≈ 28 h.

### ENGRAM-style typed retrieval — does typed memory routing raise Single-hop?

The third architectural lever we tested at this generator tier. Same setup as the prompt and embedder sweeps but varying the *retrieval routing*: instead of one undifferentiated vector store, classify each conversation turn into three typed memory stores at ingest, then fan out per-type top-k searches at retrieval and set-merge before the leader pipeline.

Based on **[ENGRAM: Effective, Lightweight Memory Orchestration for Conversational Agents](https://arxiv.org/abs/2511.12960)** (Nov 2025), which reports **77.55 LLM-as-Judge** on LoCoMo with text-embedding-3-small + GPT-4o-mini and shows a **+31 absolute pp** ablation drop when typed separation is collapsed into a single store (77.55 → 46.56). The paper's claim: typed separation matters more than the embedder or generator choice.

We replicated the architectural shape on our local stack:

- **Per-turn classifier**: `llama3.1:8b` returns a 3-bit mask `{episodic, semantic, procedural}` per LoCoMo turn at ingest. Cached by SHA-256 of the turn text, so re-runs don't re-classify. ~290 ms/turn effective at 4-way concurrency.
- **Storage**: each vector entry tagged with the mask in its metadata column.
- **Retrieval**: new `--retrieval-mode engram_typed` runs three top-k vector searches (one per type bit), set-merges by row id with dedup, then runs the existing leader stack (cross-encoder rerank + RRF + adj=2) on the merged candidate pool.
- **Question routing**: also tested an `oracle_routed` mode that uses the dataset's known `qa["category"]` to pick mode per question — categories 3 and 4 (Multi-hop / Open-dom) use `engram_typed`, categories 1 and 2 (Single-hop / Temporal) use `default`. This is the *upper bound* on category-routing; no real classifier can beat oracle.

Three-cell experiment, all at the leader recipe (`k=20 + adj=2 + llm-exp + RRF`), 200 QAs subset, external `qwen3:4b` rescore:

| cell | overall rejudge | single-hop | temporal | multi-hop | open-dom | F1 |
|---|---|---|---|---|---|---|
| **leader_baseline_repro** | **0.54** | 0.28 | 0.59 | **0.77** | 0.60 | 0.231 |
| engram_typed | 0.53 | 0.30 | 0.57 | 0.54 | 0.62 | 0.226 |
| oracle_routed | 0.53 | 0.33 | 0.60 | 0.46 | 0.58 | 0.225 |

Classification distribution on LoCoMo turns (`llama3.1:8b` over 419 turns, conversation 26): **51.8 %** pure semantic, **28.6 %** episodic+semantic dual, **18.6 %** pure episodic, **~1 %** anything procedural. The three types are clearly differentiated — *not* a degenerate "everything is everything" mask — but procedural content is essentially absent in casual chat (matches the paper's expectation).

Headlines:

- **Overall is flat (-0.01).** Whatever the typed routing buys back on Open-dom (+0.02) and Single-hop (+0.02-0.05) is paid for by Multi-hop and Temporal regressions. Net moves are within subset-200 noise (~±0.02) on identical recipes.
- **Multi-hop regresses HARD with typed retrieval (-0.23).** Counter-intuitive — Multi-hop is the category we expected to *win* most. The 3-way fanout brings in less-relevant turns (each type's per-type top-k can include weaker candidates than a single combined top-k would have surfaced), and the set-merge dilutes the cross-encoder's ranking signal across heterogeneous candidates. Same direction in both ENGRAM-using cells (engram_typed at 0.54, oracle_routed at 0.46), so it isn't noise — it's a real cost of the architectural choice at this scale.
- **Single-hop is the third null lever.** 0.28 → 0.30 → 0.33 across the three cells is within subset noise on 43 questions. Combined with the prompt and embedder nulls, **the Single-hop ceiling at our generator tier is not in retrieval at all**. The hypothesis going into Phase 1 is that the bottleneck is generator extraction quality — getting the right fact out of an already-decent retrieved context — which is not what typed routing fixes.
- **Oracle-routed is also flat (0.53).** Even with *perfect* category routing — engram_typed only where it might help — overall stays within ±0.01 of baseline. Per the decision rule, this rules out building a query-time classifier (Phase 2b): no real classifier can outperform the oracle ceiling.

Why the gap with the paper's +31 pp: ENGRAM uses GPT-4o-mini as both generator and judge, with text-embedding-3-small for retrieval — a frontier-class generator paired with a hosted retrieval embedder. At that tier, retrieval and generation hit different failure modes, and typed routing recovers recall the large generator can already extract from. At our 9 B generator + 384-dim MiniLM-L6 + qwen3:4b external judge, the bottleneck distribution is different — the leader recipe already saturates retrieval recall for Multi-hop and Open-dom, and the remaining errors are generator-extraction failures that typed routing cannot address.

Methodology note: the *first* engram_typed run on May 6 reported a +0.16 Multi-hop lift. That was a wiring bug — `--retrieval-mode` was silently ignored when `--strategy=vector-only` (the bench default) because the runner's `_retrieve` short-circuited to a direct `vmem.search` call before threading the mode through. Fixed in commit `9306bb2` on `feat/engram-typed-retrieval`; the pre-fix numbers were retracted internally. The table above is post-fix.

The fourth lever — generator size — is the next experiment. `qwen3.6:35b-a3b` (Q4_K_M, ~23 GB, MoE: 35 B total / 3 B active) at the same leader recipe runs as Phase 1 on Fedora the night of May 6.

Measured on Fedora 12 GB 3060 host, May 6 2026. Branch `feat/engram-typed-retrieval` carries the EngramRouter classifier, the `engram_typed` / `oracle_routed` retrieval modes, and the bench script `engram_routed_3cell.sh` for reproduction. Cached classifications at `/tmp/engram_classifications.json`, full sweep summary at `/tmp/engram_routed_3cell_summary.tsv` on the bench host.

### Methodology disclosures

- **Dataset**: LoCoMo-10, 1540 QAs, categories 1–4. Adversarial reserved.
- **Same prompt** across every system (`ANSWER_PROMPT` in `benchmarks/locomo_runner.py`).
- **External judge**: `qwen3:4b`, temperature 0.0, via `locomo_rescore_streaming.py`.
- **No cherry-picking**: 100 % coverage on every rescore, 0 errors. Retractions (one prediction miss, one think=false-on-judge corruption) are documented in `docs/specs/2026-04-19-locomo-scorecards.md`.
- **`think=false` on generator** for Qwen3/3.5/3.6 (PR #42). A `thinking_mode_on` control run is queued to measure whether chain-of-thought changes the result.

Full timeline, per-category breakdowns, and provenance log: `docs/specs/2026-04-19-locomo-scorecards.md`.

---

## Hardware tiers — recommended configurations

The taosmd architecture is portable; what changes per hardware tier is which generator + which retrieval flags fit the compute budget. The following recommendations name what is **measured** vs. **expected based on architecture** so you can calibrate trust.

### Always-on defaults (every tier)

- **Embedder**: `all-MiniLM-L6-v2` ONNX on CPU. 384-dim, ~90 MB, 0.3–10 ms per embed across all tested CPUs. Avoid PyTorch — it's 200× slower for the same quality at this model size.
- **Reranker**: `ms-marco-MiniLM` ONNX on CPU. Same backend, second-stage rerank over top-K vector hits.
- **Don't enable** `--multihop-decompose`. Regresses at every model size we measured (5B: -0.093, 9B: even worse). Footgun.
- **Skip** `--context-format session_date`. No-op once the answer prompt has absolute dates (taosmd's prompt-opt default).

### 12 GB NVIDIA GPU (RTX 3060, RTX 3060 Ti, RTX 4060 Ti) — measured

This is the LoCoMo benchmark host. All numbers in the LoCoMo leaderboard above were measured here.

- **Generator (production default — best quality)**: `qwen3.5:9b` Q4_K_M (5.3 GB on disk, ~12 GB used at runtime) + leader recipe (`--retrieval-top-k 20 --adjacent-turns 2 --llm-query-expansion --fusion rrf`). Measured **0.557** ext judge on full LoCoMo (1540 QAs) and 0.56 on the 200-QA subset — within noise. Use `think=false` (built into the runner since PR #42) — 20× speedup with no measured quality loss.
- **Generator (fast-tier alternative within the same hardware)**: `llama3.1:8b` (4.9 GB on disk) + the same leader recipe. Measured **0.54** ext rejudge on 200 QAs — only -0.02 from the qwen leader, but **2.4× faster per QA** (743 s vs 1749 s). Right pick for users who care about realtime turn latency or run multiple agents on one GPU.
- **Tied-but-slower (not promoted)**: `mistral-small3.2` matches qwen at 0.56 ext rejudge but takes 2.8× the wall-clock per QA, so the speed cost outweighs the no-quality-gain at this tier. Documented in the generator-candidates table above for completeness.
- **Don't use at this tier**: `phi4-reasoning` — reasoning tokens kill throughput; same failure mode as `--thinking-mode` on qwen3 (see negative results).
- **Smaller-quant fallback for tighter VRAM budgets**: `qwen3.5:9b` IQ4_XS (4.81 GB) at 0.55 — see the 9B quant cliff table. Useful if KV cache is hitting the ceiling at long contexts.
- **Judge (when running benchmarks)**: `qwen3:4b` (Q4 ~5 GB VRAM). Default thinking mode (do **not** pass `think=false` — it's an Ollama bug on qwen3:4b that exposes reasoning in the response and corrupts judge parsing).
- **Concurrency**: 2 with qwen3.5:9b or llama3.1:8b. Higher concurrency benefits from `OLLAMA_NUM_PARALLEL=3`.

### 8 GB NVIDIA GPU (RTX 3050 8 GB, RTX 4060, RTX 2070, GTX 1070) — extrapolated from 9B quant data

The 9B quant cliff sweep above gives us a defensible 8 GB-tier recommendation without a separate measurement: same architecture, same retrieval stack, smaller-bit weights. The dedicated benchmark on 8 GB hardware is queued as a follow-up.

- **Generator (best fit)**: `qwen3.5:9b` IQ4_XS (4.81 GB on disk). Measured **0.55** ext rejudge on Fedora 12 GB at 200 QAs leader recipe — within 0.01 of the Q4_K_M production default at 5.3 GB. Fits an 8 GB GPU with ~3 GB headroom for KV cache, which holds for typical conversation lengths.
- **Smaller-VRAM alternative**: `qwen3.5:9b` UD-Q2_K_XL (3.84 GB) at 0.51 — roughly -0.05 from production but leaves 4+ GB for the KV cache if you're running long contexts.
- **Same retrieval stack as the 12 GB tier.** Leader recipe (`k=20 + adj=2 + llm-exp + RRF`) is what the IQ4_XS row was measured on; no separate tuning required for this tier.
- **Don't use**: bare `FROM <gguf>` Modelfiles. The custom quants in our sweep needed full metadata (TEMPLATE / RENDERER / PARSER / PARAMETER) cloned from `ollama show qwen3.5:9b --modelfile` to behave correctly. Without it, output is verbose and doesn't terminate — looks 100×+ slower per QA than the kernel actually is.
- **Concurrency**: 1 — at 4.81 GB plus KV cache an 8 GB card has no headroom for a parallel request without spilling.

A native 8 GB measurement would replace the "extrapolated" label in the next pass; the architecture and quant numbers we have predict ~0.55 ext rejudge at this tier on the same retrieval stack.

### 16 GB Orange Pi 5 Plus (RK3588 NPU) — measured

Both LongMemEval-S 97.0% reference stack AND LoCoMo measurements now exist on this tier. Same external `qwen3:4b` judge as the 12 GB benchmarks → directly comparable.

LoCoMo measurements (qwen3-4b-chat via rkllama on the NPU, all with `--adjacent-turns 2`):

| Config | Ext Judge | Δ vs Pi baseline (0.382) |
|---|---|---|
| **adj=2 + k=20 + llm-exp + RRF (full leader stack)** | **0.490** | **+0.108** |
| adj=2 + BGE-reranker-v2-m3 | 0.456 | +0.074 |
| adj=2 + multi-level retrieval + RRF | 0.425 | +0.043 |
| adj=2 + 5 few-shot exemplars | 0.39 | +0.01 (noise) |
| adj=2 (baseline) | 0.382 | — |
| adj=2 + leader stack + few-shot | 0.38 | -0.002 vs baseline / **-0.11 vs full_stack leader** |
| adj=2 + HyDE | 0.325 | -0.057 (HyDE regresses on small generators) |

The +0.074 BGE-v2-m3 lift on this tier is ~12× the lift the same swap gives on the 12 GB GPU at qwen3.5:9b — see the "Architecture matters more at smaller compute tiers" finding above.

Practical operational notes (learned the hard way): use the official ≥4 A USB-C PSU and active cooling under sustained inference workloads. The combined NPU + CPU + co-tenant load (e.g. running scrypted on the same Pi) will overdraw a stock charger and cause silent power-related kernel hangs after many hours of continuous load. Use the Pi as a dedicated AI worker if you can — co-tenant services should ideally run on a separate machine.

This is also the LongMemEval-S 97.0% reference stack — same configuration as the LoCoMo measurements above.

- **Generator**: `Qwen3-4B` via `rkllama` on the RK3588 NPU (~17 s/turn). Exact stack used for both the 97 % LongMemEval-S claim and the LoCoMo measurements at this tier.
- **Embedder**: `all-MiniLM-L6-v2` ONNX on CPU (0.3 ms — NPU is slower for small models).
- **Reranker**: `Qwen3-Reranker-0.6B` on NPU.
- **Query expansion**: `qmd-query-expansion 1.7B` on NPU.
- **Expected best LoCoMo config**: `--adjacent-turns 2` (architecturally consistent with the 12 GB measurements; not yet validated on the NPU stack). LongMemEval 97.0% measurement on master used the cross-encoder + query expansion path without an explicit adj flag — that path's per-section scores are in the LongMemEval table above.
- **Don't run the judge on the same Pi.** Offload to a peer (Fedora or another Pi) to avoid dual-loading.

### 4 GB GPU (GTX 1050 Ti, LXC) — measured

LXC container running CUDA on a GTX 1050 Ti (4 GB VRAM). Same external `qwen3:4b` judge as the other tiers → directly comparable.

LoCoMo measurements (qwen3:4b via Ollama with CUDA, all with `--adjacent-turns 2 --retrieval-top-k 10`):

| Config | Ext Judge | Δ vs LXC baseline |
|---|---|---|
| **adj=2 + k=10 + RRF fusion** | **0.530** | flat (+0.000) |
| adj=2 + k=10 + boost fusion (baseline) | 0.530 | — |

**0.530 on a 4 GB GPU is 0.027 below the 12 GB leader (0.557)** — a remarkably small gap given the VRAM is one third and the generator is a 4 B model not a 9 B. Strong confirmation that taosmd's architecture scales down to consumer-tier GPUs without a quality cliff.

Notes on what this tells us:

- **Predictions held.** The earlier "expected best config" for this tier was `adj=2 + top-k=10`, with `--llm-query-expansion` skipped — exactly what the recipe ran. No surprises against architecture.
- **RRF vs boost is flat at 4 GB / qwen3:4b.** Same direction as the 9 B finding ("RRF alone regresses on adj=2") but smaller magnitude — neither helps nor hurts at this tier with this generator. The leader recipe's RRF lift comes from the *combination* with k=20 + llm-exp, not RRF alone.
- **Operational defaults below stay correct as written.**

- **Generator (best fit)**: `qwen3:4b` at Q4 (~2.5 GB VRAM). Fits with ~1.5 GB headroom for KV cache. `qwen3:2b` is the fallback if context budget is tight.
- **Skip**: `--llm-query-expansion` (extra LLM call too costly at this tier), `--multihop-decompose` (regresses anyway).
- **Best measured LoCoMo config**: `--adjacent-turns 2 --retrieval-top-k 10`. Confirmed at 0.530.
- **Embedder, reranker, judge**: same as larger tiers (CPU ONNX). 4 GB VRAM is for the generator only.

### Raspberry Pi 4 (8 GB, no NPU, no GPU) — not measured (extrapolation only)

CPU-only inference is going to hurt. Realistic only for low-throughput personal use, not benchmark-grade workloads.

- **Generator**: `qwen3:1.7b` or smaller at Q4_K_M, accept slow per-call latency (likely 20–60 s/turn).
- **Embedder, reranker**: ONNX on CPU works fine — the small models are designed for this.
- **Skip**: every retrieval flag that adds an LLM call (`--llm-query-expansion`, `--multihop-decompose`).
- **Expected best LoCoMo config**: `--adjacent-turns 1 --retrieval-top-k 10`. Adj=2 might over-budget the small generator's context — needs measurement.
- **Realistic deployment**: use the Pi 4 as a memory-storage / retrieval node in a taOS cluster; offload generation to a peer with more compute.

### Cluster (taOS, multi-machine)

- **Memory storage tier** (verbatim archive, vector DB, KG): runs on the Pi 4 or any cheap node. Lightweight.
- **Retrieval tier** (embedding + rerank): CPU-bound, runs on any node with the ONNX models.
- **Generation tier**: the GPU/NPU node. Pi 5 Plus, 1050 Ti, 3060, etc., depending on tier.
- **Judge tier**: any node with enough VRAM for `qwen3:4b`. Don't share with the generator.

Hardware-tier validation as we run new measurements lands in this doc; the live experimental log is in [`docs/specs/2026-04-19-locomo-scorecards.md`](specs/2026-04-19-locomo-scorecards.md).

---

## Librarian layer — vocabulary-gap benchmark

Three-axis harness on long-horizon sessions (60 turns, fact buried at turn 5). Axis C measures the vocabulary-gap case: query and fact use different surface words (e.g. query *"code editor"*, fact *"Neovim lua config done"*).

**2026-04-15, gemma4:e2b (5B) on Fedora host:**

| Config | Composite | recall@lag25 | recall@lag50 |
|--------|-----------|-------------|-------------|
| Vector-only | 0.752 | 30% | 30% |
| Full pipeline (+ cross-encoder) | 0.752 | 30% | 30% |
| **Full + Librarian** | **0.810** | **45%** | **55%** |

**+15.4% on the vocabulary-gap axis.** The cross-encoder alone adds nothing when the target fact is excluded from the candidate pool — only the Librarian's expansion bridges category→specific-name gaps. Preliminary result on one class of retrieval failure; LoCoMo-based staleness and multi-store-routing benchmarks are in progress.

## Reproducing benchmarks

```bash
# Full LongMemEval-S (500 questions, Judge)
python benchmarks/longmemeval_runner.py

# Recall@5 only
python benchmarks/longmemeval_recall.py

# Per-category breakdown
python benchmarks/longmemeval_granularity.py
```

All numbers pin a commit SHA, model, and dataset path in the output JSON under `meta`. If a number can't be re-run to the same value on the same commit + model, it doesn't ship in this document.
