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

### Leaderboard (external `qwen3:4b` judge, same dataset + same prompt)

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



This is the LongMemEval-S 97.0% reference stack. LoCoMo on this hardware is **not yet measured** (planned).

- **Generator**: `Qwen3-4B` via `rkllama` on the RK3588 NPU (~17 s/turn). Exact stack used for the 97% claim.
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
