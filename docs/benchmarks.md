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
| **taosmd** | qwen3.5:9b | k=20 + adj=1 + llm-exp | **0.509** | full stack on 9B — matches Letta/LangMem/OpenAI-memory band (who use gpt-4o-mini) |
| **taosmd** | gemma4:e2b | adj=2 | 0.499 | 5B best — architecture-heavy config on our 12 GB GPU |
| **taosmd** | qwen3.5:9b | adj=1 | 0.481 | 9B + adj=1 only |
| **taosmd** | gemma4:e2b | k=20 + adj=1 + llm-exp | 0.482 | 5B stack |
| **taosmd** | gemma4:e2b | adj=1 (C3) | 0.465 | baseline adjacent-turns win |
| **taosmd** | gemma4:e2b | baseline (prompt-opt) | 0.410 | reference point |
| MemPalace | gemma4:e2b | chromadb + MiniLM | 0.336 | same generator + same dataset, different architecture |
| mem0 | gemma4:e2b | chromadb + nomic-embed, infer=False | 0.060 | same generator + same dataset, different architecture |

All taosmd rows on the same commit series (`feat/locomo-param-configs` merged to master). Every row pins its config, generator, judge, and dataset.

### Key architectural findings

- **`adjacent_turns` is the single biggest lever.** Stitching ±1 turn around each retrieved hit took us from 0.410 → 0.465 (+0.055). adj=2 took it to 0.499 on 5B.
- **Stacking is adj-dependent.** At adj=1, adding k=20 and llm-exp compound cleanly (+0.017 → 0.482). At adj=2 the same k=20 addition *regresses* (0.499 → 0.477). Context token budget has a sweet spot; beyond it, more candidates drown the signal.
- **Stacking scales with model size.** The same `k=20 + adj=1 + llm-exp` stack gains +0.017 at 5B and **+0.028 at 9B**. Larger generators use the wider retrieval surface instead of being overwhelmed by it.
- **Generator size alone is a weak lever.** qwen3.5:9b + k=20 (0.458) is *worse* than gemma4:e2b + adj=1 (0.465). Doubling parameters gives ≤ +0.005 unless architecture scales with it.
- **Multihop decomposition regresses at 5B** (0.317, -0.093 vs baseline). Sub-query retrieval surfaces lower-quality chunks when the decomposer is a 5B model; `adjacent_turns=1` captures the "need more context" signal without the extra LLM call.
- **Date-format swap and LLM query expansion** are marginal in the presence of adj=1.

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

### Methodology disclosures

- **Dataset**: LoCoMo-10, 1540 QAs, categories 1–4. Adversarial reserved.
- **Same prompt** across every system (`ANSWER_PROMPT` in `benchmarks/locomo_runner.py`).
- **External judge**: `qwen3:4b`, temperature 0.0, via `locomo_rescore_streaming.py`.
- **No cherry-picking**: 100 % coverage on every rescore, 0 errors. Retractions (one prediction miss, one think=false-on-judge corruption) are documented in `docs/specs/2026-04-19-locomo-scorecards.md`.
- **`think=false` on generator** for Qwen3/3.5/3.6 (PR #42). A `thinking_mode_on` control run is queued to measure whether chain-of-thought changes the result.

Full timeline, per-category breakdowns, and provenance log: `docs/specs/2026-04-19-locomo-scorecards.md`.

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
