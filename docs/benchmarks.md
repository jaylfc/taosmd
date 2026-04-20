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
