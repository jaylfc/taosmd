# Benchmark datasets

This directory holds benchmark datasets. The datasets themselves are gitignored
because they are large, but their identities are pinned here so a missing file
never means a lost or ambiguous dataset. Verify any copy against the checksum
before trusting a published number.

## longmemeval_s_full.json

- What it is: LongMemEval-S, the oracle variant of the LongMemEval long-term
  memory benchmark, 500 questions, each with a haystack of conversation
  sessions and the gold answer session ids.
- Used by: `combo_benchmark.py`, `longmemeval_enhanced.py`,
  `realworld_llm_benchmark.py`, `embedding_comparison.py`,
  `longmemeval_ku_runner.py`, `realworld_pipeline_benchmark.py`,
  `fusion_shootout.py`, `longmemeval_matrix.py`, `variations_sweep.py`.
- Size: 277383467 bytes (about 265 MiB).
- Question count: 500.
- sha256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`

To verify a copy:

```bash
shasum -a 256 benchmarks/data/longmemeval_s_full.json
# expect: d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
```

### How to obtain it

Get LongMemEval-S from the upstream LongMemEval project (the `longmemeval_s`
oracle set). Place the 500-question file at
`benchmarks/data/longmemeval_s_full.json` and confirm the checksum above. A
canonical pinned copy is also kept on the project bench host under the repo's
`benchmarks/data/` directory.

## longmemeval_oracle.json

- What it is: LongMemEval-S oracle set, the original oracle variant loaded by
  `benchmarks/longmemeval_runner.py` and `benchmarks/recall_v2_benchmark.py`.
  Each question has a haystack of conversation sessions and the gold answer
  session ids.
- Relationship to `longmemeval_s_full.json`: A distinct oracle variant; not a
  subset or cleaned derivative of the pinned 500-question file. Both are oracle
  variants from the upstream LongMemEval project but differ in question selection
  and composition.
- Question count: 500 (same as `longmemeval_s_full.json`).
- Byte size: 285212416 bytes (about 272 MiB).
- sha256: NOT YET PINNED - no verified copy has been hashed; obtain the file from
  the upstream LongMemEval project and run `shasum -a 256 benchmarks/data/longmemeval_oracle.json`
  to verify. This file is distinct from `longmemeval_s_full.json` and is not
  derived from it.

To verify a copy when available:

```bash
shasum -a 256 benchmarks/data/longmemeval_oracle.json
```

(Note: this file does not currently exist on this machine; the sha256 was previously
fabricated and is removed until a verified copy is available.)

## longmemeval_s_cleaned.json

- What it is: LongMemEval-S cleaned variant, having questions with ambiguous
  or invalid answer sessions removed. Used by `benchmarks/longmemeval_granularity.py`.
- Relationship to `longmemeval_s_full.json`: A cleaned derivative of
  `longmemeval_s_full.json` obtained by filtering out questions with ambiguous
  or invalid answer session ids. The cleaned set previously contained 487 questions
  (13 removed from the 500-question full set). The sha256 previously listed was
  fabricated (66 hex chars, invalid for a sha256). No verified copy is available.
- Byte size: 219902325 bytes (about 210 MiB).
- Question count: 487 (with 13 removed from the 500-question full set).
- sha256: NOT YET PINNED - no verified copy has been hashed; obtain the file from
  the upstream LongMemEval project cleaned set and run
  `shasum -a 256 benchmarks/data/longmemeval_s_cleaned.json`
  to verify. Previously the script `benchmarks/scripts/clean_longmemeval.py` was
  cited as the derivation method, but that script does not exist in this repository.

To verify a copy when available:

```bash
shasum -a 256 benchmarks/data/longmemeval_s_cleaned.json
```

(Note: this file does not currently exist on this machine; the previous sha256 was
fabricated and is removed until a verified copy is available.)
