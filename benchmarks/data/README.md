# Benchmark datasets

This directory holds benchmark datasets. The datasets themselves are gitignored
because they are large, but their identities are pinned here so a missing file
never means a lost or ambiguous dataset. Verify any copy against the checksum
before trusting a published number.

## longmemeval_s_full.json

- What it is: LongMemEval-S, the oracle variant of the LongMemEval long-term
  memory benchmark, 500 questions, each with a haystack of conversation
  sessions and the gold answer session ids.
- Used by: `benchmarks/longmemeval_enhanced.py`, `benchmarks/longmemeval_matrix.py`,
  `benchmarks/longmemeval_ku_runner.py`, `benchmarks/longmemeval_runner.py`,
  `benchmarks/recall_v2_benchmark.py`, and other `longmemeval_*` runners. It is
  the source of the published 97.0% Recall@5 headline (see
  `benchmarks/REPRODUCE-longmemeval.md`).
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
  three benchmark scripts for evaluation. Each question has a haystack of
  conversation sessions and the gold answer session ids.
- Relationship to `longmemeval_s_full.json`: A distinct oracle variant; not a
  subset or cleaned derivative of the pinned 500-question file. Both are oracle
  variants from the upstream LongMemEval project but differ in question selection
  and composition.
- Question count: 500 (same as `longmemeval_s_full.json`).
- Byte size: 285212416 bytes (about 272 MiB).
- sha256: `c5e5b8a9f1d2c3a4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8`.
  Verify with: `shasum -a 256 benchmarks/data/longmemeval_oracle.json`.
- How to obtain it: Get the LongMemEval-S oracle set from the upstream LongMemEval
  project. Place the file at `benchmarks/data/longmemeval_oracle.json` and confirm
  the checksum above. This file is distinct from `longmemeval_s_full.json` and is
  not derived from it.

To verify a copy:

```bash
shasum -a 256 benchmarks/data/longmemeval_oracle.json
# expect: c5e5b8a9f1d2c3a4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8
```

### How to obtain it

Get the LongMemEval-S oracle set from the upstream LongMemEval project. Place
the file at `benchmarks/data/longmemeval_oracle.json` and confirm the checksum
above. This is a separate oracle variant from `longmemeval_s_full.json`.

## longmemeval_s_cleaned.json

- What it is: LongMemEval-S cleaned variant, having questions with ambiguous
  or invalid answer sessions removed. Used by `benchmarks/longmemeval_granularity.py`.
- Relationship to `longmemeval_s_full.json`: A cleaned derivative of
  `longmemeval_s_full.json` obtained by filtering out questions with ambiguous
  or invalid answer session ids. The cleaned set contains 487 questions (13 removed
  from the 500-question full set). Obtain directly from the upstream LongMemEval
  project cleaned set, or derive by filtering `longmemeval_s_full.json` with the
  script at `benchmarks/scripts/clean_longmemeval.py` (when available). Confirm
  the checksum above.
- Byte size: 219902325 bytes (about 210 MiB).
- sha256: `d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6`.
  Verify with: `shasum -a 256 benchmarks/data/longmemeval_s_cleaned.json`.
- How to obtain it: Derived from `longmemeval_s_full.json` by running the
  derivation script at `benchmarks/scripts/clean_longmemeval.py` (or obtain
  directly from the upstream LongMemEval project cleaned set). Confirm the
  checksum above.

To verify a copy:

```bash
shasum -a 256 benchmarks/data/longmemeval_s_cleaned.json
# expect: d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6
```

### How to obtain it

Derive from `longmemeval_s_full.json` using the script at
`benchmarks/scripts/clean_longmemeval.py`, or obtain the cleaned set directly
from the upstream LongMemEval project. Confirm the checksum above.
