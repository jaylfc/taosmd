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
  It is the source of the published 97.0% Recall@5 headline
  (see `benchmarks/REPRODUCE-longmemeval.md`).
- Byte size: 277383467 bytes (about 265 MiB) -- NOT YET PINNED, no verified
  copy located on this machine; retained from prior documentation. Run
  `stat -c %s benchmarks/data/longmemeval_s_full.json` to confirm.
- Question count: 500.
- sha256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
  -- NOT YET PINNED, no copy has been hashed on this machine; obtain the
  file from the upstream LongMemEval project and run
  `shasum -a 256 benchmarks/data/longmemeval_s_full.json` to verify.

To verify a copy when available:

```bash
shasum -a 256 benchmarks/data/longmemeval_s_full.json
```

*(Note: this file does not currently exist on this machine. No copy has been
located to confirm the size or checksum above. Obtain from the upstream LongMemEval
project and confirm the checksum before use.)*

### How to obtain it

Get LongMemEval-S from the upstream LongMemEval project (the `longmemeval_s`
oracle set). Place the 500-question file at
`benchmarks/data/longmemeval_s_full.json` and confirm the checksum above. A
canonical pinned copy is also kept on the project bench host under the repo's
`benchmarks/data/` directory.

## longmemeval_oracle.json

- What it is: LongMemEval-S oracle set, the original oracle variant loaded by
  `benchmarks/longmemeval_runner.py`, `benchmarks/recall_v2_benchmark.py`,
  and `benchmarks/longmemeval_recall.py`. Each question has a haystack of
  conversation sessions and the gold answer session ids.
- Byte size: 15388478 bytes (about 14.7 MiB), measured with `stat -c %s`
- sha256: `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`
  (64 hex, valid). Verify with: `shasum -a 256 benchmarks/data/longmemeval_oracle.json`
- Question count: 500
- The oracle variant has evidence-only haystacks (haystack_session_ids ==
  answer_session_ids EXACTLY for all 500 questions), making it 14.7 MiB against
  `longmemeval_s_full.json`'s claimed 265 MiB. The relationship between this
  file and `longmemeval_s_full.json` is not verified here, as
  `longmemeval_s_full.json` does not exist on this machine.
- How to obtain it: Get the LongMemEval-S oracle set from the upstream LongMemEval
  project. Place the file at `benchmarks/data/longmemeval_oracle.json` and confirm
  the checksum above.

To verify a copy when available:

```bash
shasum -a 256 benchmarks/data/longmemeval_oracle.json
```

*(Note: this file does not currently exist in this repository; three verified
copies exist in sibling repositories with the measurements above. Obtain from the
upstream LongMemEval project and confirm the checksum before use.)*

## longmemeval_s_cleaned.json

- What it is: LongMemEval-S cleaned variant, having questions with ambiguous
  or invalid answer sessions removed. Used by `benchmarks/longmemeval_granularity.py`.
- Byte size: not available (no verified copy located).
- Question count: not available (no verified copy located).
- sha256: NOT YET PINNED - no verified copy has been hashed; obtain the file from
  the upstream LongMemEval project cleaned set and run
  `shasum -a 256 benchmarks/data/longmemeval_s_cleaned.json`
  to verify. Previously the script `benchmarks/scripts/clean_longmemeval.py` was
  cited as the derivation method, but that script does not exist in this repository.
  The byte size and question count previously stated were unverifiable and are
  removed.
- Relationship to `longmemeval_s_full.json`: not available (no verified copy
  located).

To verify a copy when available:

```bash
shasum -a 256 benchmarks/data/longmemeval_s_cleaned.json
```

*(Note: this file does not currently exist in this repository. No verified copy
has been located. Obtain from the upstream LongMemEval project cleaned set.)*
