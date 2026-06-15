# Benchmark datasets

This directory holds benchmark datasets. The datasets themselves are gitignored
because they are large, but their identities are pinned here so a missing file
never means a lost or ambiguous dataset. Verify any copy against the checksum
before trusting a published number.

## longmemeval_s_full.json

- What it is: LongMemEval-S, the oracle variant of the LongMemEval long-term
  memory benchmark, 500 questions, each with a haystack of conversation
  sessions and the gold answer session ids.
- Used by: `benchmarks/longmemeval_enhanced.py` and the other
  `longmemeval_*` runners. It is the source of the published 97.0% Recall@5
  headline (see `benchmarks/REPRODUCE-longmemeval.md`).
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
