### Fixed

- Pinned identity blocks for `longmemeval_oracle.json` and `longmemeval_s_cleaned.json` in
  `benchmarks/data/README.md`, documenting what each file is, its relationship to
  `longmemeval_s_full.json`, question count, byte size, sha256, and how to obtain it.
- Fixed the README "Used by" sentence to name the actual loaders per dataset file instead of
  claiming blanket coverage across all `longmemeval_*` runners.
- Made dataset paths overridable via environment variables in
  `benchmarks/longmemeval_runner.py` (`LONGMEMEVAL_ORACLE_DATA_PATH`),
  `benchmarks/recall_v2_benchmark.py` (`LONGMEMEVAL_ORACLE_DATA_PATH`),
  `benchmarks/longmemeval_recall.py` (`LONGMEMEVAL_ORACLE_DATA_PATH`), and
  `benchmarks/longmemeval_granularity.py` (`LONGMEMEVAL_CLEANED_DATA_PATH`), so experiments
  can point at verified copies without editing code.
- Added `benchmarks/gate_identity.py` gate that FAILS when any `benchmarks/*.py` loads a
  `data/*.json` filename without an identity block in `benchmarks/data/README.md`. A
  throwaway loader was used to demonstrate the gate FAILS (RED), and removal of the loader
  demonstrated the gate PASSES (GREEN), confirming the gate is meaningful and not trivially
  satisfiable from zero data.