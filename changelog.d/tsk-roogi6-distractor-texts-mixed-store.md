### Added
- E-048 retrieval harness now supports distractor texts via TAOSMD_BENCH_DISTRACTOR_TEXTS environment variable. A JSONL file with {"id": str, "text": str} lines is loaded and its SHA256 of the **entire file bytes** is computed. Each distractor is embedded as a ranked item that crowds out captioned items in the top-k results but carries no session_id and never counts as a hit. The benchmark records distractor_in_topk_mean alongside recall hits.

### Changed
- `benchmarks/longmemeval_enhanced.py`:
  - `arm_config_from_env` now loads the distractor file (via module-level cache), populates cfg["distractor_texts"] with path, sha256, and count, and raises ValueError on missing files, missing keys, or duplicate ids.
  - `_load_distractor_texts` computes SHA256 of the **entire file bytes** (not per-line), and caches loaded texts in a module-level variable to avoid reloading.
  - `run_question` adds distractors after the session ingest loop, counts results with distractor_id metadata, and returns (recall_hit, distractor_count).
  - `run_benchmark` captures distractor_in_topk_mean and writes it to results.
  - `run_benchmark` also clears the distractor embedding cache between runs to avoid stale embeddings.

### Fixed
- `benchmarks/test_distractors.jsonl` format to be valid JSONL (removed header line).

### Tests
- Added tests for `arm_config_from_env` with distractor texts: unset, valid file, missing file, missing keys, duplicate ids.
- Added test for `run_question` with distractors: verifies distractor rows are added with distractor_id metadata.
- All existing tests continue to pass, maintaining backward compatibility when distractor texts are unset or empty.
