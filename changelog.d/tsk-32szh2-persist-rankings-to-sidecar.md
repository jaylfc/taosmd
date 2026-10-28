### Added
- Added `rank_topk()` function to benchmarks/e048/retrieval.py for per-query ranking computation
- Added `rankings_path` parameter to `run()` function to persist per-query rankings to JSONL sidecar file
- Added `--rankings-out` CLI flag to write rankings file and include `rankings_file` and `rankings_sha256` in results payload
- Rankings JSONL format: each line contains query index, caption, true item ID, true rank, and top-10 ranked items with scores
- Updated tests to verify rank_topk function correctness, rankings file format, and CLI integration
