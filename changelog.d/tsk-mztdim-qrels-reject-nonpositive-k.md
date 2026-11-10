### Fixed
- `benchmarks/e048/qrels.py`: `recall_at_k_qrels()` now raises `ValueError` for non-positive or non-int `k` values and for an empty `ks` tuple, instead of silently returning meaningless scores.
