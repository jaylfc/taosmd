### Fixed
- Made `unsorted_walk` in `tests/test_e048_siglip2.py` yield a fixed non-sorted file order (`["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"]`) instead of merely reversing filesystem order, ensuring the `provenance()` sort is exercised deterministically.
