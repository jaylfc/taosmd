### Added

- New own-data loader (`benchmarks/e048/owndata.py`) for loading private lab screenshots and metadata with built-in privacy guard
- Comprehensive test suite (`tests/test_e048_owndata.py`) covering data loading validation and recall metrics
- Privacy protection ensuring own-data directories never resolve inside repository git trees
- Validation for data format compliance, duplicate detection, and proper error handling
