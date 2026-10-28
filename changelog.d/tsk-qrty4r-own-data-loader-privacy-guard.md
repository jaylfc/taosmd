### Added

- Implement own-data loader and privacy guard for E-048 benchmark:
  - `assert_outside_repo()` prevents loading from git trees
  - `load_owndata()` loads and validates private lab screenshots
  - Git tree detection for both `.git` directory and `.git` file
  - Path resolution validation prevents directory escapes
  - Duplicate ID detection across items, texts, and queries
  - Required keys validation for all file formats
  - qrels.tsv format validation (3 tab-separated fields, rel field = "1")
  - Query and qrel reference validation
  - SHA256 hash calculation of qrels.tsv raw bytes
  - Comprehensive error messages for all validation failures