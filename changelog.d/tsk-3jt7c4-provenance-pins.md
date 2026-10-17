### Fixed
- Provenance collection: `file_sha256()` now raises `FileNotFoundError` for non-existent paths (instead of returning `None`)
- Provenance collection: `model_dir_digests()` now raises `ValueError` for non-existent or non-directory paths (instead of returning `{}`)
- Provenance collection: `git_commit` is set to `"unknown"` and `git_dirty` to `None` when git operations fail
- Added tests for all provenance fields to ensure they contain concrete expected values or types
- Added tests for git-absent case (outside any git repository)
- Added tests to verify provenance key is written in benchmark results