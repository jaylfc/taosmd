### Fixed
- Provenance collection: `file_sha256()` raises `FileNotFoundError` for non-existent paths (instead of returning `None`)
- Provenance collection: `model_dir_digests()` raises `ValueError` for non-existent or non-directory paths (instead of returning `{}`)
- Provenance collection: `git_commit` is set to `"unknown"` and `git_dirty` to `None` when git operations fail (including non-zero return codes)
