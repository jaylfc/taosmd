### Fixed
- `taosmd install-skill` now handles a non-empty directory at the manifest path on both the plain and `--force` arms, removing the obstruction instead of raising `OSError [Errno 39]`.
- A failed manifest write no longer leaves `SKILL.md` advanced; the install is rolled back so the on-disk skill and manifest stay coherent.
