### Fixed
- `taosmd install-skill` now handles a non-empty directory at the manifest path on both the plain and `--force` arms, removing the obstruction instead of raising `OSrror [Errno 39]`.
- A failed manifest write no longer leaves `SKILL.md` advanced; the install is rolled back so the on-disk skill and manifest stay coherent.
- `scripts/install-client.sh` and `scripts/install-client.ps1` no longer auto-clobber local edits by chaining `install-skill --force` on refusal.
