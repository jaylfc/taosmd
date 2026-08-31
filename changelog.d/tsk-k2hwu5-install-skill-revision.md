### Fixed
- `scripts/install-client.sh` no longer auto-forces `taosmd install-skill --force` on refusal. The previous `||` chain converted every downgrade or local-edits refusal into a silent overwrite; the script now propagates the non-zero exit so user edits are never clobbered by the installer caller.
- `scripts/install-client.ps1` no longer prints "Skill installed." when `taosmd install-skill` refuses. A `try/catch` around a native command does not fire on non-zero exit in PowerShell, so the previous code silently reported success while nothing was installed; it now checks `$LASTEXITCODE` and exits non-zero on refusal.
- `taosmd install-skill` no longer raises uncaught `IsADirectoryError` or `PermissionError` when the `.taosmd-skill-manifest.json` cannot be written. Both the force arm and the non-force upgrade arm catch those errors, remove or chmod the obstructing path, and retry so the on-disk `SKILL.md` and manifest stay coherent.
- The version regex in `_parse_skill_version` now uses `[ \\t]*` instead of `\\s*` so an empty `version:` line does not consume the next frontmatter line as the version string.

### Notes
- D3: a manifest that is valid JSON but carries no `skill_md_sha256` key is still treated as clean. This is deliberate and pinned by existing test coverage; changing it would alter the documented pre-versioning behaviour and is left as a conscious design choice rather than a silent fix.
