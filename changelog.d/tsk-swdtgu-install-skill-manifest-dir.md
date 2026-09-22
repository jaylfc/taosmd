### Fixed
- `taosmd install-skill` now handles non-empty manifest directories on both the plain and `--force` arms by removing the directory tree before writing the manifest file.
- The install is now atomic: the manifest is written to a temp file before copying skill files, so a manifest write failure leaves `SKILL.md` unchanged (no half-applied install).
- Both arms refuse unforced downgrades and local edits; `--force` overwrites cleanly.
- Removed the auto-force fallback from `scripts/install-client.sh` and `scripts/install-client.ps1` so a refused install is not silently clobbered.
