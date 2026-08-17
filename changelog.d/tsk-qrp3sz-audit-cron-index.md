### Fixed

Restructured `.claude/audit-cron-prompt.md` from a single 23k-token unbroken
paragraph into a short always-read STEP INDEX plus per-step DETAIL sections, so
the hourly re-read costs a fraction of the original while preserving every rule,
warning, and measured number verbatim. STEP 4 also documents the metadata-only
thread index proof (42,532 vs 912,681 bytes, ~21.5x reduction) so thread-name
lookups no longer require a full body read.
