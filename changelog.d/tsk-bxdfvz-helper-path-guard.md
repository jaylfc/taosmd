### Fixed

- `scripts/resume_arm_time.py` now refuses to emit the system-crontab block when `_HELPER_PATH` resolves under a temp directory or inside a linked git worktree, naming the reason instead of pinning an ephemeral path that would orphan the self-removal cron line.
