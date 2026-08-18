### Fixed

- Replaced the out-of-repo `resume_arm_time.py` at `/home/jay/.taos-fleet-tools/resume_arm_time.py` with a symlink to the canonical repo copy at `scripts/resume_arm_time.py`, ensuring `do_fire` always posts `[RESUME DUE]` to the A2A bus and never silently falls back to the old "log and remove" implementation.