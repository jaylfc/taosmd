### Fixed

- The durable resume-pair cron now posts `[RESUME DUE]` to the A2A `agent-rules` bus on firing, so a live sibling agent or Jay sees the missed wake instead of a log line no process reads. The helper is version-controlled at `scripts/resume_arm_time.py`, the prompt references the canonical path, and the emitted crontab lines carry the full path-precise marker for exact deduplication.
