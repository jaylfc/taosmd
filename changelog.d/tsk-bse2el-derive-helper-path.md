### Fixed

- `scripts/resume_arm_time.py` now derives its own path from `os.path.realpath(__file__)` instead of hardcoding an out-of-repo `/home/jay` path, so the crontab lines it emits self-reference the canonical in-repo script. The arming marker and the firing marker are constructed from the same expression, preventing one-shot lines from re-firing next year if they diverge.
