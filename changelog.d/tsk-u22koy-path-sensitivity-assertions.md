### Fixed
- `tests/test_resume_arm_time.py` adds two path-sensitivity assertions to `test_system_crontab_block_names_usr_bin_python3` that verify the emitted cron block names `/usr/bin/python3` with the `SCRIPT` constant, independent of `_HELPER_PATH`, so a mutated helper path cannot silently pass the test.
