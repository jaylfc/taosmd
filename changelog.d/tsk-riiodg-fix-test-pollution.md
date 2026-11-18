### Fixed
- Fixed test pollution in `tests/test_auto_setup_cron.py` where fake `subprocess.run` was assigned to stdlib module and never restored, breaking 56 later tests. Now uses `monkeypatch.setattr` for proper cleanup.
