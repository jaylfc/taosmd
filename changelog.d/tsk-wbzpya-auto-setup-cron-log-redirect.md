### Fixed

- Nightly librarian cron now writes log output to ``<data_dir>/logs/librarian-cron.log`` instead of discarding all output with ``2>/dev/null``. The cron command uses ``>> <q_logfile> 2>&1`` so failures (missing model, bad data dir, import error) are visible in the log file. A ``logs`` subdirectory is created with ``os.makedirs`` before writing the crontab.
