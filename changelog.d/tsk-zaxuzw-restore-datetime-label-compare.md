### Fixed

- Restored full-datetime compare (strptime(sd, fmt) == latest_dt) at benchmarks/longmemeval_runner.py:704, which #755 had changed to a date-only compare. With the latest session at 18:00, a 10:00 session on the same day is now correctly labelled "historical" instead of "most recent". Added a same-day test to guard against regression.
