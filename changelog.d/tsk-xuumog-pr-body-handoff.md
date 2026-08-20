### Fixed
- Exec PR body generator (`/home/jay/.taos-team/executor.sh`) computes the `Files:` block from an empty merge base, producing wrong diffs in PR descriptions. Reported as hand-off to the executor maintainer; fix is to compute `_MERGE_BASE` before the `--body` string and reference it as a real shell variable.
