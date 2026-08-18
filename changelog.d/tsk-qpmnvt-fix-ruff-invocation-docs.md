### Fixed
- Replaced broken `python3 -m ruff check` invocations in `docs/agent-jobs/` with the standalone `ruff check` form, and added explicit guidance for agents to report a missing `ruff` binary in the PR body and skip that verification step rather than inventing a substitute.
