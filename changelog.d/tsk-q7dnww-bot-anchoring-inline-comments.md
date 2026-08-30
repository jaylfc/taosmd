### Fixed

- `check_bot_anchoring.sh` now fetches PR reviews via `gh api graphql` so the
  `comments.totalCount` field is available, replacing the `includesCreatedEdit`
  heuristic that `gh pr view --json reviews` never populated with inline
  comment counts. A bot review with an empty body and only inline comments is
  now correctly detected as substantive.
