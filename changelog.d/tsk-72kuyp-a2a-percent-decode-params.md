### Fixed

- `GET /a2a/*` endpoints now URL-decode parameter names before checking them against the allowlist, so a percent-encoded legitimate name like `%73ince` is accepted instead of falsely rejected as unknown.
