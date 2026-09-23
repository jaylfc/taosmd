### Added

- Per-channel A2A ACL system: `GET /a2a/admin/set-channel-acl` admin endpoint,
  `config.get_acl` / `config.set_acl` helpers, and enforcement wired into
  `/a2a/messages`, `/a2a/stream`, `/a2a/send`, `/a2a/mentions`, `/a2a/inbox`,
  `/a2a/threads`, `/a2a/channels`, `/a2a/members`, and `/a2a/census`.

### Fixed

- Bounded archive scan in `GET /a2a/messages`: pages the feed up to
  `limit * 5` rows instead of fetching unbounded.
- SSE poll cursor advances past rows the caller cannot see, preventing a frozen
  `since` when all rows in a batch are denied.
- Denied POST bodies are no longer persisted to the archive (deny-on-effect,
  not deny-on-status).
- `config.set_acl` merges omitted dimensions instead of clobbering them.
- `POST /a2a/admin/set-channel-acl` rejects a non-boolean `clear` value with 400.
- `/a2a/inbox` and `/a2a/mentions` now apply the same channel read filter as
  the other A2A endpoints.
