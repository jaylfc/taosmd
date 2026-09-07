### Fixed

- `GET /a2a/mentions` now gates every returned message through the #211 `can_read` guard, so message bodies are disclosed only when the reader is entitled via a mention grant (being mentioned in the thread root) or channel ACL.  This closes the last unfiltered A2A read path: a blanket channel-ACL filter was deliberately NOT applied, as #211 grants read access on mention regardless of channel membership, and filtering would break cross-channel mentions that the design intentionally allows.
- `?limit=` on `GET /a2a/mentions` is now capped at 10,000 at the HTTP layer so `?limit=1000000000` no longer sends an unbounded `LIMIT` to the mention index SQLite query.
