### Fixed

- Percent-decode the `{thread}` path segment in all three thread-membership routes (`GET /a2a/threads/{thread}/members`, `POST /a2a/threads/{thread}/members`, `DELETE /a2a/threads/{thread}/members/{principal}`) via `unquote()`, and percent-encode it with `urllib.parse.quote(thread, safe='')` in the matching `RemoteClient` methods, so thread names containing spaces (or other reserved characters) are matched correctly instead of silently returning an empty member list.
