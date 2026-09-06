### Fixed

- Percent-encode the `{thread}` path segment in `RemoteClient.a2a_thread_messages` with `urllib.parse.quote(thread, safe='')` and percent-decode it via `unquote()` in the `GET /a2a/threads/{thread}` and `GET /a2a/threads/{thread}/messages` dispatch branch, so thread names containing spaces, hashes, slashes, or non-ASCII characters are matched correctly instead of raising `InvalidURL`, returning silent empty lists, or raising `UnicodeEncodeError`.
