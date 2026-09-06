### Fixed

- URL-decode the principal path segment in `DELETE /a2a/threads/{t}/members/{principal}` via `unquote()`, and percent-encode it in `RemoteClient.a2a_remove_member` via `quote()`, so handles containing `@`, `/`, `#`, `?`, or space are removed correctly instead of returning a silent success.
- Forward `data_dir` as a keyword from `service.a2a_threads` to `RemoteClient`, matching the four membership service functions.
- Remove inline schema DDL from `MembershipStore._init_schema` so `migrations.migrate()` owns table and index creation exclusively.
- Remove dead code: `is_principal_member`, `has_ownership`, and `count_active_members` had zero production callers.
- Drop a duplicated `await archive.close()` in `MembershipStore.archive_membership_removed`.
