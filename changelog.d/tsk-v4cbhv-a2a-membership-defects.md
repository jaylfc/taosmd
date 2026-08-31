### Fixed

- Forward `data_dir` as a keyword from `service.a2a_create_thread`, `a2a_list_members`, `a2a_add_member`, and `a2a_remove_member` to `RemoteClient`, fixing `TypeError` on every remote call.
- Map `PermissionError` from membership ownership denials to HTTP 403 in the HTTP server, matching the registry-auth pattern.
- Assert that `archive_membership_created` and `archive_membership_removed` actually write archive events; a no-op recording now fails the new `test_archive_event_lands_on_add_and_remove`.

### Added

- Documented the four new thread-membership endpoints in `taosmd/docs/a2a-comms.md`.

### Notes

- No read path is gated by membership yet: nothing outside the four new service functions reads the `MembershipStore`. The A2A read API (`/a2a/messages`, `/a2a/threads`, `/a2a/stream`, `/a2a/mentions`) does not consult membership.
