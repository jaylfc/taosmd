### Fixed

- A2A per-channel read ACL bypass: `GET /a2a/messages` without a `thread` parameter now filters out messages from channels the caller is not allowed to read instead of returning every message on the bus. `GET /a2a/threads` now filters out restricted channels, `GET /a2a/stream` filters restricted messages in the poll loop, and `GET /a2a/admin/channel-acl` now requires a valid admin token before disclosing the allowlist.
