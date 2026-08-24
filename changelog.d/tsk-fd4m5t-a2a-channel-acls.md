### Added

- Per-channel read/post ACLs for the A2A bus. Enforcement derives the acting
  principal from the verified registry token `sub` (via `_registry_verifier.
  authorize`), never from a client-supplied body field. A forged token whose
  signature fails verification is rejected even when its `sub` is in the
  allowlist. Admin endpoints `POST /a2a/admin/set-channel-acl` and
  `GET /a2a/admin/channel-acl` manage per-channel ACLs. Channels without an
  explicit ACL entry remain open (zero deploy behavior change).
