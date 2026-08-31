### Fixed

- A2A per-channel ACL: post allowlist now defaults to open when not explicitly set, so restricting reads no longer bricks all posting on that channel.
- A2A per-channel ACL: malformed `acls` configuration sections (non-dict values) now fail closed instead of falling through to a wildcard allow.
- A2A per-channel ACL: messages and stream feeds without a thread filter now fetch all rows before ACL filtering so public rows are not starved by restricted ones.
