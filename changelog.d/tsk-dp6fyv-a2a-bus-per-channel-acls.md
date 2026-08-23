### Added

- Per-channel ACLs for the A2A bus: channel -> {read: [identities|*], post: [identities|*]}, with default `*` for all existing channels (zero behavior change on deploy).
- Admin management endpoint (`admin_token`) to set/inspect a channel's ACLs.
- Enforcement on both read (GET messages) and post (POST send) using the caller's verified identity. Identity verification via registry JWT is implemented for ACLed channels; raw-LAN posts to ACLed channels are rejected when identity is not verified.
- First-use runtime config to restrict agent-rules to the lead identities (taOS-dev, website-dev, taosmd-dev, hermes, taosc-dev) as config, not code.