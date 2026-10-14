### Security

Per-channel A2A ACLs, part 3 of 4: `GET /a2a/channels`, `/a2a/census`, `/a2a/members`, `/a2a/mentions`, `/a2a/inbox` and `/a2a/inbox/unhandled` now leave out channels the caller may not read (name, members, counts and sender activity included), and `/a2a/members?channel=` on such a channel returns 403. The caller is identified only by a verified registry token; the `reader` and `consumer` query values never grant access.
