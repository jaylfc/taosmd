### Security

Per-channel A2A ACLs, part 4 of 4: `GET /a2a/threads` now leaves out threads of channels the caller may not read, `GET /a2a/threads/{thread}/messages` and `GET /a2a/threads/{thread}/members` return 403 for such a channel, and `GET /a2a/messages/{id}/receipts` and `GET /a2a/receipts` return 403 when the message belongs to one. The caller is identified only by a verified registry token.
