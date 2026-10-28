### Security

Per-channel A2A ACLs, part 2 of 4: `GET /a2a/messages` (json and ndjson, with or without `?thread=`) and `GET /a2a/stream` now leave out messages from channels the caller may not read. The caller is identified only by a verified registry token; a caller with no verified token sees nothing from a restricted channel, a malformed ACL entry closes the channel, and channels with no ACL stay open. The listing scans at most five times the requested limit, and the stream moves its poll cursor past rows it skipped.
