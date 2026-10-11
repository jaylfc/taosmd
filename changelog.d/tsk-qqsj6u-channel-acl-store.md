### Added

Per-channel A2A ACLs, part 1 of 4: an admin can set read and post allowlists
for a channel (`POST /a2a/admin/set-channel-acl`, read back with
`GET /a2a/admin/channel-acl`, both behind the admin token), and
`/a2a/send` and `/a2a/import` now refuse a post to a restricted channel with
a 403 before anything is written. The caller is identified only by a verified
registry token, never by the message's `from` field; a caller with no
verified token is refused on a restricted channel. Channels with no ACL stay
open to everyone, so existing deployments are unchanged. A malformed ACL entry
closes the channel. Read filtering on the listing and stream routes follows in
later parts.
