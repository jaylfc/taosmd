### Fixed
- Gate `/a2a/ack` on the message's channel read ACL: a caller who cannot read a restricted channel may not ack its messages.
- Require admin token for `/a2a/alarms/{key}/clear`: only callers with the configured admin token can re-arm any alarm key.