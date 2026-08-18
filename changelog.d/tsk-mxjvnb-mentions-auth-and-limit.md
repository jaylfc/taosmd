### Fixed
- `/a2a/mentions` now fails closed when `?reader=` does not match the verified token subject, instead of serving the caller's own mentions
- `a2a_mentions_feed` archive query now carries `limit=100_000` so mentions remain visible when the bus exceeds 50 messages
