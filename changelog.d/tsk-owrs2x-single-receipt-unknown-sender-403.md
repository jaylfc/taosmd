### Fixed

- `GET /a2a/receipts` now returns 403 when the message sender is unknown and the caller is requesting a receipt for a different agent, preventing cross-agent receipt leaks on never-sent message IDs.
