### Fixed

- `taosmd/docs/a2a-comms.md` now correctly documents that a cross-agent read on a missing or unknown-sender message returns 403, not 404, on `GET /a2a/receipts`.
