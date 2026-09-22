### Fixed
- Corrected stale comments in `taosmd/http_server.py` and `docs/specs/a2a-delivery-v2.md`
  that claimed A2A read receipts were not yet implemented. The receipts subsystem
  has shipped: `POST /a2a/receipts`, `PATCH /a2a/receipts`,
  `GET /a2a/messages/{id}/receipts`, `GET /a2a/receipts`, and
  `POST /a2a/admin/prune-receipts` are live,
  keyed by `(message_id, agent_id)`. `unread_count` remains omitted because the
  receipts store does not compute a per-message unread aggregate. Missing rows and
  `seen_at IS NULL` are documented as distinct from "not seen" to prevent clients
  from treating absence as false.
