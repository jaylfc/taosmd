### Added
- A2A read receipts: a `ReceiptStore` keyed by `(message_id, agent_id)` tracking a
  `delivered_at` mark that never moves once set and a `seen_at` mark that only ever
  goes from unset to set, with a registered migration, service wrappers and
  `_ensure_stores` integration.
- Receipt endpoints `POST /a2a/receipts`, `PATCH /a2a/receipts`,
  `GET /a2a/messages/{id}/receipts`, `GET /a2a/receipts` and
  `POST /a2a/admin/prune-receipts`, plus `do_PATCH` so the PATCH route dispatches.
  Delivered marks are also written for identified SSE subscribers.
- Receipt identity is taken from the verified registry token's `sub` claim via
  `_get_authenticated_agent_id`, never from the request body. A request with no
  token, or one signed by a key the registry does not know, gets a 401 and writes
  no row. Reads are not authenticated.
- Five `RemoteClient` methods (`a2a_record_delivered`, `a2a_record_seen`,
  `a2a_get_receipts`, `a2a_get_receipt`, `a2a_prune_receipts`) so remote clients
  mirror the local service API.
