### Fixed: Receipt identity docstring claims verified

- Fixed docstring of `_get_authenticated_agent_id` to no longer claim "verified" registry token, matching the code's actual behavior of using `_registry_verifier.authorize()` which verifies when a registry is configured.

### Added: A2A read receipts system

- Added `taosmd/receipts.py` with `ReceiptStore` class supporting `record_delivered`, `record_seen`, `get_receipts_for_message`, `get_receipt`, and `prune` operations.
- Added five new remote methods to `taosmd.remote.RemoteClient`: `a2a_record_delivered`, `a2a_record_seen`, `a2a_get_receipts`, `a2a_get_receipt`, `a2a_prune_receipts`.
- Added service wrappers in `taosmd.service` for all receipt operations.
- Added HTTP endpoint handlers in `taosmd.http_server` for `POST /a2a/receipts`, `PATCH /a2a/receipts`, `GET /a2a/receipts`, `GET /a2a/messages/{id}/receipts`, and `POST /a2a/admin/prune-receipts`.
- Added `ReceiptStore` integration in `taosmd.api._ensure_stores`.
- Added comprehensive test coverage in `tests/test_receipts.py`.