### Fixed
- Restored the A2A receipts subsystem (ReceiptStore, registered migration,
  service wrappers, ``_ensure_stores`` integration) that was dropped by PR #326,
  which had left all receipt-write endpoints returning 500.
- Fixed receipt identity: ``agent_id`` is now derived from the verified
  registry token's ``sub`` claim via ``_get_authenticated_agent_id``, never from
  the request body. A forged token produces a 401 with no receipt row written.
- Added ``do_PATCH`` so ``PATCH /a2a/receipts`` routes correctly, and de-duplicated
  the routing table so each receipt path is dispatched exactly once.
- Fixed ``ttl_days`` unit handling in ``POST /a2a/admin/prune-receipts``: days are
  now converted to an absolute epoch timestamp (``time.time() - ttl_days * 86400``)
  instead of being passed through as epoch seconds.
- Added five ``RemoteClient`` methods (``a2a_record_delivered``,
  ``a2a_record_seen``, ``a2a_get_receipts``, ``a2a_get_receipt``,
  ``a2a_prune_receipts``) so remote clients mirror the local service API.
