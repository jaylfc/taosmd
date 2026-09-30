### Fixed
- `tests/test_receipts.py::test_get_single_receipt_not_found` now reads the caller's own receipt (`agent=alice`) instead of a cross-agent read, matching the fail-closed rule where cross-agent reads on unknown-sender messages correctly return 403.
