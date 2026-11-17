### Added

- Added `taosmd.secret_spans` with AES-GCM span encoding, 0600 file key provider, and reveal support for raw-archive secret spans.
- Fixed reveal_leaf to correctly handle multiple spans in one leaf, fixed FileKeyProvider.current() first-use race, and added kid format validation in get().
- Corrected kid validation regex to require exactly 16 lowercase hex characters matching secrets.token_hex(8) output.
