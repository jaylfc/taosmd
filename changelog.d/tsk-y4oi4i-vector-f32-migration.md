### Added
- Dense embeddings now stored as base64-encoded little-endian float32 instead of JSON text (~4x smaller on disk). In-place migration converts existing JSON stores on first open.

### Fixed
- Migration now checks for store-mode conflicts (wrong embedder, binary-quant, late-interaction) before converting; conflicting stores raise StoreModeMismatch and are left byte-identical.
