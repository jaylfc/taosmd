### Added
- Dense embeddings are stored as base64-encoded little-endian float32 instead of JSON text (about 4x smaller on disk). A store with JSON embeddings is still read as it is and is never converted on open. Conversion is explicit, with `taosmd migrate --convert-vectors`, which takes a verified backup first and checks every row before it commits.

### Fixed
- The conversion checks for store-mode conflicts (wrong embedder, binary-quant, late-interaction) before it changes anything; a conflicting store is refused and left byte-identical.
