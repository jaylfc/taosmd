### Added
- Dense embeddings now stored as base64-encoded little-endian float32 instead of JSON text (~4x smaller on disk). In-place migration converts existing JSON stores on first open.
