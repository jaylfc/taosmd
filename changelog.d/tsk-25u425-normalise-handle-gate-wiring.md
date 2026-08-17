### Added
- `normalise_handle` gate (`scripts/normalise_handle_gate.py`) is now invoked by CI via `.github/workflows/normalise-handle-gate.yml` and covered by `tests/test_normalise_handle_gate.py`, asserting at most one `def _normalise_handle` (sync or async, at any nesting depth) under `taosmd/` so a duplicate copy landing in another lane is caught rather than silently passing an unrun check.
