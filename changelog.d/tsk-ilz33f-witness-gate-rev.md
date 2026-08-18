### Fixed
- Widened the witness-token gate scan to include ``scripts/**/*.py`` so that
  witness markers in gate scripts are checked prospectively, closing the gap
  where any future marker in ``scripts/`` would not have been verified.
  Fixed the sibling-gate citation in
  the docstring from ``normalise_handle_gate.py`` (which also scans ``tests/``)
  to ``check_deleted_symbols.py`` (which is ``taosmd/``-only, matching the
  exclusion). De-marked illustrative marker examples in the gate docstring and
  the test module docstring so the live-marker count is 0.
