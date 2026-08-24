### Fixed
- Tightened the witness-gate near-miss detector so prose carrying a plain colon
  (without the ``::`` payload) is no longer flagged, while de-marked (zero-width)
  and malformed markers still are.
- Restored the discriminating Arm D assertion in
  ``tests/test_witness_gate.py::TestWitnessGateIntegration::test_near_miss_regex_spares_prose_arms``
  that was dropped from PR #359, ensuring the gate cannot silently regress.
