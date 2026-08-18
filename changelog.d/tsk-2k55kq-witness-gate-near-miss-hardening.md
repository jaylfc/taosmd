### Fixed
- Tightened the witness-gate near-miss detector so ordinary prose that merely
  mentions WITNESS (without the ``::`` payload) is no longer flagged, while
  de-marked (zero-width) and malformed markers still are.
- Replaced the file-level de-marked-marker exemption in the gate with a
  line-level one, so a genuine de-marked marker appended to
  ``scripts/check_witness_token.py`` is reported while its three documented
  docstring examples are not.
