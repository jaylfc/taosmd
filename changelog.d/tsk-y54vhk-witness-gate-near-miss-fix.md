### Fixed
- Tightened the witness-gate near-miss detector so ordinary prose that merely
  mentions WITNESS (without the ``::`` payload) is no longer flagged, while
  de-marked (zero-width) markers that retain a colon after the broken separator
  and other malformed markers containing a colon still are.
- Replaced the file-level de-marked-marker exemption in the gate with a
  line-level one, so a genuine de-marked marker appended to
  ``scripts/check_witness_token.py`` is reported while its three documented
  docstring examples are not.
- Corrected the near-miss detector to catch de-marked markers whose payload
  is missing, fixing a silent regression introduced when the detector was
  tightened to require the ``::`` payload.
