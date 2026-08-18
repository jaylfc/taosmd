### Fixed
- The witness-token gate now reports de-marked or malformed ``# WITNESS``
  markers as violations instead of silently ignoring them. The gate's own
  docstring examples are explicitly exempted via a named exclusion. A
  maintainer who copies the documented example with real values now gets a
  failing gate instead of a silent false-positive.
