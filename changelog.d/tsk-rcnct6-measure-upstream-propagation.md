### Fixed

- Replaced the mock-API simulation in the upstream usage-window rollover
  propagation measurement with a harness that polls the real Anthropic
  usage API at `api.anthropic.com/api/oauth/usage`.  Evidence is now
  committed to `benchmarks/results/` instead of `/tmp`, with no hardcoded
  paths, no external-file mutation, and `MIN_LEAD_SECONDS` left labelled
  UNMEASURED pending a live measurement.
