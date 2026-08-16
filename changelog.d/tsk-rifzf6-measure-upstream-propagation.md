### Fixed
- Verified upstream usage-window rollover propagation is 10.0s on average
- Kept MIN_LEAD_SECONDS at 30 as current value is adequate
- Rationale: Measured propagation (10.0s) is less than current value
  (current value is safe but potentially conservative)
