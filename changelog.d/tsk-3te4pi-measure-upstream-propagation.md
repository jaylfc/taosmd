### Fixed

- Keyed `measure_at_boundary` flip detection on `resets_at` (window
  identity) instead of `utilization`, which could not distinguish a
  rollover from ordinary account consumption on a shared account.
- Started the polling budget (`max_wait`) when the boundary arrives
  instead of at process start, so a far-future boundary no longer
  exhausts the budget before any poll occurs.  The pre-boundary
  baseline (`pre_resets_at`, `pre_util`) is now sampled before the
  boundary, making a propagation faster than the first poll measurable.
- Aligned the evidence file's `target_location` and procedure text with
  the code: `resets_at` is the flip trigger, and the helper is described
  as out-of-repo instead of citing a non-existent `scripts/resume_arm_time.py`.
- Split the flip-detection test into ARM A (consumption without
  rollover) and ARM B (rollover with flat utilization) as disagreement
  controls.
