### Fixed

- Pinned the review-state filter in `scripts/merge-gate/check_bot_anchoring.sh` with a `tests/fixtures/merge_gate/pr_bot_dismissed.json` fixture and `test_bot_dismissed_at_head_fails`: a bot review in a `DISMISSED` state at head is now asserted to make the gate exit rc=10. The filter was behaviourally live but unpinned, so a mutant replacing it with `if False:` survived and left the whole test set green.
