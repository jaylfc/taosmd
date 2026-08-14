# tsk-rf5gwb: Membership identity-stem measurement

Scope: channel membership principals from EVENT_A2A archive rows in a taOSmd data dir.

## Why this exists

Stage 2's mismatch gate and the membership reconciliation migration both key on
normalised identity. The `from`-field measurement is scoped to `from` only.
Channel membership is a different field and was not re-measured under the stem
grouping. Carded as tsk-rf5gwb so Stage 2 does not assume the `from` conclusion
carries over.

## Method

1. Run the measurement tool against a data dir that contains EVENT_A2A rows:

   ```bash
   uv run --extra dev python scripts/measure_membership_stems.py --data-dir /path/to/taosmd/data
   ```

   The tool calls ``service.a2a_channels(data_dir=...)`` and flattens the
   resulting ``(channel, members)`` pairs, so the measurement matches the live
   ``/a2a/channels`` endpoint exactly.

2. The tool computes two stem groupings for each sender:
   - **without mint stripping**: strip `@`, casefold
   - **with mint stripping**: strip `@`, casefold, then strip a trailing
     `-YYYYMMDD-HHMMSS` mint stamp

3. Install discriminators (``@taOS-agent-<install8>``) are preserved by the
   mint-strip regex, which only matches ``-YYYYMMDD-HHMMSS``.  The partial
   unique index on ``(handle) WHERE status='active'`` rejects the second
   insert, so merging two installs would be a measurement bug, not a finding.

## Measured numbers

Run twice against the same data dir to confirm reproducibility:

```bash
uv run --extra dev python scripts/measure_membership_stems.py --data-dir /tmp/tmpn67fr05u/taosmd-test
```

```
Scope: archive EVENT_A2A rows in /tmp/tmpn67fr05u/taosmd-test
Total distinct principals: 11

1. Stems with >1 spelling (no mint stripping): 2
   alice: ['@alice', 'alice']
   bob: ['@bob', 'bob']
   Stems with >1 spelling (with mint stripping): 4
   alice: ['@alice', 'alice']
   bob: ['@bob', 'bob']
   hermes: ['hermes', 'hermes-20260608-153000', 'hermes-20260727-001415']
   taosmd: ['@taOSmd-20260813-001415', 'taosmd', 'taosmd-20260609-153000']

2. Canonical membership entries with bare or @-form twin: 4
   @taOSmd-20260813-001415 -> ['taosmd', 'taosmd-20260609-153000']
   taosmd-20260609-153000 -> ['@taOSmd-20260813-001415', 'taosmd']
   hermes-20260608-153000 -> ['hermes', 'hermes-20260727-001415']
   hermes-20260727-001415 -> ['hermes', 'hermes-20260608-153000']

3. Distinct principals collapsing to one stem (no mint stripping): 2
   alice: ['@alice', 'alice']
   bob: ['@bob', 'bob']

Per-channel breakdown:

  Channel: build
  Total distinct principals: 3
  Stems with >1 spelling (no mint stripping): 0
  Stems with >1 spelling (with mint stripping): 1
    taosmd: ['@taOSmd-20260813-001415', 'taosmd', 'taosmd-20260609-153000']
  Canonical membership entries with bare or @-form twin: 2
    @taOSmd-20260813-001415 -> ['taosmd', 'taosmd-20260609-153000']
    taosmd-20260609-153000 -> ['@taOSmd-20260813-001415', 'taosmd']
  Distinct principals collapsing to one stem (no mint stripping): 0

  Channel: general
  Total distinct principals: 7
  Stems with >1 spelling (no mint stripping): 2
    alice: ['@alice', 'alice']
    bob: ['@bob', 'bob']
  Stems with >1 spelling (with mint stripping): 3
    alice: ['@alice', 'alice']
    bob: ['@bob', 'bob']
    hermes: ['hermes', 'hermes-20260608-153000', 'hermes-20260727-001415']
  Canonical membership entries with bare or @-form twin: 2
    hermes-20260608-153000 -> ['hermes', 'hermes-20260727-001415']
    hermes-20260727-001415 -> ['hermes', 'hermes-20260608-153000']
  Distinct principals collapsing to one stem (no mint stripping): 2
    alice: ['@alice', 'alice']
    bob: ['@bob', 'bob']

  Channel: random
  Total distinct principals: 1
  Stems with >1 spelling (no mint stripping): 0
  Stems with >1 spelling (with mint stripping): 0
  Canonical membership entries with bare or @-form twin: 0
  Distinct principals collapsing to one stem (no mint stripping): 0
```

| Question | Answer |
|---|---|
| Total distinct principals | 11 |
| Stems with >1 spelling, no mint stripping | 2 |
| Stems with >1 spelling, with mint stripping | 4 |
| Canonical entries with bare or @-form twin | 4 |
| Distinct principals collapsing to one stem (no mint) | 2 |

## Key findings

- **hermes** on channel `general` carries three spellings (`hermes`,
  `hermes-20260608-153000`, `hermes-20260727-001415`).  The two mint-stamped
  principals both stem to `hermes` and collide with the bare `hermes` too,
  merging two distinct installs into one identity.
- **build** carries three spellings of `taosmd` (`@taOSmd-20260813-001415`,
  `taosmd`, `taosmd-20260609-153000`).  Both mint-stamped principals have the
  bare `taosmd` as a twin.
- **alice** and **bob** each appear in both `@`-form and bare-form on
  `general`.  These collapse to the same stem without mint stripping, but they
  are the same agent, not a cross-install collision.

## Conclusion

CONCLUSION [scope: archive EVENT_A2A rows in /tmp/tmpn67fr05u/taosmd-test]: mint-stamp stripping is NOT safe for membership.

The measured data shows multiple distinct installs sharing one mint-stripped
stem (`hermes` on `general`, `taosmd` on `build`).  Mint-stamp stripping would
merge those installs into one identity.  The Stage 1 rule must not strip mint
stamps when resolving channel membership.
