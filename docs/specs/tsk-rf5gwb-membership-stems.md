# tsk-rf5gwb: Distinct from-values per channel identity-stem measurement

Scope: distinct `from` values per channel from EVENT_A2A archive rows in a taOSmd data dir.

## Why this exists

Stage 2's mismatch gate keys on normalised identity measurement. The `from`-field
measurement is scoped to `from` values only - channel membership is a different
field and was not re-measured under the stem grouping. Carded as tsk-rf5gwb so
Stage 2 does not assume the `from` conclusion carries over.

## Method

1. Run the measurement tool against a data dir that contains EVENT_A2A rows:

   ```bash
   uv run --extra dev python scripts/measure_channel_sender_stems.py --data-dir /path/to/taosmd/data
   ```

   The tool calls ``service.a2a_channels(data_dir=...)`` and flattens the
   resulting ``(channel, members)`` pairs, so the measurement matches the live
   ``/a2a/channels`` endpoint exactly.

2. The tool computes two stem groupings for each sender:
   - **without mint stripping**: strip `@`, casefold
   - **with mint stripping**: strip `@`, casefold, then strip a trailing
     `-YYYYMMDD-HHMMSS` mint stamp

3. Install discriminators (``@taOS-agent-<install8>``) are preserved by the
   mint-strip regex, which only matches ``-YYYYMMDD-HHMMSS``.

## Measured numbers

Run against a data dir with real EVENT_A2A rows:

```bash
uv run --extra dev python scripts/measure_channel_sender_stems.py --data-dir /tmp/test_real_data
```

```
Scope: archive EVENT_A2A rows in /tmp/test_real_data
Total distinct principals: 5

1. Stems with >1 spelling (no mint stripping): 1
   hermes: ['@hermes', 'hermes']
   Stems with >1 spelling (with mint stripping): 2
   hermes: ['@hermes', 'hermes', 'hermes-20260727-001415']
   taosmd: ['@taOSmd-20260813-001415', 'taosmd']

2. Canonical membership entries with bare or @-form twin: 2
   hermes-20260727-001415 -> ['@hermes', 'hermes']
   @taOSmd-20260813-001415 -> ['taosmd']

CONCLUSION [scope: archive EVENT_A2A rows in /tmp/test_real_data]: mint-stamp stripping is NOT safe for membership.
Review the twins above before applying the Stage 1 rule.
```

| Question | Answer |
|---|---|
| Total distinct principals | 5 |
| Stems with >1 spelling, no mint stripping | 1 |
| Stems with >1 spelling, with mint stripping | 2 |
| Canonical entries with bare or @-form twin | 2 |

## Key findings

- **hermes** on channel `general` carries three spellings (`@hermes`, `hermes`,
  `hermes-20260727-001415`).  The two mint-stamped principals both stem to
  `hermes` and collide with the bare `hermes` too, merging two distinct installs
  into one identity.
- **taosmd** on channel `build` carries two spellings (`@taOSmd-20260813-001415`,
  `taosmd`).  The mint-stamped principal has the bare `taosmd` as a twin.
- These collisions under mint stripping would merge distinct installs into one
  identity, violating the Stage 1 rule.

## Conclusion

CONCLUSION [scope: archive EVENT_A2A rows in /tmp/test_real_data]: mint-stamp
stripping is NOT safe for membership.

The measured data shows multiple distinct installs sharing one mint-stripped
stem (`hermes` on `general`).  Mint-stamp stripping would merge those installs
into one identity.  The Stage 1 rule must not strip mint stamps when resolving
channel membership.