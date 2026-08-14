# tsk-rf5gwb: Membership identity-stem measurement

Status: CLOSED
Scope: channel membership principals from bus-spool.jsonl (499 sender/channel pairs, 753 raw lines)

## Why this exists

Stage 2's mismatch gate and the membership reconciliation migration both key on
normalised identity. The `from`-field measurement (last 400 bus messages) is
scoped to `from` only. Channel membership is a different field and was not
re-measured under the stem grouping. Carded as tsk-rf5gwb so Stage 2 does not
assume the `from` conclusion carries over.

## Method

1. Extract every distinct `(sender, channel)` pair from the bus-spool.
   Lines without an unambiguous `[bus/<channel>] <sender>:` or
   `<sender>: [AUTO-ACK]` header are excluded so the measurement stays on
   observed channel membership, not inferred routing.
2. Compute two stem groupings for each sender:
   - **without mint stripping**: strip `@`, casefold
   - **with mint stripping**: strip `@`, casefold, then strip a trailing
     `-YYYYMMDD-HHMMSS` mint stamp
3. Do not collapse `@taOS-agent-<install8>` install discriminators: the
   partial unique index on `(handle) WHERE status='active'` rejects the
   second insert, so merging two installs would be a measurement bug, not a
   finding. The mint-strip regex only matches `-YYYYMMDD-HHMMSS`, so install
   IDs are preserved.

## Measured numbers

| Question | Answer |
|---|---|
| Total distinct principals | 14 |
| Stems with >1 spelling, no mint stripping | 1 |
| Stems with >1 spelling, with mint stripping | 1 |
| Canonical entries with bare or @-form twin | 0 |
| Distinct principals collapsing to one stem (no mint) | 1 |

### Stems carrying more than one spelling

Both groupings see the same single multi-spelling stem:

```
taosmd-dev: ['@taOSmd-dev', 'taosmd-dev']
```

No other stem carries more than one spelling.

### Canonical twin check

Zero canonical (mint-stamped) membership entries have a bare or `@`-form twin.

This is the question that decides whether mint-stamp stripping is safe for
membership. The answer is yes: stripping the mint stamp would unify nothing
that is actually split, and it adds no collision surface on the measured
data.

### Collapse check (distinct principals to one stem)

One pair of distinct raw senders collapses to the same stem without mint
stripping:

```
taosmd-dev: ['@taOSmd-dev', 'taosmd-dev']
```

These are two spellings of the same agent. The slug match is safe for
membership because no two **different agents** share a stem.

## Conclusion

Mint-stamp stripping is safe for membership under the measured scope.
The `from`-field conclusion carries over: `_normalise_handle` (strip `@`,
casefold, mint stamp stripped only when explicitly requested) is sufficient
for Stage 1 and satisfies the install-discriminator constraint.

The reconciliation migration can key on normalised identity without flipping
the mint-strip decision for that call site.
