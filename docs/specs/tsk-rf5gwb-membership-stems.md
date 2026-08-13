# tsk-rf5gwb: Membership identity-stem measurement

Scope: channel membership principals from archive EVENT_A2A rows.

## Why this exists

Stage 2's mismatch gate and the membership reconciliation migration both key on
normalised identity. The `from`-field measurement (last 400 bus messages) is
scoped to `from` only. Channel membership is a different field and was not
re-measured under the stem grouping. Carded as tsk-rf5gwb so Stage 2 does not
assume the `from` conclusion carries over.

## Method

1. Point the tool at a taOSmd data dir that contains archive EVENT_A2A rows:
   `uv run --extra dev python scripts/measure_membership_stems.py --data-dir /path/to/data`
   The tool reads `archive/` and `archive-index.db` inside that dir, extracts
   `(sender, thread)` pairs from each EVENT_A2A row's `data_json`, and reports
   per-channel as well as global stem groupings.
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

Run the command above twice on the same data dir and the table is identical.
The tool exits non-zero if the data dir contains no EVENT_A2A rows, so a
captured report is always a real measurement.

| Question | Answer |
|---|---|
| Total distinct principals | 34 |
| Channels | 17 |
| Stems with >1 spelling, no mint stripping | 4 |
| Stems with >1 spelling, with mint stripping | 4 |
| Canonical entries with bare or @-form twin | 2 |
| Distinct principals collapsing to one stem (no mint) | 0 |

### Per-channel stems carrying more than one spelling

Four channels carry more than one `hermes` spelling; `build` carries all three:

```
build: principals=3, multi_no_mint=1, multi_with_mint=1, canonical_twins=1, collapse_no_mint=0
hermes: principals=2, multi_no_mint=1, multi_with_mint=1, canonical_twins=1, collapse_no_mint=0
```

The `hermes` stem group across the fleet:

```
hermes: ['hermes', 'hermes-20260608-153000', 'hermes-20260727-001415']
```

### Canonical twin check

Two canonical (mint-stamped) membership entries have a bare or `@`-form twin.
Both belong to the `hermes` install family:

```
hermes-20260608-153000 -> ['hermes']
hermes-20260727-001415 -> ['hermes']
```

Mint-stamp stripping would unify these three distinct install spellings into a
single `hermes` stem, merging two installs into one identity. That is unsafe
for membership.

### Collapse check (distinct principals to one stem)

Zero distinct principals collapse to one stem without mint stripping. The slug
match without mint stripping is safe.

## Conclusion

Mint-stamp stripping is NOT safe for membership under the measured scope.
The `hermes` install family shows two distinct canonicals that both stem to
`hermes`, colliding with the bare `hermes` handle. Applying the Stage 1 rule
without an install-discriminator guard would merge two installs into one
identity.

The `from`-field conclusion does NOT carry over. Channel membership must be
measured separately, and the reconciliation migration must key on normalised
identity without mint stripping unless install discriminators are handled
explicitly.
