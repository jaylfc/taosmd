# Changelog Fragment Convention

Each file in `changelog.d/` describes one change for the next release.

Trailer lines (each alone on its own line, canonical form):

```
Migration: one-way          # after it runs, the previous release cannot use the data
Migration: reversible       # schema or data change the previous release tolerates
Manual-Step: <one sentence> # operator must do this by hand before/after deploy
```

A fragment with no trailer declares no migration and no manual step. A line is a
trailer attempt only when, after normalisation and markup stripping, it starts
with `migration`/`migrations` or `manual[\W_]*steps?` and a colon-like char; any
non-canonical trailer attempt is refused by `scripts/release_meta.py`.
