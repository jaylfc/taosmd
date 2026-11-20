# Changelog Fragment Convention

Each file in `changelog.d/` describes one change for the next release.

Trailer lines (each alone on its own line, anywhere in the file):

```
Migration: one-way          # after it runs, the previous release cannot use the data
Migration: reversible       # schema or data change the previous release tolerates
Manual-Step: <one sentence> # operator must do this by hand before/after deploy
```

A fragment with no trailer declares no migration and no manual step.
