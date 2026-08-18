### Fixed
- Corrected "Used by" list for `longmemeval_s_full.json` to name all 9 gate-verified loaders, replacing catch-all `longmemeval_*` phrasing
- Added `longmemeval_oracle.json` section with verified byte size (15388478), sha256 (`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`), and 500 questions; relationship to `longmemeval_s_full.json` stated as unverified here since `longmemeval_s_full.json` does not exist on this machine
- Added `longmemeval_s_cleaned.json` section with `NOT YET PINNED` status: byte size, question count, and sha256 explicitly removed; relationship to `longmemeval_s_full.json` stated as not available (no verified copy located)
- Restored headline provenance sentence: "It is the source of the published 97.0% Recall@5 headline (see `benchmarks/REPRODUCE-longmemeval.md`)."
