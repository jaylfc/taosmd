### Fixed

Release metadata generator now uses an allowlist for trailer parsing: only exactly canonical
`Migration: one-way`, `Migration: reversible`, and `Manual-Step: <text>` forms are accepted.
Any other line mentioning migration or manual steps (after NFKC normalization and Unicode
format character removal) refuses with exit 2. Fence detection now allows up to 3 spaces
indent and any info string for both ``` and ~~~ fences. Modified fragments are now scanned
along with added ones (--diff-filter=AM).
