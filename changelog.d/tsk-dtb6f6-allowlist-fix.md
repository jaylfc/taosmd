### Fixed

Release metadata generator now uses an allowlist for trailer parsing: only exactly canonical
trailer forms (`Migration: one-way`, `Migration: reversible`, `Manual-Step: <text>`) are accepted.
Any other line that is a trailer attempt (after normalisation and markup stripping) but not
in canonical form refuses with exit 2. Fence detection now allows up to 3 spaces
indent and any info string for both ``` and ~~~ fences. Modified fragments are now scanned
along with added ones (--diff-filter=AM).
