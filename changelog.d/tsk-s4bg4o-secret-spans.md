### Added

- `find_secret_spans(text)` returns disjoint, sorted spans over the original text, and `redact_secrets` is rebuilt on it so replacements are spliced from original offsets and a later pattern can never match inside an earlier `[REDACTED:...]` placeholder.
- Overlapping secret spans now merge into a single span covering the union of their ranges (the kind and replacement of the span that opened it are kept, and all contributing kinds are reported in `found`).
