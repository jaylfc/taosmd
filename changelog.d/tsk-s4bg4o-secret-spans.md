### Added

- `find_secret_spans(text)` returns disjoint, sorted spans over the original text, and `redact_secrets` is rebuilt on it so replacements are spliced from original offsets and a later pattern can never match inside an earlier `[REDACTED:...]` placeholder.
