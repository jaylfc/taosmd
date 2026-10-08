### Fixed
- `expand_from_results` now ranks seed entities by relevance (highest result
  score, then mention count, then encounter order) before applying the
  `max_seeds` cap, instead of taking the first N entities in encounter order.
  The hardcoded seed cap of 10 is replaced by a `max_seeds` parameter
  (default 10); `max_seeds <= 0` returns no seeds.
