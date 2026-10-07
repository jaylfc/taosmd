### Fixed
- `expand_from_results` now accepts a configurable `max_seeds` keyword argument
  (default 10) instead of the previous hardcoded slice of the first 10 entities,
  and ranks seed entities by relevance using the highest score of any input
  result whose text contains them, with ties broken by mention count then
  encounter order. When no input result carries a numeric score, the encounter
  order is preserved exactly.
