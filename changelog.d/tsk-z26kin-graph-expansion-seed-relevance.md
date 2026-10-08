### Fixed
- Seed entities in graph expansion are now ranked by relevance (highest score of any result mentioning the entity, with mention-count and encounter-order tie-breaks) instead of taking the first ten in encounter order. Added a `max_seeds` keyword to `expand_from_results()` so callers can control the cap.
