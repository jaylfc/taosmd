### Fixed

- Graph expansion seed selection now ranks entities by relevance before capping (`max_seeds`, default 10). Relevance is the highest score from any result mentioning the entity (using `score` then `source_score`, ignoring non-numeric values). Tie-breaks: mention count across all results (more first), then encounter order (earlier first). When no results carry valid scores, the original encounter order is preserved exactly.
