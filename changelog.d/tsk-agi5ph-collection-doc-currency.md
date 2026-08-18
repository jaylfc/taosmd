### Fixed
- `_format_hit` now defensively coerces `as_of` to float (ISO-8601 string timestamps degrade to `0.0` with `logger.warning`) and guards the `review_by` comparison on `isinstance(review_by, str)`, so malformed caller metadata from `ingest_batch` degrades gracefully instead of raising `ValueError`/`TypeError` on the search path. Non-string `review_by` values also log a warning (symmetric with the `as_of` guard) instead of silently yielding `is_past_review=False`.

### Added
- Markdown front matter (`doc_id`, `version` as int, `review_by` as validated ISO date) is now parsed at collection index time via `_parse_front_matter` and stored on chunk metadata as `indexed_at` plus the doc-currency keys. Search hits gain `is_current`, `as_of` (always float), `is_past_review`, and (on superseded rows) `superseded_by`. Front-matter parsing respects both `.md` and `.markdown` extensions and ignores document keys from thematic-break documents via a closing-delimiter line budget.
