### Fixed
- Fixed `UnboundLocalError` in `_format_hit` when `timestamp` was referenced before definition
- Defensive coercion: `as_of` now uses `try/except (TypeError, ValueError)` with `logger.warning` instead of unguarded `float()`, so ISO-8601 timestamps from caller metadata degrade to `0.0` rather than raising
- `review_by` comparison gated on `isinstance(review_by, str)` to prevent `TypeError` when non-string metadata arrives through `ingest_batch`