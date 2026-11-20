### Fixed
- Hit confidence now correctly reflects the source similarity/score across all retrieval paths (fast primary, fast always-vector, minimal fallback, thorough). The `_rrf_merge` and `_deduplicate` steps no longer overwrite `source_score`.
- Adapters for KG, catalog, archive, and crystals now set `source_score` to `None` (instead of `1.0`) when the source row has no score field.
- `_format_hit` emits `confidence: null` for hits with `source_score: None`, so agents treat unknown confidence as unknown rather than certain.
- Consumers of `source_score`/`confidence` updated to handle `None`: `_attach_and_gate_claims` (api.py:479), `_score_of` (graph_expansion.py:67-79), `apply_verification_verdicts` (retrieval.py:430).
