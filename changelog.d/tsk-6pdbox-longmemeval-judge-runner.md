### Added

- LongMemEval runner supports `TAOSMD_LME_GEN_TEMP` for per-run generation temperature with fallback and warning on bad values.
- LongMemEval runner supports `TAOSMD_LME_NO_INLINE_JUDGE=1` to suppress inline LLM scoring, emitting only `n` in metrics.
- `all_results` rows now carry `question_id`, `answer`, and `gold_answer` alongside `question_type`.
