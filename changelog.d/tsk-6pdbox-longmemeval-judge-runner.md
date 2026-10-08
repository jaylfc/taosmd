### Added

- LongMemEval runner supports `TAOSMD_LME_GEN_TEMP` for per-run generation temperature with fallback and warning on bad values (empty, non-numeric, negative, NaN, ±Inf all warn and fall back to 0; unset stays silent).
- LongMemEval runner supports `TAOSMD_LME_NO_INLINE_JUDGE=1` to suppress inline LLM scoring, emitting only `n` in metrics and returning `None`.
- `all_results` rows now carry `question_id`, `question_type`, `answer`, and `gold_answer`.
