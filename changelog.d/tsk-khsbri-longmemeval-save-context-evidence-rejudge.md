### Added

- LongMemEval save-context + evidence judge + rejudge functionality (tsk-khsbri):
  - Factor per-question row into `build_row()` function that always stores 'predicted' and 'answer' keys
  - Save 'context' as `full_context[:CONTEXT_CHARS]` when `TAOSMD_LME_SAVE_CONTEXT=1`
  - Update `score_answer_llm()` to accept context parameter and handle `TAOSMD_JUDGE_MODE` environment variable
  - Move non-answer tuple to module-level `NON_ANSWER_PHRASES` constant
  - Update rejudge script to skip judge calls for missing/empty 'predicted' or 'answer' fields
  - Count unusable rows and exit with code 3 if all rows are unusable

This implements the requirements from PR #550 to support saving context per question and evidence mode judging with proper validation.
