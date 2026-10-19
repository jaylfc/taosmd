### Added

- LongMemEval row-building functions: `NON_ANSWER_PHRASES`, `is_non_answer()`, `build_row()`, and `read_save_context()`.
- Environment variable parsing for `TAOSMD_LME_SAVE_CONTEXT` with proper validation.
- Module-level non-answer phrase filtering.
- Support for `predicted` and `answer` fields in benchmark results.
- Context truncation based on `CONTEXT_CHARS` when `save_context=True`.
