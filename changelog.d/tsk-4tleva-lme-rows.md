Fix review BLOCKers:
- Move NON_ANSWER_PHRASES above is_non_answer function to fix NameError
- Remove 4 JSON test artifacts (longmemeval_*.json) from benchmarks/results/
- Add direct unit tests for read_save_context with dict inputs as required by revision requirement (a)
- Fix duplicate assertion in test_is_non_answer
- Add newline at end of tests/test_longmemeval_rows.py
