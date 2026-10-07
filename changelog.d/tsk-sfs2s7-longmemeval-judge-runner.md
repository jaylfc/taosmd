### Fixed

- `benchmarks/longmemeval_runner.py` initialises `answer` to `""` on every path in
  `run_benchmark` so the substring mode (`use_llm=False`) no longer raises
  `UnboundLocalError` when the answer is referenced.
- `benchmarks/longmemeval_runner.py` adds `TAOSMD_LME_GEN_TEMP` parsing via
  `_parse_gen_temp`, which falls back to `0.0` (with a stderr warning) for empty,
  non-numeric, negative, `nan` and `inf` values; the parsed value is plumbed
  through `_gen_options(temperature=GEN_TEMP, ...)` into the generation payload
  while the judge payload keeps its hard-coded `"temperature": 0`.
- `benchmarks/longmemeval_runner.py` adds `TAOSMD_LME_NO_INLINE_JUDGE` parsing
  (wrapped in `try/except` so a bad env var cannot crash import); when set to `1`
  the runner skips inline LLM scoring, records `answer` and `gold_answer` per
  question, omits `correct`/`accuracy` from the result metrics and the `Overall`
  summary line, and returns `None` instead of a float.
- `benchmarks/longmemeval_runner.py` adds `gen_temp` and `inline_judge` to the
  result JSON metadata, and persists `question_id`, `answer` and `gold_answer`
  in each per-question result row.

### Added

- `tests/test_longmemeval_runner_new_features.py` exercises the shipped
  `run_benchmark` (no reimplementation) and covers: substring-mode completion,
  NO_INLINE_JUDGE metric shape and return value, GEN_TEMP edge-case parsing,
  byte-identical default generation and judge payloads, and answer+gold
  persistence.
