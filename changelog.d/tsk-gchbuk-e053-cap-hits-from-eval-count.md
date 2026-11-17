### Fixed

- `benchmarks/e053_npu_speed.py`: the E-053 scorer now counts a cap hit from a
  loop row's `eval_count` (the server's generated token count, already stored
  per row) against the 512-token cap, instead of from
  `len(output_text.split())` whitespace words. 512 model tokens of English is
  roughly 350 to 400 words, so a capped output almost never reached 512 words
  and the kill clause (3) passed falsely. A valid loop row with `eval_count`
  missing is now counted in a new `cap_unknown` field, and the cap_hits kill
  check status is `MISSING` when `cap_unknown > 0` rather than silently `PASS`.
  The unused `eval_count_mismatch` field and `_tokenize_whitespace` helper were
  removed.
