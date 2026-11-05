### Added

- Loud extraction-method counters (issue #653 item 4). A permanent, silent
  degrade from LLM fact extraction to the regex path was invisible: the field
  report saw 10,063/10,063 claims from regex with nothing surfacing it.
  `taosmd.memory_extractor.extraction_stats()` now counts `llm`,
  `regex_fallback` (LLM error / empty result inside
  `extract_facts_with_llm`), and `regex_direct` (no LLM configured or
  `use_llm=False`) per call, with no double-counting when both paths come up
  empty. `ClaimStore.rate()` gains a `by_extractor` breakdown and
  `taosmd claims status` prints it. After `_FALLBACK_ALERT_AFTER` (10)
  consecutive regex fallbacks the extractor logs one WARNING containing
  "consecutive regex fallback"; any LLM success resets the streak so the
  alert re-arms.
