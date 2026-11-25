### Fixed
- Librarian task gates (crystallise, reflect, fact_extraction, query_expansion, verification, effective_fanout) now resolve the agent registry from the passed `data_dir` instead of only the canonical default, so served non-default installs (where `taosmd serve` does not export `TAOSMD_DATA_DIR`) no longer skip crystallise for agents registered in the served data dir.
