### Fixed

- A contradiction found while extracting facts no longer soft-hides unrelated memories. Resolving one used to hide every vector row in the whole store, for every agent, whose text contained the old value ("The office is cold" then "The office is nice" hid every saved turn containing "cold"). Resolving a contradiction now only invalidates the old triple in the knowledge graph. Facts from the regex extractor no longer auto-resolve by default: the conflict is queued for `taosmd review` and both triples stay as they were. `process_conversation_turn(auto_resolve_contradictions=...)` now defaults to `None` (LLM facts resolve as before, regex facts are queued); pass `True` or `False` to force either.
- `taosmd supersede --match` now needs `--agent`, only matches that agent's rows, refuses a needle under 12 characters or more than 20 matches unless `--force`, and `VectorMemory.supersede_matching` returns the list of hidden ids instead of a count.

### Added

- Every supersede is written to a new `supersede_audit` table in the vector database. `taosmd supersede --list-hidden [--agent A] [--json]` lists hidden rows with the reason (`unknown (before audit log)` for older ones), and `taosmd supersede --restore ID [ID ...]` and `--restore-all --since DATE [--agent A]` bring rows back. Nothing is deleted.
