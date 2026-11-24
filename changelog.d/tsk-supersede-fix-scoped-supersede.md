### Fixed

- A contradiction found while extracting facts no longer soft-hides unrelated memories. Resolving one used to hide every vector row in the whole store, for every agent, whose text contained the old value ("The office is cold" then "The office is nice" hid every saved turn containing "cold"). Resolving a contradiction now only invalidates the old triple in the knowledge graph. Facts from the regex extractor no longer auto-resolve by default: the conflict is queued for `taosmd review` and both triples stay as they were. `process_conversation_turn(auto_resolve_contradictions=...)` now defaults to `None` (LLM facts resolve as before, regex facts are queued); pass `True` to resolve every fact in the graph, or `False` to keep both facts as before.
- `taosmd supersede --match` now needs `--agent`, only matches that agent's rows, and refuses a needle under 12 characters or more than 20 matches unless `--force`.
- Breaking for in-process callers of `VectorMemory.supersede_matching`: it now returns a list of hidden ids instead of a count (`n > 0` raises TypeError and `n == 0` is silently False, so use `len(n)`), and a call without an agent scope raises `SupersedeRefused` instead of hiding rows store-wide. `api.supersede_vectors` raises the same way without `agent` but still returns an int.

### Added

- Every supersede is written to a new `supersede_audit` table in the vector database. `taosmd supersede --list-hidden [--agent A] [--json]` lists hidden rows with the reason (`unknown (before audit log)` for older ones), and `taosmd supersede --restore ID [ID ...]` and `--restore-all --since DATE [--agent A]` bring rows back. Nothing is deleted.
