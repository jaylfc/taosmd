### Fixed

`process_conversation_turn` now calls the public `extract_facts_with_llm` seam again (looked up as a module global at call time) instead of the private `_extract_with_llm_status`. Code that patches `taosmd.memory_extractor.extract_facts_with_llm` is intercepted again, so the per-call fallback flag travels through a module-level `contextvars.ContextVar('_llm_fell_back', default=None)` instead of being lost. A patched seam that never sets the var is treated as not fell back, so the existing `elif facts` / `else` branches decide.
