### Fixed

- `llm_complete` now accepts a `thinking_fallback: bool = True` parameter to control whether to fall back to thinking/reasoning text when the primary response is empty. When set to `False`, only returns the response/content text (empty string if empty). This fixes issues where reasoning text was incorrectly stored as insights in `reflect.py` and where ranking logic was misinterpreting reasoning text in `llm_rerank.py`.

- Updated `reflect.py` and `llm_rerank.py` to use `thinking_fallback=False` to ensure they only process the actual response content.

- Added tests `test_ollama_thinking_fallback_disabled` and `test_openai_reasoning_fallback_disabled` to verify the new behavior.
