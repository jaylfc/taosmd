### Fixed

- LFS pointer detection: treat 0-byte files and BOM-prefixed pointers as pointers; reject pointers at all sibling sites (auto_setup, cross_encoder, recipes)
- qrels k validation rejects bool (True/False) in addition to non-int and non-positive
- Restore reextract scaffold changelog fragment to ### Added (was incorrectly ### Fixed)
- Restore add_claim_ex docstring sentence about busy_timeout
- Fix test_store.py docstring: 'persistent counter lives here' -> 'by_extractor tally in rate()'
- Claims by_extractor: empty source_extractor counted as 'unknown'; CLI claims status shows by_extractor suffix
- Memory extractor: pin test that _llm_fell_back ContextVar is reset before _extract_with_llm_status runs
- llm_complete: temperature, max_tokens, timeout passed through to both Ollama and OpenAI dialects; crystallize fallback on router non-200
