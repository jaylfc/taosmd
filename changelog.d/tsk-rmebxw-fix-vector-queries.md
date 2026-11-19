### Fixed

- Fixed bug where "what is X?" questions never reached vector memory in fast and minimal strategies
- Renamed all 'qmd' source names to 'vector' in SEARCH_STRATEGIES (intent_classifier.py)
- Added 'qmd' as an alias to 'vector' in retrieve() for backward compatibility
- Modified fast and minimal strategies to always query vector source when present, even if it's the tertiary source
- Added tests in tests/test_fast_strategy_sources.py to verify vector is always queried
