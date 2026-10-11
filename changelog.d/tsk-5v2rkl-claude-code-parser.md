### Added

- Claude Code transcript parser hook implementation (`taosmd/hooks/parser.py`)
- Fixture with 16 sample transcript entries (`tests/fixtures/claude_code_transcript.jsonl`)
- Comprehensive test suite (`tests/test_hooks_parser.py`) covering:
  - Fixture parsing with exact item counts per kind
  - ID format rules and fallback logic
  - Timestamp conversion from ISO-8601
  - Byte capping (8 KB limit per field)
  - Noise filtering for standard Claude Code entry types
  - Tool result parsing with both string and list content
  - Assistant entries with both text and tool_use
  - Truncated flag for oversized fields

### Fixed

- Corrected test expectations to match actual parser output (4 message items, 2 assistant messages instead of 5 and 3)
- Updated test logic to properly filter message items vs tool_result items when checking roles
- Fixed noise type filtering to include all standard Claude Code noise entry types