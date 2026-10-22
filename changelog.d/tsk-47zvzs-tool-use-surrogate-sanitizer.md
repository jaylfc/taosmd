### Fixed
- `taosmd hooks parser`: a transcript line whose JSON holds a lone UTF-16
  surrogate escape in a `tool_use` `name` or `id` field no longer makes
  `parse_entry` raise `UnicodeEncodeError`. Both fields are now passed through
  `_sanitize_text` before being stored in the item metadata.