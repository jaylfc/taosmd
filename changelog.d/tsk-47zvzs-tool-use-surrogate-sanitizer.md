### Fixed
- `taosmd hooks parser`: a lone UTF-16 surrogate escape in a `tool_use` `name` or `id` field is now passed through `_sanitize_text` before being stored as `tool_name` / `tool_id` in the item metadata, so the metadata always encodes to UTF-8.
