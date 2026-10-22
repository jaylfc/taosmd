### Fixed
- `taosmd hooks parser`: a transcript line whose JSON holds a lone UTF-16
  surrogate escape (a `\ud800`-style escape in user `content`, a `text` block,
  a `tool_result` `content`, an assistant `text` block, or a `uuid`) no longer
  makes `parse_entry` raise `UnicodeEncodeError` from `_cap_field`'s UTF-8
  encode. Such values are now passed through `_sanitize_text` (an
  `encode("utf-8", "replace").decode("utf-8")` round-trip) before field capping
  and before `base_id` is built from `session_id_from`/`uuid`, so a single bad
  line no longer stalls capture for the rest of the session.
