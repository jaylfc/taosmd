### Fixed
- `DELETE /collections/{id}/grants/{agent}` now percent-decodes `{agent}` in the
  handler before handing it to the store. The segment is caller-supplied, so an
  agent id carrying a space, `+`, `/`, or a non-ASCII character was revoked under
  its encoded spelling while `grant` stored the decoded one: the DELETE matched
  no row and still returned `200` with the grant still live. Decoding at the
  HTTP boundary (the same place `_handle_a2a_alarms_clear` already decodes its
  `{key}`) makes revoke and grant agree on the spelling.
