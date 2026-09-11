### Fixed
- `_qs_param_names` in `taosmd/http_server.py` now URL-decodes parameter names
  before comparing against the allowlist, matching the behaviour of
  `urllib.parse.parse_qs`. Percent-encoded legitimate parameter names (e.g.
  `%73ince` for `since`) are now correctly accepted.