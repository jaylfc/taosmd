### Fixed

- `GET /a2a/*` endpoints now reject unknown query parameters that carry an empty value (e.g. `?since_id=`). Previously `parse_qs(keep_blank_values=False)` dropped blank parameters before the strict-params validator could see them, so a misspelt cursor like `since_id=` was silently ignored instead of returning 400.
