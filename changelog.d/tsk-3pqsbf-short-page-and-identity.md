### Fixed

- **Short page paging**: Fixed `_handle_a2a_messages` in `taosmd/http_server.py` to track previous `len(rows)` and break only when the feed is truly exhausted (no new rows returned from a larger scan), instead of breaking on `len(rows) < scan` which could terminate early when `a2a_feed` drops superseded/admin_action/deleted-channel rows. Keeps `scan_cap = limit_i * 5` and the doubling behavior.

- **Identity resolution caching**: Fixed `_get_authenticated_agent_id` in `taosmd/http_server.py` to cache the verified caller result per request instance, preventing repeated identity resolution on every row through `_filter_readable` -> `_can_read_channel` -> `_acl_allows`.
