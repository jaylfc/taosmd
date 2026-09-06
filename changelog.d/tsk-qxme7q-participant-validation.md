### Fixed

- Validated each element of `participants` in `a2a_create_thread` before it reaches the database, rejecting `None`, non-string types, and empty strings with `ValueError` (mapped to HTTP 400) instead of leaking internal schema names via HTTP 500. Duplicate participants are now de-duplicated so that each principal receives exactly one `membership_created` archive event and a single, unchanged `created_at` in the store.
