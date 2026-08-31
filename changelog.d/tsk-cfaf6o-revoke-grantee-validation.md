### Fixed

- `revoke` now validates its grantee and raises `ValueError` on `''`, whitespace-only, `None`, and non-string ids, matching the existing `grant` behaviour. The HTTP surface returns 400 for these cases.
