### Fixed
- `/a2a/mentions` now sends `reader` over the remote path and accepts it as a query parameter when auth is enabled, preventing silent identity substitution.
- Handles are normalised via the shared `_normalise_handle` helper so `@bob`, `bob`, and `BOB` resolve to the same identity.
- Cross-channel reply chains are excluded from the mentions feed; only replies within the same thread are followed.
- `limit` is applied once; `-1`, `0`, `nan`, `inf`, and absurd values are rejected with 400 instead of returning empty or 500.
- Mention regex no longer matches `@` inside email addresses or URLs.
- Dead code removed from `can_read`; quadratic reply-chain traversal replaced with an O(n) adjacency-list walk.
