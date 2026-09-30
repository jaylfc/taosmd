### Fixed

- `POST /a2a/import` now binds each envelope's `from` to the verified registry token `sub` instead of comparing the token to itself. A token for agentA can no longer import messages attributed to agentB. Mixed-sender batches are rejected as a whole. The grant check mirrors `/a2a/send`: tokens without an active `a2a_send` grant are refused. Warn mode (missing token when `a2a_auth_enforce` is off) now matches `/a2a/send` and accepts with a warning instead of hard-401'ing.
