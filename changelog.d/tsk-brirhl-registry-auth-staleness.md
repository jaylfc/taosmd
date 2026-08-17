### Added

- `RegistryVerifier` (and `verifier_from_url`) now accepts a configurable
  `staleness_bound` (seconds); the revocation cache fails closed once it exceeds
  the bound on a refresh failure, instead of tolerating stale data indefinitely.
  The bound is configurable via the `TAOSMD_REGISTRY_STALENESS_BOUND` env var or
  the `registry_staleness_bound` config key, defaulting to `6 * refresh_interval`.

### Fixed

- A2A bus auth now distinguishes missing-credentials from presented-credential
  failures: a missing Bearer token is still warn-and-accept in verify-and-warn
  mode (the default), but any token that is presented and fails verification
  (bad signature, issuer mismatch, revoked id, sub != from) returns a hard 403
  regardless of the `a2a_auth_enforce` flag. Grant failures -- a valid token
  whose `sub` matches `from` but holds no active grant -- remain warn-and-accept
  in warn mode and 403 in enforce mode, preserving the documented migration
  contract.
