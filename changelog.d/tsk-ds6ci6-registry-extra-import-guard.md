### Fixed

- When `registry_url` is configured but the optional `pyjwt` and `cryptography`
  packages are missing, the server now fails loudly at startup with an actionable
  error naming `taosmd[registry]`, instead of starting silently and degrading on
  the first auth-gated request.
- The deploy scripts (`scripts/setup.sh` and `scripts/install-server.sh`) now
  install `taosmd[registry]` so fleet service installs include the crypto extra
  required for registry auth.
