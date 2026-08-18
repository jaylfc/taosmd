### Fixed

- `uninstall_systemd()` in `taosmd/service_install.py` now returns non-zero when
  `daemon-reload` fails or the unit file cannot be removed, instead of always
  returning 0. Added unit tests for both failure paths and the missing-unit
  no-op path.
