### Fixed

- The `taosmd serve` startup banner now prints an `A2A registry auth mode:` line that
  reads `OFF`, `ENFORCE`, or `WARN (verify-and-warn)` from the same config the
  enforcement path branches on, so the banner no longer claims `ENFORCE` when no
  registry_url is configured and no longer advertises `no auth` on the `where` line
  once enforcement is active.
