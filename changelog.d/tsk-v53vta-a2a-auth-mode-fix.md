### Fixed

- Revised A2A auth mode banner to correctly reflect three states: OFF (no registry_url),
  ENFORCE (registry_url + enforce on), and WARN (verify-and-warn). The banner no longer
  claims "ENFORCE" when no registry_url is configured, and the `where` line no longer
  asserts "no auth" when enforcement is active.