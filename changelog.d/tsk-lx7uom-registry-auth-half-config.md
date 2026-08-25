### Fixed

- When `registry_url` is configured but `registry_token` is not, `taosmd serve`
  now emits a clear startup diagnostic naming both keys, and rejected
  `/a2a/send` requests return an error that explicitly mentions the missing
  `registry_token` instead of a bare revocation-feed failure. `taosmd config
  show` also reports both keys and warns on the half-configuration.
