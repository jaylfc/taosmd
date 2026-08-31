### Fixed

- `taosmd collections revoke` now exits non-zero and reports to stderr when no
  grant matched, instead of returning 0 with identical output to a successful
  revocation. The service layer already carried the `revoked` signal; the CLI
  caller was discarding it, leaving the false-negative defect from tsk-siwsp7
  alive on the command surface. Added a CLI test that distinguishes the two
  cases.
