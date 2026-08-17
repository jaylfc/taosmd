### Added
- A `witness-token-gate` CI check (`scripts/check_witness_token.py`) that
  verifies any `# WITNESS: <test>::<token>` marker in a `taosmd/` source file
  resolves to a real `<token>` inside the named test file. A source comment that
  cites a test as the justification for a constant can no longer stay green after
  the witness it points at is deleted from that test, because the gate resolves the
  token, not just the path. Only the explicit `WITNESS:` marker is an assertion: a
  bare test-filename mention in ordinary prose is ignored.
