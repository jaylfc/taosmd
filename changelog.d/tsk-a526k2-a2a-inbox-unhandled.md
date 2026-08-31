### Fixed

- `a2a_inbox` now propagates `acked_by` on envelopes when present, and accepts `exclude_acked_by` to omit acknowledged messages before the limit budget is applied.
- A2A inbox auth tests now pin the registry issuer (`expected_iss=REGISTRY_ISS`) instead of leaving it unbound, and reject tokens carrying a different issuer. This covers the `/a2a/inbox`, `/a2a/inbox/advance`, `/a2a/ack`, and `/a2a/inbox/unhandled` auth paths.
- `GET /a2a/inbox` parameter docs in `taosmd/docs/a2a-comms.md` and the `taosmd/http_server.py` module docstring now list `exclude_acked_by` and state that it omits messages already acknowledged by the named principal.
