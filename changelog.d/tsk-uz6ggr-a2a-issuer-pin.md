### Fixed

- A2A auth tests now pin the registry issuer in their verifier construction,
  exercising the production `expected_iss=registry_auth.REGISTRY_ISS` check
  instead of passing `expected_iss=None`. Wrong-issuer tokens are rejected with
  401 on all four inbox-family endpoints, and a production-path test confirms
  the pin is enforced when the verifier is built by `_make_handler` rather than
  injected.

### Docs

- `GET /a2a/inbox` parameter list in `taosmd/http_server.py` and
  `taosmd/docs/a2a-comms.md` now includes `exclude_acked_by`, with a description
  of what it does (omits messages already acked by the named principal).
