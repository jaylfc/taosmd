### Fixed

- `tests/test_secret_spans._expected_found` now takes the text and reports every pattern with a non-empty match, instead of only the kind that opened each span. The old helper collected `span.kind` per span, so for any sample whose secrets overlap (a merged span) it under-reported: `redact_secrets` returns every contributing kind of a merged span (`['bearer_token', 'jwt']` for `Authorization: Bearer eyJ...`) while the helper expected only `['bearer_token']`. The call site in `test_redact_matches_spans` now passes `sample`.
