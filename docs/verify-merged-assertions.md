# Revert-and-rerun sweep: PR #212 and #213

## Environment verification
```
$ uv run python -c "import pytest, jwt, cryptography"
OK
```
Python 3.12.13, virtualenv created by uv sync.

## Experiment setup
Scratch worktree at `/tmp/scratch-verify` checked out at `origin/master` (4e792d1).
Each experiment modifies only the behaviour claimed by the PR, leaving imports intact.

---

## PR #212 — envelope refs+blocks validation block

**Behaviour reverted:** removed the envelope field validation block in `taosmd/http_server.py` `_handle_a2a_send` (refs list/max-8/dict-items/kind-enum, blocks list-of-dicts, 64KB cap, blocks-implies-non-empty-body invariant).

**Revert command:**
```python
# In taosmd/http_server.py _handle_a2a_send, replaced the validation block
# (lines 1508–1549) with a single body-empty check, leaving refs/blocks
# passthrough to service.a2a_send intact.
```

**Suite tail (validation block removed):**
```
FAILED tests/test_a2a.py::test_http_a2a_refs_not_list_returns_400
FAILED tests/test_a2a.py::test_http_a2a_refs_too_many_returns_400
FAILED tests/test_a2a.py::test_http_a2a_refs_item_not_dict_returns_400
FAILED tests/test_a2a.py::test_http_a2a_refs_bad_kind_returns_400
FAILED tests/test_a2a.py::test_http_a2a_blocks_not_list_returns_400
FAILED tests/test_a2a.py::test_http_a2a_blocks_item_not_dict_returns_400
FAILED tests/test_a2a.py::test_http_a2a_message_too_large_returns_400
FAILED tests/test_a2a.py::test_http_a2a_blocks_without_body_returns_400
8 failed, 27 passed in 17.15s
```

**Table — tests added by PR #212 (13 tests):**

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `test_a2a_send_with_refs_and_blocks_roundtrip` | DECORATIVE | Tests storage roundtrip, not validation. Stays green because valid inputs are still stored/returned. |
| 2 | `test_a2a_send_without_refs_blocks_omits_keys` | DECORATIVE | Tests storage roundtrip, not validation. Stays green. |
| 3 | `test_http_a2a_send_refs_blocks_roundtrip` | DECORATIVE | Tests HTTP storage roundtrip, not validation. Stays green. |
| 4 | `test_http_a2a_send_without_refs_blocks_omits_keys` | DECORATIVE | Tests HTTP storage roundtrip, not validation. Stays green. |
| 5 | `test_http_a2a_sse_with_refs_and_blocks` | DECORATIVE | Tests SSE storage roundtrip, not validation. Stays green. |
| 6 | `test_http_a2a_refs_not_list_returns_400` | REAL | Fails: returns 200 and stores invalid refs without the validation block. |
| 7 | `test_http_a2a_refs_too_many_returns_400` | REAL | Fails: returns 200 and stores >8 refs without the validation block. |
| 8 | `test_http_a2a_refs_item_not_dict_returns_400` | REAL | Fails: returns 200 and stores non-dict refs without the validation block. |
| 9 | `test_http_a2a_refs_bad_kind_returns_400` | REAL | Fails: returns 200 and stores invalid kind without the validation block. |
| 10 | `test_http_a2a_blocks_not_list_returns_400` | REAL | Fails: returns 200 and stores non-list blocks without the validation block. |
| 11 | `test_http_a2a_blocks_item_not_dict_returns_400` | REAL | Fails: returns 200 and stores non-dict blocks without the validation block. |
| 12 | `test_http_a2a_message_too_large_returns_400` | REAL | Fails: returns 200 and stores >64KB message without the validation block. |
| 13 | `test_http_a2a_blocks_without_body_returns_400` | REAL | Fails: returns 400 but error message does not mention `blocks` without the validation block (service layer raises ValueError for empty body, but the message is generic). |

**Decorative-test fix status:** The 5 roundtrip tests (rows 1–5) stay green under this revert because they exercise the storage path with valid inputs, not the validation path. Their assertions (`status == 200`, refs/blocks equality) are correct for roundtrip testing and cannot be strengthened to fail under a validation-only revert without changing their purpose. **No assertion fix applied.**

---

## PR #213 — /version endpoint + capabilities probe

Two independent reverts were applied separately.

### #213a — capability probe broken (`_resolves` always returns True)

**Behaviour reverted:** `taosmd/capabilities.py` `_resolves` changed to unconditionally return `True`, so capabilities are advertised even when their backing symbols are missing.

**Revert command:**
```python
# In taosmd/capabilities.py, replaced _resolves body with `return True`
```

**Suite tail (probe broken):**
```
FAILED tests/test_version_capabilities.py::test_capability_is_dropped_when_its_backing_symbol_is_missing
1 failed, 20 passed in 4.27s
```

**Table — tests added by PR #213 (21 tests):**

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `test_capabilities_are_contract_identifiers_with_version_suffix` | DECORATIVE | Tests capability name format. Passes with broken probe because all declared caps still have .vN suffixes. |
| 2 | `test_capabilities_include_collections_and_grants_on_this_build` | DECORATIVE | Tests specific caps present. Passes with broken probe because all caps are falsely advertised. |
| 3 | `test_capability_is_dropped_when_its_backing_symbol_is_missing` | REAL | Fails: collections.v1 remains advertised after its symbol is deleted. |
| 4 | `test_capability_declarations_do_not_diverge_from_the_http_surface` | DECORATIVE | Tests route markers exist in dispatcher source. Passes with broken probe because routes are unchanged. This is the test the card claimed "verified to FAIL by pointing a capability at a bogus route"; it does NOT fail when the probe is broken, confirming it tests route divergence, not probe resolution. |
| 5 | `test_every_declared_capability_probe_resolves_on_this_build` | DECORATIVE | Tests all declared caps resolve. Passes with broken probe because `_resolves` always returns True. |
| 6–21 | commit/build-info/health/version/token tests | DECORATIVE | Test build identity, /health contract, /version shape, token gating, etc. None depend on probe resolution; all pass with broken probe. |

**Decorative-test fix status:** The 20 passing tests either test unrelated behaviour (commit resolution, build-info caching, /health contract, data-plane gating) or test a different aspect of capabilities (format, route markers). Their assertions are strong for what they test and cannot be strengthened to fail under a probe-only revert without changing their purpose. **No assertion fixes applied.**

### #213b — /version route registration removed

**Behaviour reverted:** removed the `elif method == "GET" and path == "/version"` handler from `taosmd/http_server.py`.

**Revert command:**
```python
# In taosmd/http_server.py, removed the /version route handler block.
```

**Suite tail (/version route removed):**
```
FAILED tests/test_version_capabilities.py::test_version_endpoint_shape
FAILED tests/test_version_capabilities.py::test_version_matches_the_module_derivation
FAILED tests/test_version_capabilities.py::test_version_and_health_leak_nothing_sensitive
FAILED tests/test_version_capabilities.py::test_version_is_public_when_a_server_token_is_configured
4 failed, 17 passed in 4.27s
```

**Table — tests added by PR #213 (21 tests, route revert):**

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `test_capabilities_are_contract_identifiers_with_version_suffix` | DECORATIVE | Tests capability format. Passes without /version route. |
| 2 | `test_capabilities_include_collections_and_grants_on_this_build` | DECORATIVE | Tests specific caps present. Passes without /version route. |
| 3 | `test_capability_is_dropped_when_its_backing_symbol_is_missing` | DECORATIVE | Tests probe drops missing caps. Passes without /version route. |
| 4 | `test_capability_declarations_do_not_diverge_from_the_http_surface` | DECORATIVE | Tests route markers in dispatcher. Passes without /version route. |
| 5 | `test_every_declared_capability_probe_resolves_on_this_build` | DECORATIVE | Tests all probes resolve. Passes without /version route. |
| 6–13 | commit/build-info tests | DECORATIVE | Test build identity resolution. Passes without /version route. |
| 14 | `test_version_endpoint_shape` | REAL | Fails: /version returns 404 / dashboard SPA instead of JSON. |
| 15 | `test_version_matches_the_module_derivation` | REAL | Fails: /version is absent. |
| 16 | `test_health_keeps_its_existing_contract` | DECORATIVE | Tests /health contract. Passes without /version route. |
| 17 | `test_health_gains_the_capability_list` | DECORATIVE | Tests /health has capabilities. Passes without /version route. |
| 18 | `test_version_and_health_leak_nothing_sensitive` | REAL | Fails: /version endpoint is absent, causing JSONDecodeError on dashboard HTML. |
| 19 | `test_version_is_public_when_a_server_token_is_configured` | REAL | Fails: /version returns 404 / HTML instead of 200 JSON. |
| 20 | `test_health_stays_public_when_a_server_token_is_configured` | DECORATIVE | Tests /health public. Passes without /version route. |
| 21 | `test_data_plane_is_still_gated_alongside_the_public_version` | DECORATIVE | Tests /projects gated. Passes without /version route. |

**Decorative-test fix status:** The 17 passing tests test behaviour unrelated to the /version route (format, inclusion, probe, route markers, commit resolution, build-info, /health, data-plane gating). Their assertions are strong for what they test and cannot be strengthened to fail under a route-only revert. **No assertion fixes applied.**

---

## Summary

| PR | Revert | REAL | DECORATIVE | Decorative with fixable weak assertion |
|----|--------|------|------------|----------------------------------------|
| #212 | validation block removed | 8 | 5 | 0 |
| #213a | probe always True | 1 | 20 | 0 |
| #213b | /version route removed | 4 | 17 | 0 |

**Zero decorative tests with fixable weak assertions were found.** All tests that fail under their respective reverts have strong, specific assertions. Tests that stay green verify different behaviour (storage, format, route markers, commit resolution, build-info caching, /health contract, data-plane gating) and cannot be strengthened to fail without also reverting that other behaviour.

## Clean-master suite tails (for comparison)

```
$ uv run --extra dev pytest tests/test_a2a.py tests/test_version_capabilities.py -q
........................................................                 [100%]
56 passed in 21.40s
```
