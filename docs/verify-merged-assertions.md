# Revert-and-rerun sweep: PR #212 and #213

## Label definitions

| Label | Meaning |
|-------|---------|
| **TARGETED** | The test FAILS when its own subject is perturbed (the rule it claims to guard is disabled). Verified by per-rule reverts (Finding 3) or by a direct perturbation of the subject. |
| **UNRELATED** | The test stays green under a specific perturbation because it exercises a *different* behaviour. This is the normal state of a healthy suite: not every test guards every code path. |
| **DECORATIVE** | **Reserved** for a test that cannot fail under *any* perturbation of its own stated subject -- i.e., a tautology. (Per the original `tsk-s2keyh` usage; none were found here.) |

The original report used **REAL** / **DECORATIVE**. That conflated two different
things: "DECORATIVE" was applied both to tautological tests *and* to tests of
unrelated behaviour. A property of a test cannot change depending on which
unrelated thing is broken (the same test was labelled REAL in the #213a table
and DECORATIVE in the #213b table). The correct word for the second case is
**UNRELATED**, not decorative. The totals below read "8 targeted, 5 unrelated"
rather than "8 real, 5 decorative".

**Zero decorative (tautological) tests with fixable weak assertions were found.**
All tests that fail under their respective reverts have strong, specific
assertions. Tests that stay green verify different behaviour and cannot be
strengthened to fail without also reverting that other behaviour.

---

## Environment verification
```
$ uv run python -c "import pytest, jwt, cryptography"
OK
```
Python 3.12.13, virtualenv created by `uv sync`.

## Experiment setup
Scratch worktree checked out at `origin/master`. Each experiment modifies only
the behaviour claimed by the PR, leaving imports intact. Every perturbation is
followed by a restore + re-run to confirm the suite returns to green (the GREEN
half of the RED-FIRST demonstration).

### Methodology: per-rule reverts (Finding 3)

The original #212 sweep removed the entire validation block at once (8 reds,
27 green). That is consistent with full coverage but cannot demonstrate it: if
any of the eight enforced rules had *no* test, the aggregate would look
identical, because the other seven tests would still fire.

The correct granularity is **per-rule**: perturb each rule to its degenerate
value separately, disable the check, and confirm that *only* the test targeting
that rule goes red. Eight small reverts instead of one big one. The difference
is "the block is tested" versus "each thing the block enforces is tested," and
only the latter can find a gap.

The eight rules enforced by the validation block in
`taosmd/http_server.py` `_handle_a2a_send`:

| # | Rule | Perturbation |
|---|------|-------------|
| R1 | refs is a list | disable `isinstance(refs, list)` |
| R2 | refs <= 8 items | disable `len(refs) > _A2A_MAX_REFS` |
| R3 | refs items are dicts | disable `isinstance(ref, dict)` |
| R4 | refs kind in enum | disable `kind not in _A2A_REF_KINDS` |
| R5 | blocks is a list | disable `isinstance(blocks, list)` |
| R6 | blocks items are dicts | disable `isinstance(block, dict)` |
| R7 | 64KB total cap | disable `len(serialized) > _A2A_MAX_MESSAGE_BYTES` |
| R8 | blocks => body non-empty | disable the `else` body-invariant branch |

### RED FIRST

For each rule the demonstration is: **RED** (rule disabled → its test fails),
then **GREEN** (rule restored → test passes). A perturbation that does *not*
turn the corresponding test red has measured nothing.

### Negative control

A perturbation that *should* leave every validation test green actually does:
changing the unrelated `server_version` string in `_make_handler` (used for the
`Server` HTTP header) leaves all 8 validation tests green, confirming the suite
distinguishes relevant from irrelevant changes.

---

## PR #212 -- envelope refs+blocks validation block

**Code under test:** the eight rules above, lines 1510-1551 of
`taosmd/http_server.py` `_handle_a2a_send`.

### Per-rule sweep results

Each row below is a *single-rule* perturbation. "RED test" is the one test that
fails; "other fails" is always zero (no coupling).

| Rule | Perturbation | RED test | Failure mode | Restored | GREEN |
|------|-------------|----------|-------------|----------|-------|
| R1 | refs-is-list check disabled | `test_http_a2a_refs_not_list_returns_400` | 500 (TypeError: `len(42)`) instead of 400 | yes | 8 passed |
| R2 | refs-max-8 check disabled | `test_http_a2a_refs_too_many_returns_400` | 200 (9 refs stored) instead of 400 | yes | 8 passed |
| R3 | refs-item-dict check disabled | `test_http_a2a_refs_item_not_dict_returns_400` | 200 (non-dict ref stored) instead of 400 | yes | 8 passed |
| R4 | refs-kind-enum check disabled | `test_http_a2a_refs_bad_kind_returns_400` | 200 (bad kind stored) instead of 400 | yes | 8 passed |
| R5 | blocks-is-list check disabled | `test_http_a2a_blocks_not_list_returns_400` | 500 (TypeError: `'int' object is not iterable`) instead of 400 | yes | 8 passed |
| R6 | blocks-item-dict check disabled | `test_http_a2a_blocks_item_not_dict_returns_400` | 200 (non-dict block stored) instead of 400 | yes | 8 passed |
| R7 | 64KB cap disabled | `test_http_a2a_message_too_large_returns_400` | 200 (oversized message stored) instead of 400 | yes | 8 passed |
| R8 | blocks=>body invariant disabled | `test_http_a2a_blocks_without_body_returns_400` | 400 but error message lacks `blocks` (service-layer ValueError) | yes | 8 passed |

**All 35 tests in `test_a2a.py`:** baseline 35 passed. Under each per-rule
perturbation: exactly 1 failed, 34 passed. No rule removal caused an unexpected
secondary failure.

### Note on R1 and R5 test inputs

The per-rule sweep originally showed R1 and R5 as *false negatives*: disabling
the `isinstance` check did not turn their tests red, because the original test
inputs used the string `"not a list"`, which has `len("not a list") == 10 > 8`
(R2 catches it first) and iterates into characters (R3/R6 catch it first). The
is-a-list check was not exercised in isolation.

The tests were fixed to use non-iterable values (`42`) instead of strings, so
that disabling only the `isinstance` check causes a `TypeError` (500) that the
test correctly fails on. This is the gap the all-or-nothing revert could not see.

### Tests added by PR #212 (13 tests)

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `test_a2a_send_with_refs_and_blocks_roundtrip` | UNRELATED | Tests service-layer storage roundtrip with valid envelope. Stays green under all 8 per-rule reverts. |
| 2 | `test_a2a_send_without_refs_blocks_omits_keys` | UNRELATED | Tests key-omission behaviour. Stays green. |
| 3 | `test_http_a2a_send_refs_blocks_roundtrip` | UNRELATED | Tests HTTP storage roundtrip. Stays green. |
| 4 | `test_http_a2a_send_without_refs_blocks_omits_keys` | UNRELATED | Tests HTTP key-omission. Stays green. |
| 5 | `test_http_a2a_sse_with_refs_and_blocks` | UNRELATED | Tests SSE storage roundtrip. Stays green. |
| 6 | `test_http_a2a_refs_not_list_returns_400` | TARGETED | Fails under R1 (RED). Fixed input isolates the isinstance check. |
| 7 | `test_http_a2a_refs_too_many_returns_400` | TARGETED | Fails under R2 (RED). |
| 8 | `test_http_a2a_refs_item_not_dict_returns_400` | TARGETED | Fails under R3 (RED). |
| 9 | `test_http_a2a_refs_bad_kind_returns_400` | TARGETED | Fails under R4 (RED). |
| 10 | `test_http_a2a_blocks_not_list_returns_400` | TARGETED | Fails under R5 (RED). Fixed input isolates the isinstance check. |
| 11 | `test_http_a2a_blocks_item_not_dict_returns_400` | TARGETED | Fails under R6 (RED). |
| 12 | `test_http_a2a_message_too_large_returns_400` | TARGETED | Fails under R7 (RED). |
| 13 | `test_http_a2a_blocks_without_body_returns_400` | TARGETED (status duplicated) | Fails under R8 (RED), but only on the **message** assertion. See special note below. |

### Row 13 special note (Finding 2)

`test_http_a2a_blocks_without_body_returns_400` is counted as TARGETED, but
its load-bearing surface is narrower than the other seven.

Its docstring now states: the `assert status == 400` half is **duplicated** at
the service layer -- `service.a2a_send` raises `ValueError` for an empty body
independently. Disabling R8 still returns 400 (from the service-layer ValueError);
the only thing that changes is the error message: `body must be a non-empty string`.
The test asserts
`"blocks" in body["error"]`, which is the **message** assertion -- the only
check that is unique to the HTTP-level invariant. The status-code assertion
cannot fail without the service layer also failing, so a future "tidying" change
to `assert status == 400` would silently delete the coverage without turning
anything red.

A comment was added to the test itself documenting this: the message assertion
is the part that matters.

### UNRELATED-test fix status

The 5 roundtrip tests (rows 1-5) stay green under all 8 per-rule reverts
because they exercise the storage path with valid inputs, not the validation
path. Their assertions (`status == 200`, refs/blocks equality) are correct for
roundtrip testing and cannot be strengthened to fail under a validation-only
revert without changing their purpose. **No assertion fix applied.**

---

## PR #213 -- /version endpoint + capabilities probe

Two independent reverts were applied separately.

### #213a -- capability probe broken (`_resolves` always returns True)

**Behaviour reverted:** `taosmd/capabilities.py` `_resolves` changed to
unconditionally return `True`, so capabilities are advertised even when their
backing symbols are missing.

**Perturbation:** replaced `_resolves` body with `return True`.

**Suite tail (probe broken):**
```
FAILED tests/test_version_capabilities.py::test_capability_is_dropped_when_its_backing_symbol_is_missing
1 failed, 20 passed
```

### RETRACTION (prominent, per smaller-notes request)

The original PR card claimed
`test_capability_declarations_do_not_diverge_from_the_http_surface` was
"verified to FAIL by pointing a capability at a bogus route." **It does not.**
Breaking the probe leaves it green: it tests route-marker presence in the
dispatcher source, not probe resolution. This is the kind of correction these
sweeps exist to produce and is the most valuable single line in this document.

**Table -- tests added by PR #213 (21 tests, probe revert):**

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `test_capabilities_are_contract_identifiers_with_version_suffix` | UNRELATED | Tests name format. Stays green with broken probe: suffixes unchanged. |
| 2 | `test_capabilities_include_collections_and_grants_on_this_build` | UNRELATED | Tests specific caps present. Stays green: caps still advertised. |
| 3 | `test_capability_is_dropped_when_its_backing_symbol_is_missing` | TARGETED | Fails: collections.v1 remains advertised after its symbol is deleted. |
| 4 | `test_capability_declarations_do_not_diverge_from_the_http_surface` | UNRELATED | Tests route markers in dispatcher source, NOT probe resolution. Stays green with broken probe despite the PR card's earlier claim that it "verified to FAIL." See retraction above. |
| 5-21 | commit/build-info/health/version/token-data-plane tests | UNRELATED | None depend on probe resolution; all pass with broken probe. |

### #213b -- /version route registration removed

**Behaviour reverted:** removed the `elif method == "GET" and path == "/version"`
handler from `taosmd/http_server.py`.

**Perturbation:** deleted the `/version` route handler block.

**Suite tail (/version route removed):**
```
FAILED tests/test_version_capabilities.py::test_version_endpoint_shape
FAILED tests/test_version_capabilities.py::test_version_matches_the_module_derivation
FAILED tests/test_version_capabilities.py::test_version_and_health_leak_nothing_sensitive
FAILED tests/test_version_capabilities.py::test_version_is_public_when_a_server_token_is_configured
4 failed, 17 passed
```

**Table -- tests added by PR #213 (21 tests, route revert):**

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `test_capabilities_are_contract_identifiers_with_version_suffix` | UNRELATED | Tests name format. Stays green without /version. |
| 2 | `test_capabilities_include_collections_and_grants_on_this_build` | UNRELATED | Tests caps present. Stays green without /version. |
| 3 | `test_capability_is_dropped_when_its_backing_symbol_is_missing` | UNRELATED | Tests probe drops missing caps. Stays green without /version. |
| 4 | `test_capability_declarations_do_not_diverge_from_the_http_surface` | UNRELATED | Tests route markers in source. Stays green without /version. |
| 5 | `test_every_declared_capability_probe_resolves_on_this_build` | UNRELATED | Tests all probes resolve. Stays green without /version. |
| 6-13 | commit/build-info tests | UNRELATED | Tests build identity. Stays green without /version. |
| 14 | `test_version_endpoint_shape` | TARGETED | Fails: /version returns 404/dashboard SPA instead of JSON. |
| 15 | `test_version_matches_the_module_derivation` | TARGETED | Fails: /version absent. |
| 16 | `test_health_keeps_its_existing_contract` | UNRELATED | Tests /health contract. Stays green without /version. |
| 17 | `test_health_gains_the_capability_list` | UNRELATED | Tests /health has capabilities. Stays green without /version. |
| 18 | `test_version_and_health_leak_nothing_sensitive` | TARGETED | Fails: /version absent -> JSONDecodeError on dashboard HTML. |
| 19 | `test_version_is_public_when_a_server_token_is_configured` | TARGETED | Fails: /version returns 404/HTML instead of 200 JSON. |
| 20 | `test_health_stays_public_when_a_server_token_is_configured` | UNRELATED | Tests /health public. Stays green without /version. |
| 21 | `test_data_plane_is_still_gated_alongside_the_public_version` | UNRELATED | Tests /projects gated. Stays green without /version. |

### UNRELATED-test fix status

The 17 passing tests either test unrelated behaviour (format, route markers,
commit resolution, build-info caching, /health contract, data-plane gating) or
test a different aspect of capabilities (probe resolution, route divergence).
Their assertions are strong for what they test and cannot be strengthened to
fail under a route-only revert without changing their purpose. **No assertion
fixes applied.**

---

## Summary

| PR | Perturbation | TARGETED | UNRELATED | DECORATIVE (tautological) | Decorative with fixable weak assertion |
|----|-------------|----------|------------|---------------------------|----------------------------------------|
| #212 | 8 per-rule reverts | 8 | 5 | 0 | 0 |
| #213a | probe always True | 1 | 20 | 0 | 0 |
| #213b | /version route removed | 4 | 17 | 0 | 0 |

**Zero decorative (tautological) tests with fixable weak assertions were found.**
All tests that fail under their respective reverts have strong, specific
assertions. Tests that stay green verify different behaviour (storage, format,
route markers, commit resolution, build-info caching, /health contract,
data-plane gating) and cannot be strengthened to fail without also reverting
that other behaviour.

## Clean-master suite tails (for comparison)

```
$ uv run --extra dev pytest tests/test_a2a.py tests/test_version_capabilities.py -q
........................................................                 [100%]
56 passed in 21.28s
```

### Negative control

Perturbation: change the unrelated `server_version` string in `_make_handler`.
All 8 validation tests stay green (`8 passed`), confirming the suite
distinguishes relevant from irrelevant changes.

```
$ uv run --extra dev pytest tests/test_a2a.py -q --tb=no \
  -k "test_http_a2a_refs_not_list or test_http_a2a_refs_too_many or \
test_http_a2a_refs_item_not_dict or test_http_a2a_refs_bad_kind or \
test_http_a2a_blocks_not_list or test_http_a2a_blocks_item_not_dict or \
test_http_a2a_message_too_large or test_http_a2a_blocks_without_body"
8 passed
```

Changing an unrelated string leaves every validation test green, confirming
the per-rule reds are signal, not noise.
