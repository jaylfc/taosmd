PR body — tsk-od3z2g (taosmd[mcp] dead on arrival with mcp 2.x)

## RED-FIRST proof (tests run against unchanged origin/master)

```
FAILED tests/test_mcp_server.py::TestMCPDependencyErrors::test_incompatible_mcp_version_is_reported
FAILED tests/test_mcp_server.py::TestMCPDependencyErrors::test_cli_prints_one_line_for_incompatible_version
FAILED tests/test_mcp_server.py::test_mcp_extra_is_pinned_below_2 - AssertionError
3 failed in 0.68s
```

- `test_incompatible_mcp_version_is_reported` failed on `ImportError: cannot import name 'IncompatibleMCPVersionError' from 'taosmd.mcp_server'` (the real call, not a collection error).
- `test_cli_prints_one_line_for_incompatible_version` failed on `assert 'mcp>=1,<2' in "error: the MCP server requires the optional 'mcp' SDK. Install it with: pip install taosmd[mcp]\n"` — the old "not installed" message, not a traceback.
- `test_mcp_extra_is_pinned_below_2` failed on `assert ['mcp'] == ['mcp>=1,<2']`.

## MUTATION test (the tests must catch today's behaviour)

`_require_fastmcp` reverted to raise `MissingMCPDependencyError` unconditionally on ImportError:
`FAILED tests/test_mcp_server.py::TestMCPDependencyErrors::test_incompatible_mcp_version_is_reported` — `taosmd.mcp_server.MissingMCPDependencyError: the MCP server requires the optional 'mcp' SDK. Install it with: pip install taosmd[mcp]`. Restored afterwards.

## Verification

- `uv run python -m pytest -q -p no:randomly tests/test_mcp_server.py tests/test_collections_mcp.py` → `12 passed in 1.04s`
- `uv run ruff check taosmd/ tests/ benchmarks/` → `All checks passed!`
- `python scripts/check_trailing_newline.py` → `trailing-newline-gate: clean`
- `git diff origin/master --stat` → only `pyproject.toml`, `taosmd/cli.py`, `taosmd/mcp_server.py`, `tests/test_mcp_server.py`, `changelog.d/tsk-od3z2g-mcp-extra-pin.md`

## Review blockers addressed

- BLOCKER: stray `taosmd/pyproject.toml` (a duplicate of the root one) — deleted; it was never in the card's file list.
- BLOCKER: no changelog.d fragment — added `changelog.d/tsk-od3z2g-mcp-extra-pin.md`.
- NIT: docstring/`__all__` updates — kept; they are accurate and the card's "MissingMCPDependencyError and _require_fastmcp only" was about the *code* surface, not the docs.

## One extra file

`taosmd/cli.py` — the CLI handler for `taosmd mcp` now catches `IncompatibleMCPVersionError` the same way as `MissingMCPDependencyError` (one-line `error:` message + exit 1, no traceback). Allowed by the card.

## Success path

The existing `tests/test_mcp_server.py` success-path tests ran, not skipped — mcp 1.30.0 is installed in the dev environment, so `test_registers_expected_tools` and the roundtrip tests exercised the real `build_server` and passed.