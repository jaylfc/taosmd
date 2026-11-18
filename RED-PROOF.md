```text
FAILED tests/test_version_capabilities.py::test_health_reports_commit_matching_version_route - AssertionError: health missing commit
FAILED tests/test_version_capabilities.py::test_version_matches_pyproject - AssertionError: 0.4.0 != 0.4.1
```

```text
FAILED tests/test_version_capabilities.py::test_health_reports_commit_matching_version_route - AssertionError: health missing commit
```

```text
.........................
25 passed in 5.55s
```

```text
All checks passed!
```

```text
trailing-newline-gate: clean
```

```text
 taosmd/__init__.py                 |  45 +++++++++++++++-
 taosmd/http_server.py              |  12 +++--
 tests/test_version_capabilities.py | 102 +++++++++++++++++++++++++++++++++++--
 changelog.d/tsk-jgen4d-version-health-commit.md |  5 +++++
 4 files changed, 157 insertions(+), 7 deletions(-)
```