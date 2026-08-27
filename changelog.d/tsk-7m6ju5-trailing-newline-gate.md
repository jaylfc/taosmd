### Added
- Trailing-newline gate added to prevent silent corruption of release notes from missing trailing newlines in changelog fragments and the benchmarks data README.
- The gate runs on every pull request via `.github/workflows/trailing-newline-gate.yml`, and against the repository's own tree in the test suite, so a fragment committed without its terminating byte fails both CI and a plain `pytest` run.
