### Fixed

- `benchmarks/e050_coverage.py` `main()` no longer passes an unsupported
  `path=` keyword to `argparse.add_argument`; `main()` now accepts `argv=None`
  and is directly testable.