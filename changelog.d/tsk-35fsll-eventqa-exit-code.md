### Fixed
- EventQA runner now exits non-zero (`sys.exit(1)`) on refusal instead of bare `return`, so automated chains checking `$?` correctly read a refusal as a failure.
