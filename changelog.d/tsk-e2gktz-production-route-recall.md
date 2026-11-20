### Added

- New test `tests/test_production_route_recall.py` that exercises the REAL POST /search route for every recipe and strategy, guarding against regressions where fast/minimal strategies skip the vector store for queries where the intent primary is not vector.
