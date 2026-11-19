### Added

- New test `tests/test_production_route_recall.py` that exercises the real POST /search route for every recipe and strategy, verifying recall of seeded facts across question, keyword, and paraphrase query forms. Fast/minimal strategy cases where intent classification routes to KG first are marked xfail (tsk-rmebxw).
