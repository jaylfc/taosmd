### Added

- Automated per-merge releases: `scripts/release_version.py` computes the `0.5.N` version from the `v0.5.0` anchor tag using first-parent commit counting, stamps the `version = "..."` line in `[project]` of `pyproject.toml`, and writes `taosmd/_build_info.py`; every merge to master that passes CI becomes its own numbered release.
