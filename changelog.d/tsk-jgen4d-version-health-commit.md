### Fixed
- Version resolution now reads from pyproject.toml at import time, so GET /health and GET /version report the correct version (0.4.1) instead of the stale hard-coded value (0.4.0)
- GET /health now includes `commit` and `commit_source` fields matching GET /version, enabling staleness detection by commit comparison