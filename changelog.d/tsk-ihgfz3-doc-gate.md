### Added

- Documentation-drift gate (`scripts/check_doc_gate.py` with `.github/workflows/doc-gate.yml` and `docs/doc-gate.toml`) that blocks PRs changing taosmd code, the A2A handlers (`taosmd/http_server.py`, `taosmd/service.py`), or contributor-surface files (`.github/workflows/`, `pyproject.toml`) without a matching doc update or a `Docs-Reviewed:` trailer; protected docs are asserted on their required section headings so they cannot be silently gutted by a path-only rule or a one-character edit.
