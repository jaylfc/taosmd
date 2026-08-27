### Added
- Ruff is configured for pyflakes rules (`[tool.ruff.lint] select = ["F"]`), declared as a pinned dev dependency, and run as its own `ci.yml` step against `taosmd/`, so a dead import or dead local binding fails the job instead of accumulating.

### Fixed
- Cleared the 35 pyflakes findings that configuration exposed: 28 unused imports, 6 dead local bindings and one redefinition. Six apparent findings in `taosmd/__init__.py` were deliberate re-exports and are now declared in `__all__` rather than deleted.
- The five `docs/agent-jobs/` files now say `uv run ruff check`, matching how `ci.yml` invokes everything else. The previous `python3 -m ruff check` form was never broken; it failed only because ruff was not installed anywhere, which the dev dependency above fixes.
