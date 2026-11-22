### Fixed
- `import taosmd` no longer fails on Python 3.10: tomllib is no longer imported at the top of taosmd/__init__.py, `_resolve_version` imports it lazily with a stdlib line-reader fallback, only trusts a pyproject whose [project] name is taosmd, and degrades any importlib.metadata error to `0+unknown`.
