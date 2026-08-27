### Fixed
- The `deleted-symbols-gate` CI check now also catches re-exported names silently dropped from a package `__all__` (e.g. an entry in `taosmd/__init__.py` that is backed by `from .x import Name`) when the underlying `def`/`class` still exists at HEAD. Previously only same-file definitions were guarded, leaving the package's top-level public API invisible to the gate.
