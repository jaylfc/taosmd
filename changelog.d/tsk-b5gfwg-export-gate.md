### Fixed
- The `deleted-symbols-gate` CI check now also catches names silently dropped from a module `__all__` while their top-level `def`/`class` still exists, the shape a mechanical merge resolution produces when one side of a conflicted `__all__` block is taken wholesale.
