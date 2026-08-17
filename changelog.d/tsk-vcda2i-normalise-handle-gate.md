### Fixed
- Skip ``@typing.overload`` and ``@overload`` stubs when counting ``_normalise_handle`` definitions, so the gate no longer fails on legitimate typed overload pairs.
- Catch ``UnicodeDecodeError`` in addition to ``OSError`` when reading files, so the gate no longer crashes on non-UTF-8 input its docstring already claimed it would survive.
