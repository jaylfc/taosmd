### Fixed
- Config setters with a `clear` bool before `data_dir` now make `clear` keyword-only, preventing silent data loss when a path is passed positionally.
