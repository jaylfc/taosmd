### Fixed

- Trailer key match is now case-insensitive after stripping leading whitespace, list markers (-, *, +, >), and emphasis characters (**,* _, backticks)
- Near-miss trailer lines that match the pattern but are not exactly canonical raise ReleaseMetaError (CLI exit 2, no output file)
- Migration: one-way, Migration: reversible, and Manual-Step: non-empty text continue to work exactly as before
- Fenced code blocks () are now ignored entirely for trailer parsing
- previous_version now strips leading v ref (e.g., v0.5.0 -> 0.5.0)
