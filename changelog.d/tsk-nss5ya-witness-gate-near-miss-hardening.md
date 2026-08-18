### Fixed

- Revised the near-miss witness token regex from ``#\s*WITNESS[^:](?=.*::)`` to ``#\s*WITNESS[^:](?=[^:]*:)``. This fixes the defect where de-marked markers without a ``::`` payload were silently dropped, and widens the regex to also catch prose carrying a plain colon without ``::``.