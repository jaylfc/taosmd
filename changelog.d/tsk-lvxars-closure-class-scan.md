### Fixed
- The duplicate-definition gate now descends into classes defined inside a function body (`factory` closures) instead of skipping them, so a redefined class within one closure is reported while the same class name in different closures still does not collide. The `if`/`elif`/`else` and `try`/`except` sibling-arm behaviour is unchanged.
