### Fixed
- The duplicate-definition gate now catches definitions inside module-level ``if``/``try``/``for``/``while``/``with`` blocks and same-name closures within one parent function, matching Python's actual binding rules. Same-name closures in different parents remain legal, and the ``try:`` / ``except ImportError:`` fallback pattern stays silent.
