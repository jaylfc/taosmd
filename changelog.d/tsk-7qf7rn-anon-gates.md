### Fixed
- Catalog pipeline: anonymous callers (empty `agent_name`) now always run intake classification and catalog enrichment stages, matching the behavior of crystallize/reflect/fact-extraction gates. Previously the nightly cron silently disabled both stages for default installs.
