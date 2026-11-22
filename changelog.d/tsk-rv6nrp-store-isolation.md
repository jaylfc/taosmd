### Fixed
- Vector store failures no longer take down the A2A bus or archive. `_ensure_stores` now opens stores independently: archive first (must succeed), then vector and KG with per-store try/except, 30s retry cooldown, and error recording. `/health` reports per-store status and returns `status: degraded` when any store is unhealthy.
