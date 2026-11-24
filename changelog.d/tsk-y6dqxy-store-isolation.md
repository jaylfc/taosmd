### Fixed
- Carry forward store isolation (#723 + #732): stores open independently, `/health` reports per-store status and returns `status: degraded` when any store is unhealthy, and a repaired store recovers on a later request via the retry-on-cache-hit path.
