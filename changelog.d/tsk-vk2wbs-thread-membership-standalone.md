### Fixed

- Standalone (no registry verifier) A2A thread membership read gate no longer blocks access to member-restricted threads. The membership read gate now applies ONLY when a registry verifier is configured; without a verifier, threads with membership rows remain fully readable to all callers, matching the pre-gate default behavior.
