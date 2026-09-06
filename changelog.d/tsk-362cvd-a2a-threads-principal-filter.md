### Fixed

- `GET /a2a/threads?principal=` now applies a sender-derived filter, returning only threads the named principal has sent to. Previously the parameter was accepted and silently ignored, causing every thread to be returned.
