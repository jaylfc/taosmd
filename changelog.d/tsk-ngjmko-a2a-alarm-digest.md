### Added

- **Alarm_key convergence**: A2A messages may carry an ``alarm_key`` (stable subject+condition string) and optional ``alarm_fingerprint``. Same-(key, fingerprint) alarms inside the key's min-interval are deduped: the second sends ``{"deduped": true}`` and is not stored. Enforced atomically via unique key+fingerprint check. ``POST /a2a/alarms/{key}/clear`` re-arms the key, resetting the cooldown so new alarms with that key are stored.

- **Server-side digest coalescing**: Non-mention owned-thread traffic coalesces into a per-thread digest event flushed on a 30-minute boundary. SSE stream and `/a2a/feed` deliver a ``kind: digest`` event containing batched message IDs and bodies, so a woken consumer never re-fetches what it was woken for. This promotes the client-side wake diet of 2026-08-17 into bus policy so every harness inherits it.

- **``kind`` field in A2A messages**: Every A2A message now carries a ``kind`` field: ``"alarm"`` when ``alarm_key`` is present, otherwise ``"chat"``. This is propagated through the SSE stream and feed so consumers can filter by message kind.