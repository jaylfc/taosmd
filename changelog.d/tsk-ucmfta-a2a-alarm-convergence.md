### Added

- Server-enforced alarm dedup: `POST /a2a/send` with `kind="alarm"` plus `alarm_key` and optional `alarm_fingerprint` now suppresses duplicate alarms within the module-level min interval, returning `{"deduped": true}`. Dedup state is stored in `a2a_alarm_state` (unique index on `alarm_key, fingerprint`) so it survives restarts and is visible to every process.
- `POST /a2a/alarms/{key}/clear` re-arms an alarm key, allowing the next identical alarm to store before cooldown re-applies.
- `RemoteClient.a2a_send` now forwards `alarm_key` and `alarm_fingerprint` in the HTTP payload so remote-configured senders get the same dedup guarantees.
