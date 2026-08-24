### Added

- `taosmd.service.a2a_inbox` and `a2a_inbox_advance`: server-side consumer cursors and inbox query for the A2A bus, with persisted cursor state in the archive store and default exclusion of alarm, ack, receipt, and digest kinds.
