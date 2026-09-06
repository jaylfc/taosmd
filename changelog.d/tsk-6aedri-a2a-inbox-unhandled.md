### Added

- `GET /a2a/inbox/unhandled` returns messages past the consumer's cursor that are addressed to the consumer and have not been acknowledged. Registry auth gates apply; `exclude_acked_by` is forwarded through the remote path so local and remote answers agree. `a2a_inbox_unhandled` is also exported from `taosmd.service`.

### Fixed

- `a2a_inbox` now propagates `acked_by` on envelopes when present, and accepts `exclude_acked_by` to omit acknowledged messages before the limit budget is applied.
