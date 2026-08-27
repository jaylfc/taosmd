### Added

- `a2a_ack(message_id, by, *, data_dir=None)` records an acknowledgement as server-side state on the message envelope (`acked_by` list via `archive.update_event_data_json`), never as a new bus message. Acks are surfaced by the existing `a2a_feed` and `a2a_thread_messages` read paths and are idempotent per principal.
