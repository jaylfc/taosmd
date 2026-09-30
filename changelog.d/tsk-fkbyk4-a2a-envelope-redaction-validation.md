### Fixed

- Recursively redact secrets inside A2A `refs` and `blocks` envelope fields before storage, so nested tokens no longer leak into the JSONL archive or SQLite index.
- Move A2A envelope validation (ref kind enum, max ref count, list/object shape, 64KB total cap) into a shared helper called from `service.a2a_send()` and `service.a2a_import()`, with the HTTP handler mapping the same `ValueError` to `_BadRequest` so the 400 response shape is unchanged.
