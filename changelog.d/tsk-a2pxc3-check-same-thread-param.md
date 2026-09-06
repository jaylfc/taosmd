### Added
- `_db.connect` now accepts a keyword-only `check_same_thread: bool = True` parameter, forwarded to `sqlite3.connect`. This lets a caller explicitly opt into a cross-thread connection without bypassing the shared helper. The default preserves sqlite3's thread-affinity safety check for every existing caller.

### Fixed
- Thread-affinity audit of the 16 sites that open a store via `_db.connect` (excluding ReceiptStore, which already bypasses with a raw `sqlite3.connect(..., check_same_thread=False)`, and `mentions.py:31`, carded separately as tsk-wxymim). None of the remaining 15 are genuinely cross-thread under ThreadingHTTPServer: every connection is created and used on the single `_ServiceLoop` service-loop thread, because every HTTP handler delegates all DB work through `runner.run()` / `runner.spawn()` and no handler opens a store directly. The 15 sites are single-threaded by design:
  - `archive.py:119`, `vector_memory.py:234`, `knowledge_graph.py:112`, `claims/store.py:42` -- created in `_ensure_stores` (api.py), cached in the module-level `_stores_cache`, and accessed only via `runner.run(service.xxx(data_dir=...))` on the service-loop thread.
  - `tasks.py:121` (`_get_db`, module-level `_db_cache`) and `tasks.py:776` (`rebuild_from_archive`) -- accessed only from `service.task_*` wrappers on the service-loop thread.
  - `pending_decisions.py:92` (api.py:915, api.py:951) -- opened and closed within a single service call on the service-loop thread.
  - `collections.py:184` (`CollectionStore`) -- opened and closed within a single `_collection_store()` call in the service-layer wrappers.
  - `access_tracker.py:54`, `browsing_history.py:39`, `crystallize.py:72`, `reflect.py:139`, `session_catalog.py:146` -- not instantiated in the HTTP dispatch table at all; only used in `auto_setup.py` (standalone) or `catalog_pipeline.py` (nightly cron, separate process with its own `asyncio.run`).
  - `taosmd_backend.py:91` and `taosmd_backend.py:336` -- `TaOSmdBackend` is a MemoryBackend implementation for the taOS Memory app, not wired into the HTTP service dispatch or any background worker.
