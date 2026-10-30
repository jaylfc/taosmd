### Fixed

- `VectorMemory` in `embed_mode="qmd"` now creates and owns its own `httpx.AsyncClient` when `init()` is called without an explicit client, instead of leaving `self._http` as `None` (which made `_embed_qmd` return `[]` for every chunk, so `add()` stored nothing and `search()` fell back to FTS only). The owned client is closed by `close()`; a caller-supplied client is reused and never closed.
