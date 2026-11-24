### Fixed

- `taosmd migrate --check` now reports an unconverted vector store as
  PENDING and exits 1 instead of printing current: the new `vector_memory`
  step 3 (`vector_memory_dense_to_f32`) is registered in
  `taosmd/migrations.py` with a detect probe (marker carries
  `dense_format=f32` and no `embedding LIKE '[%'` rows), and `migrations.status`
  consults that probe so a store converted outside the runner is clean while
  a legacy JSON store stays pending. The step itself is a no-op apply: the
  conversion stays CLI-only (Card B's `taosmd migrate --convert-vectors`)
  and off the on-open path, so opening a legacy store still converts nothing
  implicitly.
