### Fixed

- `taosmd migrate --check` now reports an unconverted vector store as
  PENDING and exits 1 instead of printing current: the
  `vector_memory_dense_to_f32` migration step (version 3) is registered in
  `taosmd/migrations.py` with a detect probe (the mode marker carries
  `dense_format=f32` and no row still holds a JSON embedding), and
  `migrations.status` consults that probe so a store converted outside the
  runner is clean while a legacy JSON store stays pending. The step's apply
  is a no-op: the conversion stays CLI-only (Card B's `taosmd migrate
  --convert-vectors`) and off the on-open path, so opening a legacy store
  still converts nothing implicitly.
