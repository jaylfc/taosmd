### Fixed

- ArchiveStore seq allocation is now atomic across instances and processes sharing one archive directory, preventing duplicate archive_index ids. Allocation runs inside a `BEGIN IMMEDIATE` transaction on the index db using a persisted counter, so concurrent stores serialize on the SQLite write lock; a failed raw write after allocation burns the seq (a gap) rather than reusing it.
