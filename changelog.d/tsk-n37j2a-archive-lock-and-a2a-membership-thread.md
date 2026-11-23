### Fixed

- ArchiveStore now takes an exclusive advisory lock on the archive directory when opened for writing, using a lock file plus ``fcntl.flock`` on POSIX. A second writer in another process gets a clear ``RuntimeError`` naming the lock holder's pid. Read-only opens take no lock.
- ``a2a_membership.py``'s ``archive_membership_created`` and ``archive_membership_removed`` now accept an already-open :class:`~taosmd.archive.ArchiveStore` from the caller (the service) instead of constructing a second ``ArchiveStore`` on the same directory, eliminating the double-writer problem and ensuring line numbers stay strictly increasing.
- The service layer ``service.py`` threads the shared archive from ``_ensure_stores`` through to the membership archive functions.
