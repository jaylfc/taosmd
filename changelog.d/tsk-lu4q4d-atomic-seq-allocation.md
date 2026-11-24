### Fixed

- ArchiveStore seq allocation is now atomic across instances and processes sharing one archive directory, preventing duplicate archive_index ids.
