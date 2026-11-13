### Fixed

- `archive.compress_old_files` now verifies each `.gz` round-trips byte-equal to the source `.jsonl` before unlinking it: the gz is written to a `<name>.jsonl.gz.tmp`, flushed and `os.fsync`'d, re-opened and decompressed for comparison, then `os.replace`'d onto the final path with a directory `fsync`. A mismatch or any exception removes the temp, keeps the `.jsonl`, logs a warning, and is not counted. Previously the source was unlinked immediately after writing the gz, so a short write or power cut could lose a day of raw transcripts.
