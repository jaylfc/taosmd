### Fixed

- Revised PR #230: fixed upgrade path broken on every existing install by removing source/source_id from INDEX_SCHEMA and changing migration guard from has_column to index_exists; also added _get_remote branch to a2a_import so remote-configured installs don't silently write history to local archive