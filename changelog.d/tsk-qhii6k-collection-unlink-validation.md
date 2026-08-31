### Fixed

- CollectionStore.unlink now validates its ext_id argument (non-empty string), matching the guard already present in CollectionStore.link, closing the link/unlink asymmetry that permitted silent no-ops on empty, whitespace-only, or non-string ext_id values.