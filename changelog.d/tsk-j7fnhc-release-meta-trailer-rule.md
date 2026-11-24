### Fixed

- `scripts/release_meta.py`: trailer detection now matches only trailer attempts (key at line start after markup stripping, followed by a colon-like char), not any prose mention of migration or manual steps. Canonical trailers still parse; non-canonical trailer attempts refuse with exit 2; `migrate`/`migrating` prose is ignored. Adds confusable-key homoglyph refusal and normalisation pin tests.
