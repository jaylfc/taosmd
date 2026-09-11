### Fixed

- `POST /collections/{id}/grants` now rejects agent ids that are empty or
  whitespace-only, matching the guard already present on the corresponding
  `DELETE /collections/{id}/grants/{agent}` endpoint. Previously a padded
  agent id such as `" "` (single space) was accepted on grant but rejected on
  revoke, leaving an unrevokable grant. The HTTP handler and the store layer
  both normalise via `strip()`, so the two ends agree by construction.
