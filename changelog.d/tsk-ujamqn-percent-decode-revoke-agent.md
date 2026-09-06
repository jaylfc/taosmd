### Fixed

- `DELETE /collections/{id}/grants/{agent}` now percent-decodes the `{agent}`
  path segment before looking up the grant, so agent ids containing spaces,
  pluses, slashes, non-ASCII bytes, or literal percent signs round-trip
  correctly through the revoke endpoint.
