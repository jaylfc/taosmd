### Fixed: identity comparison and revocation check normalised

- Fixed `taosmd/registry_auth.py:authorize_sender` to normalise handle identity
  using :func:`taosmd.service._normalise_handle` in both the ``sub != claimed_from``
  comparison (line 83) and the revocation check (line 92), so ``@``-prefixed and
  bare handles compare equal case-insensitively across both code paths.

### Added: taosmd/mentions.py

- Created `taosmd/mentions.py` with `MentionStore` class for recording and
  querying ``@``-handle mentions on the A2A bus, enabling the
  ``/a2a/mentions`` feed and thread-scoped visibility anti-bypass rule from
  taOSmd ``#211``. The store normalises handles via `:func:`_normalise_handle``
  so write and read paths agree on handle spelling.
