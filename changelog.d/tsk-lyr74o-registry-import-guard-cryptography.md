### Fixed

- Registry auth startup guard now checks that both `pyjwt` and `cryptography` are importable when `registry_url` is configured, not just `pyjwt`. A partial install (e.g. `pip install pyjwt` without its crypto extra) previously started the server silently and failed on the first EdDSA verification; the server now fails loudly at startup with an actionable `pip install taosmd[registry]` error.
