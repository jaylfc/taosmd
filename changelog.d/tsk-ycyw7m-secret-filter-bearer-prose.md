### Fixed

- bearer_token pattern no longer redacts prose containing "bearer" (e.g. "the bearer of bad news"). The regex now requires a token-shaped run of at least 16 characters with at least one digit or special character (._~+/).
