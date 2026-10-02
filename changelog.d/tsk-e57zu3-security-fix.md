### Fixed
- A2A bus now uses "reject" mode instead of "redact" for secret filtering, preventing silent rewriting of message bodies
- Swift argument labels (e.g., `join(email:password:deviceName:)`) are no longer treated as credentials and are not redacted
- Generic API key pattern fixed to avoid matching Swift-like identifier patterns