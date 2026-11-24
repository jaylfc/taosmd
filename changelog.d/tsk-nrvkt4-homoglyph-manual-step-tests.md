### Fixed
- Homoglyph rule now covers all Manual-Step variants (Manual-Step, Manual_Step, ManualStep, Manual Step, manual steps) and Migration keys beyond the first word, refusing confusable Cyrillic/Greek lookalikes in any position of the key.

### Added
- Tests for homoglyph refusal on Manual-Step variants and Migration keys (Cyrillic/Greek lookalikes).
- Tests for Manual-Step spelling variants parsing (underscore, no separator, space, plural).
