### Fixed
- `answer_context` now trims a session to the hit windows that fit instead of dropping the whole session when its full block exceeds the budget, so found evidence is no longer discarded with budget left.
