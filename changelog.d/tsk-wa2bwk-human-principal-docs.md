### Added
- Documented that `human_principal_ids` exempts listed canonical_ids from the registry revocation check AND from the fail-closed refusal that fires when the revocation feed has never loaded. The id is not validated as belonging to a human, so naming an agent canonical_id there silently disables that agent's revocation.
