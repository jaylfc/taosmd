### Added
- Model role resolution via config: new `models` section in `config.json` with roles `crystallize`, `reflect`, `query_expansion`, `cross_encoder_path`, `cross_encoder_provider`. Functions `config.get_model_role`/`set_model_role` and `generator_profiles.resolve_model` allow overrides without redeploy. Existing installs with no `models` section behave identically (same defaults resolved).
