### Added
- Four new tests in `tests/test_longmemeval_enhanced_arms.py` for LongMemEval enhanced retrieval arms:
  1. `test_vector_memory_kwargs_recorded`: Verifies VectorMemory constructor receives onnx_path and binary_quant from arm config.
  2. `test_rescore_reorders_via_float_cosine`: Proves rescore reorders results via float cosine (C > A > B) vs binary order (A, B, C), both directions in one test.
  3. `test_run_benchmark_writes_arm_config`: Runs real `run_benchmark` with monkeypatched DATA_PATH, __file__, VectorMemory, and env vars; asserts output arm_config matches env-derived config.
  4. `test_mrl_dim_vector_length`: Asserts stored float vectors have length equal to mrl_dim when set.
