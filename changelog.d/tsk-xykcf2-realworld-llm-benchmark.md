### Fixed

- `benchmarks/realworld_llm_benchmark.py:300`: Changed bare `return` to `sys.exit(1)` when the LLM model is not available in Ollama, so that automated benchmark chains detect the refusal as a non-zero exit code instead of treating it as a clean run.