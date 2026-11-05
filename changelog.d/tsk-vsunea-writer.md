### Added

- **`--writer-url` and `--writer-model` CLI flags** for the LongMemEval runner (`benchmarks/longmemeval_runner.py`): `--writer-url` specifies an OpenAI-compatible base URL for the LLM writer at ingest time (POST to `<url>/v1/chat/completions`); `--writer-model` sets the model name. When `--writer-url` is empty, the regex-only fact extraction path is used (back-compatible). When set, an `httpx.AsyncClient` is created for the writer and passed to `process_conversation_turn` with `llm_url`, `http_client`, and `extraction_model` kwargs.

- **Method counting** (`benchmarks/longmemeval_runner.py`): The `method` field of `process_conversation_turn` return dicts is counted across the run (`method_llm` and `method_regex` totals), stored in `result_doc["writer"]`, and printed as `writer: model=<m> llm=<a> regex=<b>`.

- **RED-FIRST tests** (`tests/test_longmemeval_retrieve_wiring.py`): Three new tests verifying (a) with `writer_url` empty, `process_conversation_turn` is called without `llm_url`; (b) with `writer_url` set, it receives `llm_url`, `http_client` and `extraction_model`; and (c) `result_doc["writer"]` correctly counts returned methods.