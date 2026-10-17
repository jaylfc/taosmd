### Added
- Add reranker arm selection via TAOSMD_BENCH_RERANKER (off/ms-marco/bge-v2-m3) to longmemeval_enhanced.py; hard-fail when requested reranker is not loadable/available and pass reranker through run_question to rescore the pooled results.
