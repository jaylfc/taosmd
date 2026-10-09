### Added
- LongMemEval runner: `TAOSMD_LME_SAVE_CONTEXT=1` saves the assembled context per question for re-judging
- LongMemEval runner: `TAOSMD_JUDGE_MODE=evidence` adds retrieved context to the LLM judge prompt (default `reference` preserves byte-identical prompt)
- New `benchmarks/longmemeval_rejudge.py` CLI to re-score past runs with different judge modes and compute agreement