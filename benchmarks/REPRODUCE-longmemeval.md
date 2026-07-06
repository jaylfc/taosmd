# Reproducing the LongMemEval-S 97.0% Recall@5 headline

This is the pinned recipe for the headline number taOSmd publishes on
LongMemEval-S. Anyone with the repo, the dataset, and the ONNX model can
re-run it and get the same result. It exists because the repo previously could
not reproduce its own headline: the dataset lived as an untracked file and the
exact recipe was never written down, which is how the number came to be
mislabelled for a while (see "What this number is" below).

## What this number is (and is not)

The headline is **97.0% Recall@5**, a *retrieval* metric: for each question,
does at least one gold answer session appear in the top 5 retrieved sessions.
It is the `query_expand` configuration of `longmemeval_enhanced.py`, scoring
485 of 500 questions.

It is **not** end-to-end QA accuracy and **not** an LLM-as-judge score. A
separate harness measures the genuine end-to-end Judge number (retrieve, then
generate an answer, then grade it); that is tracked as E-012 and is a much
lower number, because retrieval being solved does not mean the small local
generator answers correctly. Post-N-017 (PR #176 fixed a judge verdict-parser
bug that counted INCORRECT as a pass), the corrected full-500 Judge baseline
is 42.8% (qwen3.5:9b) / 51.2% (llama3.1:8b). Do not describe this Recall@5
figure as Judge accuracy.

The comparison point is MemPalace, which reports 96.6% Recall@5 on the same set
with the same metric, so the head-to-head is like for like.

## Prerequisites

- Dataset: `benchmarks/data/longmemeval_s_full.json` (500 questions). It is
  gitignored for size; see `benchmarks/data/README.md` for the identity, the
  sha256 pin, and how to obtain it.
- Embedding model: `models/minilm-onnx/` (all-MiniLM-L6-v2 exported to ONNX,
  384 dim). This is the same model family MemPalace uses, chosen so the
  comparison is fair.
- A Python env with the repo installed (`pip install -e .`) plus
  `onnxruntime`, `httpx`, `bm25s`, and `spacy`. No GPU and no Ollama are
  needed: this is pure CPU ONNX retrieval.

## The command

```bash
python benchmarks/longmemeval_enhanced.py --limit 500 --top-k 5
```

The runner sweeps five retrieval configurations and writes a timestamped JSON
to `benchmarks/results/enhanced_<YYYYMMDD_HHMMSS>.json`. The published headline
is the `query_expand` row.

## What the recipe does

For each question, fresh per question:

1. Ingest the haystack. Each session is reduced to its user turns, joined, and
   truncated to 512 characters, then embedded as one vector tagged with its
   `session_id`. (User-turns-only ingestion was the best strategy in prior
   ablations.)
2. Build the query. For `query_expand`, the question is extended with up to
   three entity keywords from `expand_query_fast`, plus preference terms from
   `extract_preferences` when the question is a preference question. Other
   configs differ here (temporal rerank, wider retrieve-then-trim, etc.).
3. Retrieve with `vmem.search(query, limit=5, hybrid=True)` (dense + BM25
   hybrid).
4. Score Recall@5: a hit if any `answer_session_id` is among the retrieved
   sessions' ids.

## Expected output (committed provenance)

The committed proof file is
`benchmarks/results/enhanced_20260413_133215.json`. Its five configurations:

| config           | Recall@5 | hits    |
|------------------|----------|---------|
| hybrid_baseline  | 96.6%    | 483/500 |
| query_expand     | 97.0%    | 485/500 |
| temporal_boost   | 96.6%    | 483/500 |
| combined_v2      | 97.0%    | 485/500 |
| wider_retrieval  | 96.2%    | 481/500 |

The published `query_expand` per-category breakdown:

| question type             | Recall@5 |
|---------------------------|----------|
| single-session-user       | 68/70    |
| multi-session             | 131/133  |
| single-session-preference | 27/30    |
| temporal-reasoning        | 127/133  |
| knowledge-update          | 78/78    |
| single-session-assistant  | 54/56    |

A fresh run reproduces these counts because retrieval here is deterministic
(fixed dataset, fixed ONNX weights, no sampling), so the result is independent
of the host it runs on.

## Reproduction log

- 2026-04-13: original run, `enhanced_20260413_133215.json` (query_expand
  485/500 = 97.0%). This file is the committed provenance above.
- 2026-06-14: independent re-run from the pinned recipe on the project bench
  host (CPU, MiniLM ONNX), file `enhanced_20260614_175453.json`. It reproduces
  the committed proof exactly: `query_expand` = 485/500 = 97.0% Recall@5, with
  the same five-config table (baseline 96.6, query_expand 97.0, temporal_boost
  96.6, combined_v2 97.0, wider_retrieval 96.2) and the same per-category counts
  (knowledge-update 78/78, multi-session 131/133, temporal-reasoning 127/133,
  single-session-user 68/70, single-session-assistant 54/56,
  single-session-preference 27/30). The repo now reproduces its own headline.
