# taosmd Next Steps — April 2026

## Context

A /last30days sweep across Reddit, HN, and GitHub (Apr 2026) surfaced a shifted competitive landscape:

- **Genesys** (r/AI_Agents, Apr 14): claims 89.9% on **LoCoMo**, +22 pts above Mem0, open-sourced. Uses "causal graph instead of flat vector storage."
- **Mem0** is at 53K stars; **Letta** at 22K; **MemPalace** launched with 35.7K stars in 6 days.
- **Benchmark controversies**: Zep published a formal rebuttal to Mem0's paper (Mem0 reported Zep at 65.99% LoCoMo, Zep's reproduction says 75.14%). Separately `dial481/locomo-audit` retested Zep's 84% self-claim and got 58.44%.
- **Emerging critique** (Signet, r/AI_Agents): *"most agent memory systems are still just query-time retrieval — it breaks when relevant context is implicit."*
- **Red-team threat lists** now include "memory poisoning" alongside prompt injection.

**taosmd's position:** 97.0% on LongMemEval-S (SOTA on that benchmark) but **no published LoCoMo score**. LongMemEval is the minority benchmark; LoCoMo is the lingua franca.

## The four workstreams

### 1. Benchmark rigor (LoCoMo adoption + infrastructure cleanup)

**Why this ships first:** without a LoCoMo number, taosmd is invisible to the comparison tables everyone else is in.

**Methodology to match (from Mem0 paper `arxiv:2504.19413` and `snap-research/locomo`):**
- Dataset: `github.com/snap-research/locomo` → `data/locomo10.json` (10 conversations, 1,986 QAs, avg 300 turns / 9K tokens / 19 sessions per conversation)
- Categories: 1 single-hop, 2 temporal, 3 multi-hop, 4 open-domain, 5 adversarial
- Metrics: F1, BLEU-1, LLM-as-Judge (J). Per-category breakdown required.
- Standard practice: skip or separately report category 5 (adversarial).
- Ingest protocol: ingest full conversation (all sessions in order with timestamps), then ask QA questions, score answers against ground truth.

**Tasks:**
1. **Build `benchmarks/locomo_runner.py`** — ingest LoCoMo conversations via `index_day()`, ask QAs via `retrieve()` + LLM answer generation, score F1/BLEU-1/LLM-J per category.
2. **Run on Fedora** with the existing reference stack (MiniLM + cross-encoder + qwen3:4b for answer generation).
3. **Publish scores** to README in the same table shape Mem0/Zep/Letta use.
4. **Rewrite `benchmarks/longmemeval_runner.py`** — currently imports from dead `tinyagentos` monorepo (memory note confirms). Port to taosmd public API.
5. **Fix token cost metric in `eval/librarian_eval.py`** — replace `infx` ratio (divide-by-zero vs zero baseline) with absolute threshold `≤2000 tokens/query`.

**Success:** taosmd reports a full LoCoMo scorecard (F1/BLEU/J × 4 categories), and we know whether the librarian stack is competitive with Genesys's 89.9%.

### 2. Librarian mechanisms (validate the unvalidated)

**Why:** 16 prompts exist in `taosmd/prompts.py`, only 1 (query expansion) has a benchmark proving it earns its LLM cost.

**Tasks:**
1. **Supersede chain benchmark** — LoCoMo category 2 (temporal) is tailor-made for this. Test whether `is_fact_superseded_prompt` + supersede chains beats naive vector retrieval on temporal questions. This directly addresses Axis A's flat-scoring issue from the previous eval cycle.
2. **Routing benchmark** — LoCoMo has four category types (single/multi/temporal/open-domain). Test whether intent classifier (via `query_route_prompt`) correctly routes to different memory layers. This reclaims the Axis B concept in a real-world setting.
3. **Taxonomy confirmation** — verify `taxonomy_propose_prompt` + 3-use confirmation actually converges on sensible categories over a LoCoMo run. Report confirmed-taxonomy count after ingest.

**Success:** a second benchmark table showing per-Librarian-task contribution (ablation): vector-only, +routing, +supersede, +taxonomy, full Librarian.

### 3. Retrieval/architecture (positioning + hardening)

**Tasks:**
1. **Reframe README Librarian section around Signet's critique.** Current README leads with the library metaphor; add a concrete "why not just RAG" frame: *"query-time retrieval breaks when the relevant context is implicit — the Librarian layer is how we fix that."* Cite the public critique without naming Signet directly (they're a competitor).
2. **Causal/temporal graph framing.** taosmd has `KnowledgeGraph` (Card Catalogue) with supersede chains — that's the same category as Genesys's "causal graph." Surface this in the architecture diagram section, not just as implementation detail.
3. **Memory poisoning positioning.** `retrieve(verify=True)` already exists plus archive-first immutability. Add a short "Security" section to README calling out: archive-first transcripts are append-only (no poisoning vector), `verify` flag optionally validates retrieved facts against the archive before returning, extracted facts live in supersede chains (never silently overwritten).

**Success:** README reads like it's engaging with the 2026 memory-systems conversation, not the 2024 one.

### 4. Ecosystem/visibility

**Tasks:**
1. **Imran Ahmad AMA (Apr 24, r/LLMeng)** — prep a thoughtful memory/planning question that references taosmd organically. He wrote "30 Agents Every AI Engineer Must Build"; his interest in memory architecture is the right hook.
2. **Publish a methodology post** riding the Zep/Mem0 controversy — not attacking them, but showing taosmd's bench protocol (reproducible, open fixtures, dev-branch-first, single seed). "Receipts-heavy honesty" travels; Zep's rebuttal proved it.
3. **Target-market subreddits**: r/LocalLLaMA, r/AI_Agents, r/LangChain. Don't post until (1) LoCoMo numbers are published and (2) README is reframed.

**Success:** organic first-100-stars via substance rather than celebrity launch.

## Execution order

**Phase A — unblocks everything (today/tomorrow):**
- `benchmarks/locomo_runner.py` built and running on Fedora
- First LoCoMo result in hand (even if imperfect)

**Phase B — publish + reframe (while results come in):**
- README LoCoMo table + Librarian critique reframe + Security section
- Fix token cost metric
- Rewrite `longmemeval_runner.py`

**Phase C — deepen the moat (next):**
- Supersede-chain benchmark (LoCoMo category 2 ablation)
- Routing benchmark (LoCoMo category 1 vs 2 vs 3 vs 4 ablation)
- Taxonomy convergence report

**Phase D — visibility:**
- AMA participation
- Methodology post
- Subreddit posts with published numbers

## Risks & guardrails

- **Don't fudge LoCoMo scores.** Publish whatever comes back, in the same honesty tradition that makes the Zep/Mem0 controversy look bad for Zep. Include `run_seed`, commit SHA, and raw answer log.
- **Category 5 (adversarial) handling.** Mem0 skips; we should report separately with a note — standard practice.
- **Don't overpromise timelines.** This spec covers four workstreams. Phase A ships this week; everything else gets its own checkpoint.
- **Memory poisoning framing.** Don't imply taosmd is bulletproof — framing is "archive-first + verify gives you a primitive to build against poisoning," not "we solved it."

## Open questions

1. **Answer-generation LLM for LoCoMo.** Mem0 paper used GPT-4o-mini. Our reference stack is qwen3:4b (local). Cheaper/private but may score lower. Do we also run one GPT-4.1 pass for an apples-to-apples number? Decide after Phase A baseline.
2. **Category 5 policy.** Report it, skip it, or report separately with "unanswerable accuracy" as its own metric?
3. **Librarian task ablation cost.** Full matrix (5 configs × 4 categories × 1,986 QAs × LLM answer generation) is expensive. Budget-cap via sampling or run full matrix once?
