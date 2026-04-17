# Methodology transparency post — draft

**Status:** DRAFT. Hold until LoCoMo numbers are in and validated. Target posting: r/LocalLLaMA, r/AI_Agents, HN. Possibly a blog post on the taosmd GitHub readme.

**Framing principle:** do not attack Zep, Mem0, or anyone else. The post rides the *existence* of the benchmark controversy, not either side of it. The argument is about protocol, not personalities.

---

## Title options (pick one)

1. **"Publishing a memory-system benchmark is easy. Making it reproducible isn't. Here's what we did."**
2. **"taosmd on LoCoMo: [X]% F1, every number reproducible from a single commit"**
3. **"How to benchmark an agent memory system without cooking the numbers"** (spicier, higher click-through, more likely to get flamed)

---

## Post body (target 600-800 words)

### The lede

The agent-memory benchmark space had a rough month. Mem0 published a LoCoMo comparison. Zep published a rebuttal claiming Mem0 misconfigured their system and the real score was ~10 points higher. An independent audit (`dial481/locomo-audit`) then retested Zep's own 84% self-claim and reported 58.44%.

All three parties probably believe their numbers. That's the problem.

When a field's benchmark numbers can't be cross-checked by anyone outside the authoring team, the numbers stop being evidence. They become marketing. The lesson isn't "who was right" — it's that agent memory benchmarks need protocols, not just scoreboards.

### What we did when we measured taosmd on LoCoMo

taosmd hits **[X]% F1, [Y]% LLM-judge** on LoCoMo's standard categories (single-hop, multi-hop, temporal, open-domain). But the scorecard is not the interesting part. The interesting part is what sits underneath it:

**1. The benchmark commit is tagged.** Every published number names the exact git SHA it was produced by. Roll back to that SHA, run the command in the README, get the same number (modulo LLM non-determinism, which we document).

**2. The run command is a single line.** `python3 benchmarks/locomo_runner.py --run-id <tag>`. No hidden config, no private ingest preprocessor. The runner is [387 lines](https://github.com/jaylfc/taosmd/blob/master/benchmarks/locomo_runner.py) of argparse + retrieval + scoring. Read it in five minutes.

**3. The seed is fixed and printed.** Every aggregation uses a seeded RNG. If your run produces different numbers from ours, we want to know which seed and commit.

**4. The adversarial category (LoCoMo cat 5) is reported separately, not folded into the overall.** Mem0's paper skips it. Some systems fold "unanswerable" questions into the average in a way that inflates or deflates headline numbers. We report overall *and* per-category *and* adversarial as its own line. You decide what matters.

**5. The raw answer log is committed with the results.** Every predicted answer, every ground truth, every F1/BLEU/Judge triple — in the repo, under `benchmarks/results/locomo_<timestamp>.json`. If you disagree with a judge call, read the answer and argue.

**6. The LLM judge is documented and reproducible.** Judge = `qwen3:4b` at `temperature=0.0`. Same prompt every time. Not hidden behind an API contract with a provider that might change weights next month.

**7. When our numbers disagree with someone else's claim about us, the burden's on us to rerun, not to rebut.** If you benchmark taosmd and get a different number, file an issue with the command you ran and we'll find the discrepancy. This is how Zep's 84% should have been caught — not after publication, but during.

### The benchmark we'd rather win

We're also publishing a benchmark we haven't solved yet: **staleness detection** — the case where a user said "I use Flask" three months ago and "I'm on FastAPI now" last week, and retrieval surfaces both. Vector search returns the stale one; supersede chains should beat it. We have the mechanism (taosmd's `is_fact_superseded_prompt` + supersede chains in the KG) but the ablation benchmark is still coming. We'd rather show the number that surprises us than the number that flatters us.

### Why this matters beyond taosmd

If the agent-memory field runs on unreproducible scorecards, the winner will be the system with the best PR, not the best algorithm. The Zep/Mem0 argument isn't healthy. It's a symptom of infrastructure that lets both sides be plausible at once.

A minimum viable standard for anyone publishing an agent-memory benchmark in 2026:

- Single-command run script, public.
- Tagged commit SHA per number.
- Raw answer log committed with the result.
- LLM judge identified by name + version + temperature.
- Adversarial categories reported separately.
- Third-party reproduction welcomed, not litigated.

We're doing this. We'd rather have a [78]% number everyone can reproduce than a [91]% number nobody can.

---

## After publishing

- Monitor for third-party runs — make the issue tracker welcoming, not defensive.
- If someone posts a different number, thank them publicly, investigate, and fix the discrepancy visibly. This is the whole point.
- Don't engage in the Zep/Mem0 argument directly. The post is the engagement.

## What NOT to say

- Don't name-check Mem0, Zep, or Letta. Referring to "the benchmark controversy" is enough context for anyone who follows the field, and keeps us out of the fight.
- Don't claim taosmd is "more honest." Claim the *protocol* is more honest. Protocols are arguable; character attacks aren't.
- Don't promise new features in the same post. This is about the numbers and the methodology, not the roadmap.
