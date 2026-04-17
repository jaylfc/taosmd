# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. Benchmark Integrity

Numbers are the product. Cook the protocol, not the result.

- Every published number names its git SHA, model, seed, and dataset. If it's not reproducible, it doesn't go in the README.
- Run on `feat/<bench-name>` branches. Merge to `dev` when verified, to `master` only when published.
- Don't skip failing categories, don't fold adversarial QAs into overall averages silently, don't cherry-pick runs. Report what happened.
- When someone else's benchmark disagrees with ours, assume the burden to re-run, not to rebut.

## 6. Fedora / Infra Hygiene

- Heavy inference and benchmarks run on the Fedora host (ssh host alias). The taosmd repo is at `/home/jay/taosmd`; models live under `models/`.
- **Never commit IPs, Tailscale addresses, private hostnames, or env-specific paths to git.** Use env vars (e.g. `TAOSMD_OLLAMA_URL`) with safe defaults like `localhost`.
- ONNX models (MiniLM, cross-encoder) run via `embed_mode="onnx"`, not qmd — qmd is the Orange Pi path, not Fedora's.

## 7. Architecture Landmarks

So a fresh pair of eyes gets oriented quickly:

- **Six holdings** — Hall of Records (archive, verbatim, immutable), Stacks (vector), Card Catalogue (KG with supersede chains), Reading Room (session catalog), Digests (crystals), Reference Desk (insights).
- **Librarian layer** is the LLM-assisted pre-retrieval tier (query expansion, intent classification, supersede checking, routing). Orchestrators gate these via `run_if_enabled()`; extractors stay pure.
- **Archive-first** is sacred: extracted facts can be revised, the raw transcript never is. Supersede, don't delete.
- Benchmarks: LongMemEval-S (97.0%, master), LoCoMo (in progress — `feat/locomo-benchmark`).

## 8. Running Tests

- Fast test run (skips legacy integration dir): `python3 -m pytest tests/ -q --ignore=tests/integration` locally or the Fedora equivalent. Do this before pushing benchmark/code changes; don't push on red.

## 9. No AI Indicators in Public Artifacts

Reinforces global policy. In this repo specifically:

- No "Generated with Claude" / "Co-Authored-By: Claude" in commits, PR bodies, or issue comments.
- No AI-shaped doc patterns (e.g., overly structured headers, em-dash abuse in user-facing copy).
- `CLAUDE.md` and `docs/specs/*` are internal conventions — fine. Public-facing `README.md` / `AGENTS.md` / commit messages must read like human engineering output.
