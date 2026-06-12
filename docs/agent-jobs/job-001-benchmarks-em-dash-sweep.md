# JOB-001: Replace every em dash in docs/benchmarks.md

Read docs/agent-jobs/README.md first and follow its absolute rules.

- Branch: `docs/benchmarks-em-dash-sweep` (from `origin/master`)
- Commit message: `docs(benchmarks): replace em dashes with plain punctuation`
- PR title: `docs(benchmarks): replace em dashes with plain punctuation`
- Allowed files: `docs/benchmarks.md` ONLY.

## Why

House style forbids em dashes. docs/benchmarks.md predates the rule and
carries about 121 of them. Numbers and content must not change, punctuation
only.

## Steps

1. `git fetch origin && git checkout -b docs/benchmarks-em-dash-sweep origin/master`
2. Count first: `grep -c "—" docs/benchmarks.md` (expect roughly 121; record
   the exact number for the PR body).
3. Replace each em dash by hand-reviewing context. Rules of thumb:
   - ` — ` joining two clauses becomes `: ` when the second clause explains
     the first, otherwise `, `.
   - ` — ` introducing an aside becomes ` (aside)` parentheses only when the
     aside is short, otherwise a comma.
   - `word—word` with no spaces becomes `word, word`.
   - In table cells, prefer a comma or splitting into two sentences.
   A plain `sed` of all of them to one character is NOT acceptable: read each
   line. Work in batches of about 20 and re-read your diff after each batch.
4. Verify: `grep -c "—" docs/benchmarks.md` must print `0`.
5. Verify no content drift: `git diff --word-diff docs/benchmarks.md | grep -E "^\[-[0-9]" ` must print nothing (no number was touched). Also eyeball
   `git diff --stat`: one file changed, insertions equal deletions or close.
6. Run the suite (it does not test docs, but rule 7 is unconditional):
   `python3 -m pytest -q -m "not slow"`
7. Commit (one commit), push, open the PR. PR body: the before-count, the
   after-count (0), and one sentence stating numbers were not modified.

## Verification the reviewer will run

`grep -c "—" docs/benchmarks.md` is 0; word-diff shows no digit changes.
