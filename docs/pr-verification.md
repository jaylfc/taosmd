# PR verification on taOSmd

How a pull request on this repo gets verified before it is merged. Adopted 2026-07-28 from
Jay's fleet directive, after an evening in which four of six PRs in one merge queue had real
defects sitting behind genuinely green CI.

The rule this whole document exists to serve: **a check whose failure is indistinguishable
from success is not a check.** Everything below is arranged so that "the gate passed" and
"the gate never ran" can always be told apart.

## The split

Verification is two stages with a hard line between them.

### Stage 1: evidence gathering (cheap model)

Mechanical work only. A stage 1 pass fetches the PR branch into a worktree, runs the named
test files and pastes the output verbatim, performs scripted red-first reversions, lists
every bot review and inline comment with its `commit_id` against the current head, and
captures CI status per required context.

Stage 1 returns raw evidence and is **forbidden from concluding anything**. No verdicts, no
adjudication, no "looks fine". If a stage 1 report contains a judgement, the judgement is
discarded and the evidence is re-read.

### Stage 2: adjudication (lead model)

Reads the evidence and decides. Are the bot findings real or false? Is the defect material?
MERGE or BLOCK.

This is where the catches live. On the evening this split was adopted the real finds were a
PR whose branch silently contained a second unreviewed PR, an existence oracle moved from
the status code into the response body, and a credential leaked through a URL parse. None of
those are pattern matches. A cheap model does not make these calls.

Two of the loudest bot findings that same evening were **false**: an "auth bypass" that was a
reachability allowlist, and a "crash" that was guarded by a ternary two lines up. Bot output
is evidence for stage 2, never a verdict on its own.

## The zero-token layer

Three checks need no model at all and belong in the merge gate as scripts. They are being
built under `scripts/merge-gate/` (card `tsk-zgbkal`).

1. **Bot anchoring.** At least one substantive bot review whose `commit_id` equals the current
   head. A review anchored to an earlier sha means the head is unreviewed, no matter what the
   checks page says.
2. **Fake-green detection.** A CodeRabbit commit status reading `success` with the description
   "Review rate limited" means no review happened. So does the bare "Review finished"
   acknowledgement comment with no review object behind it. Note that a plain
   `@coderabbitai review` no-ops on a PR that was reviewed and then pushed to; only
   `@coderabbitai full review` forces a real pass.
3. **Red-first runner.** Mechanizes revert, fail, restore, pass. A test that stays green when
   the fix is reverted proves nothing. See issue #217 for a merged example: a test asserting
   `busy_timeout > 0` passes against plain `sqlite3.connect`, because SQLite's default is
   already 5000.

Every script prints `SUCCESS:` or `FAILED:` as its first line and uses a distinct non-zero
exit code per failure class. A timeout or a killed process yields no output and a generic
non-zero status, and the gate must be able to tell that apart from a real failure.

### Two ways the anchoring check gets built wrong

Both of these were found within hours by leads who had just built this gate, and neither was
caught by the gate's own unit tests. They are part of the acceptance contract for
`tsk-zgbkal`.

**GitHub rewrites `commit_id` on review comments.** For a pull-request review comment,
`.commit_id` is updated to the newest commit the comment still applies to. `.original_commit_id`
is where the bot actually looked. So `commit_id == head` is **true for stale reviews**, and a
check built on it reports "reviewed at head" for a review that happened several commits ago.
That is the exact failure the check exists to prevent, reimplemented as the check itself.

- Review **objects**: use `commit_id`. It is not rewritten. This is the primary signal.
- Inline **comments**: use `original_commit_id` only.

**The rate-limit marker is an HTML comment, not a status description.** CodeRabbit writes
`<!-- This is an auto-generated comment: rate limited by coderabbit.ai -->` into the PR
comment body. A detector looking only for a commit status described "Review rate limited"
matches nothing on a rate-limited PR and reports a clean sweep. Match the HTML-comment
string; that is the marker verified against real taOSmd data on PR #212.

### Prove the negative

Run the finished gate against PRs you already know are unreviewed, not only against ones you
expect to pass. Measured 2026-07-28: **9 of taOSmd's 15 most recently merged PRs were
CodeRabbit rate-limit-only at their merged head** (#212, #207, #196, #195, #193, #192, #191,
#190, #189). The gate must print `FAILED:` on every one of them. If it passes any, the gate is
wrong, not the PR.

Bot review coverage is a property to **measure per repo**, not to assume from a bot's presence
in the checks list.

## Local preconditions

`uv sync` installs the dev and test dependencies by default since PR #219 (PEP 735
`[dependency-groups]`). If `uv run pytest` reports "Failed to spawn: pytest", the environment
is wrong and any test result from it is meaningless. Say so rather than forcing the deps in
by hand, because a hand-forced environment hides the same breakage from the next person.

## Documentation drift gate

A pull request that changes feature code, an A2A handler, or contributor-surface files
(`.github/workflows/`, `pyproject.toml`) must also touch the doc that covers it, or carry a
`Docs-Reviewed: <why>` trailer. The check is `scripts/check_doc_gate.py`, wired in GitHub
Actions as `.github/workflows/doc-gate.yml`; the rule set lives in `docs/doc-gate.toml`.

The gate has two layers. Layer A (`invariants`) always runs against the tree: it verifies
that path tokens mentioned in shipped docs still exist, and that every protected doc still
contains the section headings declared under `[invariants.required_headings]` in
`docs/doc-gate.toml`. This is what catches a doc that has been silently gutted -- emptied of
the sections it exists to hold -- even when the change triggers no code rule. Layer B
(`diff-gate`) is the per-change rule engine: a rule fires on an added or deleted file (and,
for rules with `on_modify = true`, on a modification too), and is satisfied by editing its
`require_doc` file *with its required sections still present*, or by a `Docs-Reviewed:`
trailer. A touched doc is content-asserted, so a one-character edit cannot mask a gutted
section.

This gate applies to its own PR: adding `.github/workflows/doc-gate.yml` is a change to
`.github/workflows/`, so this section is the matching doc update.
