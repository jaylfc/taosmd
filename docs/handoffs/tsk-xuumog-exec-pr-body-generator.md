# Hand-off: exec PR bodies misstate their diffs (tsk-xuumog)

Status: HAND-OFF. The generator of the `Files:` block and acceptance narrative
in exec PR bodies does not live in `jaylfc/taosmd`. It is part of the taOS
executor/lane system that opens exec PRs. This card cannot be built here, so it
is filed as a hand-off for the executor owner to act on.

## What is wrong

Seven consecutive exec PRs shipped a `Files:` block and acceptance narrative
that do not match the actual diff: #340, #341, #345, #346, #347, #348, #357.
Six omit a suite count or a proof the card demanded and read as done; #357 is
the sharpest case because its file list is another PR's manifest, not a
rounding error.

PR #357 (`exec/tsk-r44fqf`, card `tsk-r44fqf`) body says:

```
Files:
 taosmd/mentions.py                             |  92 +++
 taosmd/remote.py                               |  20 +
 taosmd/service.py                              | 168 +++++-
 tests/test_a2a_mentions.py                     | 646 +++++++++++++++++++++
 tests/test_api.py                              | 292 +++++++++-
 tests/test_collections_ingest.py               | 148 +++++
 tests/test_normalise_handle_gate.py            | 162 ++++++
 24 files changed, 1863 insertions(+), 38 deletions(-)
```

Its real diff (verified against the GitHub API):

```
benchmarks/data/README.md                     | 14 +++++++++++---
changelog.d/tsk-r44fqf-unverify-s-full-pin.md |  2 ++
2 files changed, 13 insertions(+), 3 deletions(-)
```

`gh api repos/jaylfc/taosmd/pulls/357/files` returns exactly those two files, and
the PR object reports `additions=13, deletions=3, changed_files=2`. The 24-file
manifest in the body is the mentions-feed work from #349. Note that #349 is
itself wrong the same way: its body carries the identical 24-file manifest while
its real diff is 7 files / 994+2. So the stale manifest was pasted into more
than one PR.

## Where the generator lives (and why this is a hand-off)

Nothing in `jaylfc/taosmd` generates the PR body. The body-template markers do
not appear anywhere in this checkout. A checkout-wide grep returns no hits:

```
grep -rn "Autonomous build of board card" --include=*.py --include=*.md --include=*.sh .
```

The template markers are:

- the preamble `CARD TITLE (intent, not commit subject):`
- the line `Autonomous build of board card <tsk>.`
- the automated banner `REVIEW WARNING (automated):`
- the `Files:` block in `git diff --stat` form (the ` |  N ++` columns and the
  trailing `N files changed, X insertions(+), Y deletions(-)` summary)
- the `Suite: N passed, M skipped` acceptance line

The only references to the components that produce exec PRs are in `STATUS.md`,
which names `executor.sh`, `dispatch_loop.sh`, `review_loop.sh`,
`next_card.py`, and `throttle_check.py` as the taOS fleet dispatcher/executor.
None of those files exist in this repo. Per STATUS.md, `executor.sh` builds the
lane prompt from the card body alone, so the body template and the `Files:` /
`Suite:` derivation both originate in the taOS executor/lane system, not in
`jaylfc/taosmd`.

Acceptance #5: "If the generator turns out to be outside this repo, say so and
report where, and this card becomes a hand-off rather than a build." It is
outside. Reported: the taOS executor/lane system referenced by `executor.sh` and
its lane prompt. The orchestrator host is the taOS fleet (jaylfc/taOS per the
taOS convention); the executor owner should confirm the exact script and path.

## The fix to hand off

Derive the `Files:` block from the real diff at PR-open time, re-deriving the
merge-base rather than assuming the branch point. An assumed or stale branch
point is the likely mechanism here: it sweeps an already-merged sibling PR
(mentions-feed, #349) into the diff and surfaces as a pasted manifest:

```
git diff --stat "$(git merge-base HEAD origin/master)"..HEAD
```

Use that output verbatim for the `Files:` block. Where a card demands a proof
(a suite count, a byte size, a sha256), either carry the value measured from the
actual `pytest`/probe run, or state explicitly that it was not run. Never emit a
cached or intended file list that was not verified against the branch tip, and
never omit a demanded proof silently. The acceptance line that is currently
optional should become mandatory: if the run did not happen, say so, so silence
cannot read as done.

## Demonstration (no PR opened; the sandbox forbids touching remotes)

`gh` is authenticated here as `jaylfc`, but opening or updating a PR pushes to a
remote, which the session rules forbid. The real-diff method is verified
against an existing PR instead, on `exec/tsk-r44fqf` (the branch backing #357),
using only local refs and a read-only API call:

```
$ git merge-base origin/exec/tsk-r44fqf origin/master
e75933613f748099ceac4af5fe481c885bf90b0e

$ git diff --stat e7593361..origin/exec/tsk-r44fqf
benchmarks/data/README.md                     | 14 +++++++++---
changelog.d/tsk-r44fqf-unverify-s-full-pin.md |  2 ++
2 files changed, 13 insertions(+), 3 deletions(-)

$ gh api repos/jaylfc/taosmd/pulls/357/files --jq '.[].filename'
benchmarks/data/README.md
changelog.d/tsk-r44fqf-unverify-s-full-pin.md
```

The two sets match exactly. The current body matches neither. If the executor
generates the body with `git diff --stat "$(git merge-base HEAD origin/master)"..HEAD`
at open time, PR #357 would list these two files at 13+/3-, identical to
`gh api repos/jaylfc/taosmd/pulls/357/files`.

For reference, the working branch for this card, `exec/tsk-xuumog`, is a clean
descendant of `origin/master` (zero divergent commits, verified with
`git rev-list --count origin/master..exec/tsk-xuumog` == 0), so a PR opened from
it would expose only this hand-off doc and its changelog fragment in
`pulls/N/files`.

## Also possible from this repo (not built here, to keep the change a pure hand-off)

A `pull_request` workflow in `jaylfc/taosmd` could read `pr.N.files` and `pr.N`
from the GitHub API and fail the check when the body's `Files:` block does not
match the real file list, or when a demanded `Suite:` proof is absent. That
would enforce the contract from the taosmd side regardless of the generator, but
it is a separate change and is left as a suggestion here.
