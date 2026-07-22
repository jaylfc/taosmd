# Agent jobs: rules and repo facts

Self-contained jobs for coding agents working on this repo while the primary
session is unavailable. Each job file in this directory is complete on its
own: branch name, commit message, PR title, the exact files you may touch,
step by step instructions, and verification commands. Do the steps in order.
Do not improvise.

## Absolute rules (every job, no exceptions)

1. Git identity: user.name `jaylfc`, user.email `jaylfc25@gmail.com`. Commit
   with `git -c user.name=jaylfc -c user.email=jaylfc25@gmail.com commit ...`.
2. NEVER add Co-Authored-By lines, "Generated with" lines, or any mention of
   AI assistance to commits, PR descriptions, or code comments.
3. Never use em dashes anywhere (commit messages, comments, docs). Use
   commas, colons, or periods.
4. One branch per job, named exactly as the job file says, created from
   `origin/master`. PR to `master`. NEVER merge the PR yourself: the primary
   session reviews and merges.
5. Touch ONLY the files the job lists under "Allowed files". If completing
   the job seems to require touching anything else, STOP and leave a comment
   on the PR instead of guessing.
6. No new dependencies. Never edit pyproject.toml, requirements, or CI.
7. The full test suite must pass before you push:
   `python3 -m pytest -q -m "not slow"` (well over a thousand tests, all
   green; record the exact number in your PR body). Lint the files you
   changed: `python3 -m ruff check <files>`.
8. Never commit IP addresses, hostnames, tokens, or credentials.
9. STOP on any surprise: a failing test you did not cause, a merge conflict,
   a file that does not look like the job describes. Open the PR with what
   you have, describe the surprise in the PR body, and stop.

## Repo facts you will not know (your training data is older than this repo)

- The product name in prose is `taOSmd`. Lowercase `taosmd` is only for the
  package, CLI, and repo slug. Do not "correct" either form.
- Main branch is `master`. There is no `dev` branch in this repo.
- Python 3.11+, stdlib-first: the HTTP server (taosmd/http_server.py) is
  deliberately stdlib-only, no Flask or FastAPI. Do not introduce frameworks.
- Tests are plain pytest functions in tests/, no test classes. Mirror the
  style of the file you are editing.
- Type hints use the modern `X | None` form, not `Optional[X]`.
- `ruff` is the linter. Some findings in files you did not touch are known
  and pre-existing: never "fix" code outside your allowed files.
- docs/research-report.md and docs/benchmarks.md carry benchmark numbers with
  strict provenance rules. NEVER change a number in either file.
- The A2A bus, registry auth, and grants verifier are security surfaces.
  No job in this pack touches them; if your job seems to lead there, STOP.

## Job index

Every job carries a Status line at the top of its file. Check it before you
start. Only take a job whose status is OPEN. The pack was last verified
against master on 2026-07-21.

| Job | File | Status | Risk | Scope |
|---|---|---|---|---|
| JOB-001 | job-001-benchmarks-em-dash-sweep.md | OPEN | minimal | punctuation only, one doc |
| JOB-002 | job-002-cross-encoder-path-fix.md | ON HOLD, see #199 | n/a | do not start |
| JOB-003 | job-003-http-server-dead-import.md | OPEN | minimal | delete dead code, one function |
| JOB-004 | job-004-eventqa-runner-exit-code.md | OPEN | low | two returns in one runner, one test file |
| JOB-005 | job-005-collections-db-connect.md | OPEN | low | one line plus an import, one test file |

Take them in any order; no job depends on another. If two of you are working
at once, do not both take the same job, and never touch a file another job
lists under "Allowed files".

## Work that is deliberately NOT in this pack

Some open issues look like small jobs and are not. Do not pick these up even
if you find the issue and it reads as mechanical:

- **#199 (reranker model path).** Superseded JOB-002. The read path and the
  download path have separate defaults, so fixing either one alone desyncs
  them, and the right resolution mechanism is a design decision.
- **#203 (unregistered databases in the migration REGISTRY).** Blocked on
  PR #201, which is not merged. The issue calls each entry mechanical, but a
  registry entry needs a `detect` probe written against each database's real
  baseline schema, and a wrong probe stamps the wrong version against a live
  store. That is the exact silent-corruption class the framework exists to
  prevent. The `knowledge-graph.db` two-schema-owners half of the issue is
  design work outright.
- **#206 (CLI silently operating on a remote store).** The issue proposes two
  fixes and one of them is a breaking change. Picking between them is the
  work.
