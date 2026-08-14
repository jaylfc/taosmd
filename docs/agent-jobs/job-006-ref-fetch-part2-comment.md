# JOB-006: Repair the mangled Part 2 comment block in tests/test_ref_fetch.py

**Status: OPEN (verified 2026-08-14).** Comments only. No code changes, no
behaviour changes, no new tests. If you find yourself editing an executable
line, you have gone wrong: stop and re-read this file.

Read docs/agent-jobs/README.md first and follow its absolute rules.

- Branch: `docs/ref-fetch-part2-comment` (from `origin/master`)
- Commit message: `docs(tests): repair the mangled Part 2 comment block in test_ref_fetch.py`
- PR title: `Repair the mangled Part 2 comment block in tests/test_ref_fetch.py`
- Allowed files: `tests/test_ref_fetch.py` and nothing else.

## What is wrong

PR #273 added an explanatory comment above "Part 2" of
`test_redirect_no_authorization_leak`. It landed malformed. Three separate
problems, all in one block:

1. **A comment at column 0 inside a method body.** Line 260 starts at the
   left margin while every line around it is indented 8 spaces. Python allows
   this (comments are not statements) so nothing fails, but it reads as if
   the method ended there.
2. **Inconsistent indentation below it.** The lines that follow sit at 8
   spaces, then the block runs straight into the pre-existing
   `# ---- Part 2: control without _NoRedirect ----` banner, and the line
   after that banner is indented 12 spaces where its neighbours are at 8.
3. **It pastes review prose into the file, including a sentence that is now
   false.** The text ends with "Right now it reads like the control, which is
   misleading." That was a review comment written *before* the payload fix
   landed. The fix landed in the same PR, so the sentence describes a state
   that no longer exists.

Nothing about this affects behaviour. `python3 -m ast` parses the file fine
and the test passes. It is untidy, in a file that a future reader will come
to precisely when they are trying to understand a security control.

## Step by step

### 1. Branch

```
git fetch origin
git checkout -b docs/ref-fetch-part2-comment origin/master
```

### 2. Find the block

```
grep -n "Part 2" tests/test_ref_fetch.py
```

You will get two hits. The first (around line 260) is the malformed block.
The second (around line 267) is the pre-existing
`# ---- Part 2: control without _NoRedirect ----` banner, which is correctly
indented and which you are keeping.

### 3. Replace the malformed block

Everything from the column-0 `# Part 2: stdlib demo ...` line down to and
including the `# which is misleading.` line is what you are replacing. Do
**not** touch the `# ---- Part 2: control without _NoRedirect ----` banner
below it, and do not touch the `# Build an opener without _NoRedirect ...`
line under that banner except as described in step 4.

Replace the malformed block with exactly this, indented 8 spaces to match its
neighbours:

```python
        # What Part 2 is, and what it is not. It hand-builds an opener with
        # urllib.request.build_opener() and a Request carrying a hardcoded
        # Authorization header, to demonstrate two things: that a default
        # opener follows redirects and forwards headers across origins, and
        # that origin B is capable of recording a header at all. It is a
        # demonstration of the stdlib default, not the control for our fetch
        # path. Part 1 above is that control: it exercises the real
        # fetch_by_ref path and asserts the header does not reach origin B.
```

Keep the wording. Do not add an em dash (see README rule 3). Do not carry
over the "Right now it reads like the control, which is misleading" sentence
in any form: it is the false one.

### 4. Fix the one over-indented line

Directly under the `# ---- Part 2: control without _NoRedirect ----` banner
there are two comment lines. The first is at 8 spaces and correct. The
second reads

```
            # follows redirects). Origin B MUST receive the Authorization header.
```

and sits at 12 spaces. Re-indent that one line to 8 spaces so it lines up
with the line above it. That is the only whitespace change outside the block
from step 3.

## Verification (run all four, paste the output in your PR body)

1. Nothing but comments changed. This must print nothing at all:

```
git diff -U0 origin/master -- tests/test_ref_fetch.py | grep -E "^[+-]" | grep -v "^[+-][+-]" | grep -vE "^[+-]\s*#" | grep -vE "^[+-]\s*$"
```

If it prints any line, you have edited code. Undo it.

2. No column-0 comments inside the class body. This must print nothing:

```
awk 'NR>200 && /^#/' tests/test_ref_fetch.py
```

3. The file still parses and the test still passes:

```
python3 -m ast tests/test_ref_fetch.py > /dev/null && echo "parses"
python3 -m pytest tests/test_ref_fetch.py -q
```

Record the exact pass count in the PR body.

4. The false sentence is gone. This must print nothing:

```
grep -n "reads like the control" tests/test_ref_fetch.py
```

## STOP conditions

- `tests/test_ref_fetch.py` does not look like the description above, or the
  malformed block is already gone: someone else fixed it. Open no PR, and say
  so on the card.
- Any test in `tests/test_ref_fetch.py` fails before you have changed
  anything.
- Verification command 1 prints a line and you cannot see why.

In all three cases, stop and describe what you saw. Do not improvise a fix.
