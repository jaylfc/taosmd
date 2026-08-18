"""No doc may name an out-of-repo copy of scripts/resume_arm_time.py (tsk-cdqsgy).

Since #354 the script derives its own location from ``__file__``, which makes
``scripts/resume_arm_time.py`` the only copy that can be correct. An out-of-repo
copy or symlink is therefore a second document, and one drifted: for weeks
``~/.taos-team/resume_arm_time.py`` resolved to a checkout fifteen commits behind
master, so the armed resume pair ran a pre-#354 script while every path involved
still resolved successfully. Resolving a path proves a file is REACHABLE, never
that its contents are CURRENT, so the durable fix has to be that no doc names a
path that can drift in the first place.

The scan is deliberately checkable on a machine that has no ``/home/jay`` at all:
it reads files out of the repo and never touches the filesystem it describes.

Four of these tests exist to stop the fifth from being vacuous. A detector that
flagged everything would satisfy the positive fixture alone, and a tree scan that
reached no files would report zero violations and read as a pass, so the
engagement of both the detector and the scan is asserted rather than assumed.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Doc trees whose invocations are meant to be run from the repo working directory.
DOC_ROOTS = (".claude", "docs")

# The filename, with whatever path-shaped run of characters immediately precedes
# it. The lookbehind keeps `test_resume_arm_time.py` from matching on its own
# suffix: there the preceding character is `_`, which is a path character, so the
# filename is part of a longer name rather than a reference to the script.
_HELPER_REF = re.compile(r"(?<![\w.\-])(?P<prefix>[\w.~\-/]*/)?resume_arm_time\.py")

# A bare mention (no prefix) names the script without claiming a location, which
# is how the doc refers back to an invocation it already spelled out.
ALLOWED_PREFIXES = {None, "scripts/", "./scripts/"}


def scan_text(text):
    """Return the offending path prefixes named in ``text``."""
    return [
        m.group("prefix")
        for m in _HELPER_REF.finditer(text)
        if m.group("prefix") not in ALLOWED_PREFIXES
    ]


def iter_doc_files(root):
    """Every file under the doc trees of ``root``, sorted for stable reporting."""
    files = []
    for name in DOC_ROOTS:
        base = root / name
        if base.is_dir():
            files.extend(p for p in base.rglob("*") if p.is_file())
    return sorted(files)


def scan_tree(root):
    """Return ``(path, prefix)`` for every out-of-repo reference under ``root``."""
    violations = []
    for path in iter_doc_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend((path, prefix) for prefix in scan_text(text))
    return violations


def test_detector_flags_an_out_of_repo_path():
    """Positive control: the exact form the audit prompt used to carry."""
    text = "arm it with `python3 ~/.taos-team/resume_arm_time.py <resets_at>` first"
    assert scan_text(text) == ["~/.taos-team/"]


def test_detector_flags_an_absolute_path():
    text = "run /home/jay/.taos-fleet-tools/resume_arm_time.py"
    assert scan_text(text) == ["/home/jay/.taos-fleet-tools/"]


def test_detector_accepts_the_in_repo_path_and_bare_mentions():
    """Negative control: without this, a detector that flagged everything passes."""
    assert scan_text("`python3 scripts/resume_arm_time.py <resets_at>`") == []
    assert scan_text("the RETRY CRON line printed by that resume_arm_time.py run") == []


def test_detector_ignores_the_test_module_name():
    assert scan_text("tests/test_resume_arm_time.py covers the guard") == []


def test_scan_flags_a_fixture_file(tmp_path):
    """The directory scan itself fires, not merely the regex it calls."""
    doc = tmp_path / ".claude" / "audit-cron-prompt.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("primary: `python3 ~/.taos-team/resume_arm_time.py <resets_at>`\n")

    violations = scan_tree(tmp_path)

    assert [(p.name, prefix) for p, prefix in violations] == [
        ("audit-cron-prompt.md", "~/.taos-team/")
    ]


def test_the_scan_reaches_the_doc_that_regressed():
    """Engagement: a scan that read nothing would report zero violations."""
    scanned = iter_doc_files(REPO)
    assert scanned, f"no doc files found under {DOC_ROOTS}, so the tree scan is vacuous"

    prompt = REPO / ".claude" / "audit-cron-prompt.md"
    assert prompt in scanned, f"{prompt} is not being scanned"
    assert _HELPER_REF.search(prompt.read_text(encoding="utf-8")), (
        f"{prompt} no longer mentions the helper at all. If it was renamed or the "
        "invocation moved, point this test at its new home rather than deleting it."
    )


def test_no_doc_names_an_out_of_repo_helper_path():
    violations = scan_tree(REPO)
    assert not violations, "docs naming an out-of-repo helper copy: " + ", ".join(
        f"{path.relative_to(REPO)} -> {prefix}resume_arm_time.py"
        for path, prefix in violations
    )
