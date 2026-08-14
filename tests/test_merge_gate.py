import json
import os
import subprocess
import stat

import pytest


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "merge_gate")


def _read(name):
    with open(os.path.join(FIXTURES_DIR, name), "r") as f:
        return f.read()


def _make_fake_gh(tmpdir, **fixtures):
    fake = os.path.join(tmpdir, "gh")
    with open(fake, "w") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("import sys\n\n")
        for key, value in fixtures.items():
            f.write(f"{key.upper()} = {repr(value)}\n")
        f.write("""
def main():
    a = sys.argv[1:]
    if not a:
        print("{}")
        return
    if a[0] == 'pr' and len(a) > 1 and a[1] == 'view':
        print(PR)
        return
    if a[0] == 'repo' and len(a) > 1 and a[1] == 'view':
        print(REPO)
        return
    if a[0] == 'api' and len(a) > 1:
        p = a[1]
        if 'status' in p:
            print(STATUS)
            return
        if 'check-runs' in p:
            print(CHECKRUNS)
            return
    print("{}")

if __name__ == '__main__':
    main()
""")
    os.chmod(fake, stat.S_IRWXU)
    return fake


def _run(script, args, tmpdir, **fixtures):
    fake_gh = _make_fake_gh(tmpdir, **fixtures)
    env = os.environ.copy()
    env["PATH"] = tmpdir + ":" + env.get("PATH", "")
    script_path = os.path.abspath(os.path.join("scripts", "merge-gate", script))
    result = subprocess.run(
        ["bash", script_path] + args,
        capture_output=True,
        text=True,
        env=env,
    )
    return result


def test_anchored_head_passes(tmp_path):
    result = _run(
        "check_bot_anchoring.sh",
        ["218"],
        str(tmp_path),
        pr=_read("pr_anchored_head.json"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("SUCCESS:")


def test_anchored_old_sha_fails(tmp_path):
    result = _run(
        "check_bot_anchoring.sh",
        ["218"],
        str(tmp_path),
        pr=_read("pr_anchored_old_sha.json"),
    )
    assert result.returncode == 10, result.stdout + result.stderr
    assert result.stdout.startswith("FAILED:")


def test_human_only_fails(tmp_path):
    result = _run(
        "check_bot_anchoring.sh",
        ["218"],
        str(tmp_path),
        pr=_read("pr_human_only.json"),
    )
    assert result.returncode == 10, result.stdout + result.stderr
    assert result.stdout.startswith("FAILED:")


def test_rate_limited_fails(tmp_path):
    result = _run(
        "check_fake_green.sh",
        ["218"],
        str(tmp_path),
        pr=_read("pr_rate_limited.json"),
        repo=_read("repo_info.json"),
        status=_read("status_rate_limited.json"),
        checkruns=_read("checkruns_empty.json"),
    )
    assert result.returncode == 11, result.stdout + result.stderr
    assert result.stdout.startswith("FAILED:")
    assert "Review rate limited" in result.stdout


def test_bare_ack_fails(tmp_path):
    result = _run(
        "check_fake_green.sh",
        ["218"],
        str(tmp_path),
        pr=_read("pr_bare_ack.json"),
        repo=_read("repo_info.json"),
        status="{}",
        checkruns="{}",
    )
    assert result.returncode == 11, result.stdout + result.stderr
    assert result.stdout.startswith("FAILED:")


def _write_old_anchoring_script(tmpdir):
    """Write a copy of check_bot_anchoring.sh using the OLD includesCreatedEdit logic,
    so we can prove the bug existed before the fix."""
    script = os.path.join(tmpdir, "old_check_bot_anchoring.sh")
    with open(script, "w") as f:
        f.write("""#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "FAILED: usage: check_bot_anchoring.sh <pr-number>"
    exit 2
fi

PR_NUMBER="$1"

PR_JSON=$(gh pr view "$PR_NUMBER" --json number,headRefOid,reviews 2>/dev/null) || {
    echo "FAILED: gh/network error"
    exit 3
}

python3 - "$PR_JSON" << 'PYEOF'
import json, sys

try:
    data = json.loads(sys.argv[1])
except json.JSONDecodeError:
    print('FAILED: could not parse PR data')
    sys.exit(3)

head_oid = data.get('headRefOid', '')
reviews = data.get('reviews', [])
bot_authors = {'coderabbitai', 'qodo-code-review', 'kilo-code-bot'}

for r in reviews:
    author = (r.get('author') or {}).get('login', '')
    if author not in bot_authors:
        continue
    state = r.get('state', '')
    if state not in ('COMMENTED', 'APPROVED', 'CHANGES_REQUESTED'):
        continue
    commit_oid = (r.get('commit') or {}).get('oid', '')
    if commit_oid != head_oid:
        continue
    body = r.get('body') or ''
    has_inline = r.get('includesCreatedEdit', False)
    if body.strip() or has_inline:
        print('SUCCESS: anchored substantive bot review found')
        sys.exit(0)

print('FAILED: no anchored substantive bot review found')
sys.exit(10)
PYEOF
exit $?
""")
    os.chmod(script, stat.S_IRWXU)
    return script


def test_inline_only_review_fix(tmp_path):
    """An inline-only bot review (empty body, comments.totalCount > 0,
    includesCreatedEdit=false) must be rejected by the old logic and
    accepted by the fix."""
    fixture = _read("pr_inline_only.json")

    fake_gh = _make_fake_gh(str(tmp_path), pr=fixture)
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")

    # --- Fixed code accepts (exit 0) ---
    script_path = os.path.abspath(
        os.path.join("scripts", "merge-gate", "check_bot_anchoring.sh")
    )
    result = subprocess.run(
        ["bash", script_path, "218"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("SUCCESS:")

    # --- Old code rejects (exit 10) - proves the bug existed ---
    old_script = _write_old_anchoring_script(str(tmp_path))
    old_result = subprocess.run(
        ["bash", old_script, "218"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert old_result.returncode == 10, old_result.stdout + old_result.stderr
    assert old_result.stdout.startswith("FAILED:")
    assert "no anchored substantive bot review found" in old_result.stdout


def test_red_first_usage_error():
    result = subprocess.run(
        ["bash", "scripts/merge-gate/red_first.sh"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stdout.startswith("FAILED:")
