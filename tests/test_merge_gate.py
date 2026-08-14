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


def test_inline_only_review_rejected_by_buggy_logic(tmp_path):
    fixture = _read("pr_inline_only.json")
    _make_fake_gh(str(tmp_path), pr=fixture)
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")

    # Reconstruct the pre-fix logic that relied on includesCreatedEdit
    with open(os.path.join("scripts", "merge-gate", "check_bot_anchoring.sh")) as f:
        buggy_script = f.read().replace(
            "has_inline = (r.get('comments') or {}).get('totalCount', 0) > 0",
            "has_inline = r.get('includesCreatedEdit', False)",
        )
    buggy_path = os.path.join(str(tmp_path), "buggy.sh")
    with open(buggy_path, "w") as f:
        f.write(buggy_script)

    result = subprocess.run(
        ["bash", buggy_path, "218"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 10, result.stdout + result.stderr
    assert result.stdout.startswith("FAILED:")


def test_inline_only_review_accepted_by_fix(tmp_path):
    result = _run(
        "check_bot_anchoring.sh",
        ["218"],
        str(tmp_path),
        pr=_read("pr_inline_only.json"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("SUCCESS:")


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


def test_red_first_usage_error():
    result = subprocess.run(
        ["bash", "scripts/merge-gate/red_first.sh"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stdout.startswith("FAILED:")
