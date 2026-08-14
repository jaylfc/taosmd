#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "FAILED: usage: red_first.sh <pr-number> --paths <src paths> --tests <pytest node ids>"
    exit 2
fi

PR_NUMBER="$1"
shift

SRC_PATHS=()
TEST_IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --paths)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SRC_PATHS+=("$1")
                shift
            done
            ;;
        --tests)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TEST_IDS+=("$1")
                shift
            done
            ;;
        *)
            echo "FAILED: unknown argument: $1"
            exit 2
            ;;
    esac
done

if [[ ${#SRC_PATHS[@]} -eq 0 || ${#TEST_IDS[@]} -eq 0 ]]; then
    echo "FAILED: both --paths and --tests are required"
    exit 2
fi

PR_JSON=$(gh pr view "$PR_NUMBER" --json headRefOid,headRefName,baseRefName 2>/dev/null) || {
    echo "FAILED: gh/network error"
    exit 3
}

python3 - "$PR_JSON" -- "${SRC_PATHS[@]}" -- "${TEST_IDS[@]}" << 'PYEOF'
import json, sys, subprocess, os, tempfile

args = sys.argv[1:]
sep1 = args.index('--')
sep2 = args.index('--', sep1 + 1)

try:
    data = json.loads(args[0])
except json.JSONDecodeError:
    print('FAILED: could not parse PR data')
    sys.exit(3)

src_paths = args[sep1 + 1:sep2]
test_ids = args[sep2 + 1:]

head_ref = data.get('headRefName', '')
base_ref = data.get('baseRefName', '')

def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)

def eprint(*a, **kw):
    print(*a, file=sys.stderr, **kw)

repo_root = run(['git', 'rev-parse', '--show-toplevel'], check=True).stdout.strip()
os.chdir(repo_root)

tmpdir = tempfile.mkdtemp(prefix='red-first-')
try:
    result = run(['git', 'worktree', 'add', '-f', tmpdir, head_ref])
    if result.returncode != 0:
        eprint('FAILED: could not check out PR branch into worktree')
        eprint(result.stderr, end='')
        print('FAILED: could not check out PR branch into worktree')
        sys.exit(3)
except Exception:
    eprint('FAILED: could not check out PR branch into worktree')
    print('FAILED: could not check out PR branch into worktree')
    sys.exit(3)

os.chdir(tmpdir)

# Stage 1: tests must PASS
eprint('Stage 1: running tests on PR branch...')
result = run(['uv', 'run', 'pytest'] + test_ids + ['-q'])
if result.returncode != 0:
    eprint(result.stdout, end='')
    eprint(result.stderr, end='')
    print('FAILED: tests did not pass on PR branch')
    os.chdir(repo_root)
    run(['git', 'worktree', 'remove', tmpdir], capture_output=True)
    sys.exit(13)

# Get merge base
try:
    merge_base = run(['git', 'merge-base', head_ref, 'origin/' + base_ref], check=True).stdout.strip()
except subprocess.CalledProcessError:
    try:
        merge_base = run(['git', 'merge-base', head_ref, base_ref], check=True).stdout.strip()
    except subprocess.CalledProcessError:
        eprint('FAILED: could not find merge base')
        print('FAILED: could not find merge base')
        os.chdir(repo_root)
        run(['git', 'worktree', 'remove', tmpdir], capture_output=True)
        sys.exit(3)

# Stage 2: revert source paths to merge-base version
eprint('Stage 2: reverting source paths to merge-base...')
result = run(['git', 'checkout', merge_base, '--'] + src_paths)
if result.returncode != 0:
    eprint('FAILED: could not revert source paths')
    eprint(result.stderr, end='')
    print('FAILED: could not revert source paths')
    os.chdir(repo_root)
    run(['git', 'worktree', 'remove', tmpdir], capture_output=True)
    sys.exit(3)

# Stage 2 re-run: tests must FAIL
eprint('Stage 2: running tests after revert...')
result = run(['uv', 'run', 'pytest'] + test_ids + ['-q'])
if result.returncode == 0:
    eprint(result.stdout, end='')
    eprint(result.stderr, end='')
    print('FAILED: tests passed without the fix in place')
    os.chdir(repo_root)
    run(['git', 'worktree', 'remove', tmpdir], capture_output=True)
    sys.exit(12)

# Stage 3: restore
eprint('Stage 3: restoring source paths...')
result = run(['git', 'checkout', 'HEAD', '--'] + src_paths)
if result.returncode != 0:
    eprint('FAILED: could not restore source paths')
    eprint(result.stderr, end='')
    print('FAILED: could not restore source paths')
    os.chdir(repo_root)
    run(['git', 'worktree', 'remove', tmpdir], capture_output=True)
    sys.exit(3)

# Stage 3 re-run: tests must PASS
eprint('Stage 3: running tests after restore...')
result = run(['uv', 'run', 'pytest'] + test_ids + ['-q'])
if result.returncode != 0:
    eprint(result.stdout, end='')
    eprint(result.stderr, end='')
    print('FAILED: tests did not pass after restore')
    os.chdir(repo_root)
    run(['git', 'worktree', 'remove', tmpdir], capture_output=True)
    sys.exit(13)

# Cleanup
os.chdir(repo_root)
run(['git', 'worktree', 'remove', tmpdir], capture_output=True)

print('SUCCESS: red-first cycle completed')
sys.exit(0)
PYEOF
exit $?
