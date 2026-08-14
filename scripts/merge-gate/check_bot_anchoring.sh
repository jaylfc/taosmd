#!/usr/bin/env bash
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
    has_inline = (r.get('comments') or {}).get('totalCount', 0) > 0
    if body.strip() or has_inline:
        print('SUCCESS: anchored substantive bot review found')
        sys.exit(0)

print('FAILED: no anchored substantive bot review found')
sys.exit(10)
PYEOF
exit $?
