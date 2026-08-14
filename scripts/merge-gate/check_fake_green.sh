#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "FAILED: usage: check_fake_green.sh <pr-number>"
    exit 2
fi

PR_NUMBER="$1"

PR_JSON=$(gh pr view "$PR_NUMBER" --json number,headRefOid,reviews,comments 2>/dev/null) || {
    echo "FAILED: gh/network error"
    exit 3
}

REPO_JSON=$(gh repo view --json owner,name 2>/dev/null) || {
    echo "FAILED: gh/network error"
    exit 3
}

HEAD_OID=$(echo "$PR_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['headRefOid'])") || {
    echo "FAILED: could not parse PR data"
    exit 3
}

OWNER=$(echo "$REPO_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['owner']['login'])")
REPO_NAME=$(echo "$REPO_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['name'])")

STATUS_JSON=$(gh api "repos/$OWNER/$REPO_NAME/commits/$HEAD_OID/status" 2>/dev/null) || {
    echo "FAILED: gh/network error"
    exit 3
}

CHECKRUNS_JSON=$(gh api "repos/$OWNER/$REPO_NAME/commits/$HEAD_OID/check-runs" 2>/dev/null) || {
    echo "FAILED: gh/network error"
    exit 3
}

python3 - "$PR_JSON" "$STATUS_JSON" "$CHECKRUNS_JSON" << 'PYEOF'
import json, sys

try:
    pr_data = json.loads(sys.argv[1])
    status_data = json.loads(sys.argv[2])
    checkruns_data = json.loads(sys.argv[3])
except json.JSONDecodeError:
    print('FAILED: could not parse gh response')
    sys.exit(3)

reviews = pr_data.get('reviews', [])
comments = pr_data.get('comments', [])

# Check (a): success status/check with "Review rate limited"
for item in status_data.get('statuses', []):
    if item.get('context') != 'CodeRabbit':
        continue
    if (item.get('state') or '').upper() != 'SUCCESS':
        continue
    desc = item.get('description') or ''
    if 'Review rate limited' in desc:
        print('FAILED: CodeRabbit commit status is SUCCESS but description contains "Review rate limited"')
        print('Remediation: the plain "@coderabbitai review" command no-ops on a PR that was reviewed and then pushed to; only "@coderabbitai full review" forces a real pass.')
        sys.exit(11)

for item in checkruns_data.get('check_runs', []):
    if 'CodeRabbit' not in item.get('name', ''):
        continue
    conclusion = (item.get('conclusion') or '').upper()
    if conclusion != 'SUCCESS':
        continue
    output = item.get('output') or {}
    output_text = output.get('text') or output.get('title') or output.get('summary') or ''
    if 'Review rate limited' in output_text:
        print('FAILED: CodeRabbit check run is SUCCESS but output contains "Review rate limited"')
        print('Remediation: the plain "@coderabbitai review" command no-ops on a PR that was reviewed and then pushed to; only "@coderabbitai full review" forces a real pass.')
        sys.exit(11)

# Check (b): bare "Review finished" comment with no review object
cr_reviews = [
    r for r in reviews
    if (r.get('author') or {}).get('login') == 'coderabbitai'
    and r.get('state') in ('COMMENTED', 'APPROVED', 'CHANGES_REQUESTED')
]

cr_comments = [
    c for c in comments
    if (c.get('author') or {}).get('login') == 'coderabbitai'
]

BARE_ACK_PATTERNS = ['Review finished', 'review finished']

def is_bare_ack(body):
    body_lower = body.lower()
    for pat in BARE_ACK_PATTERNS:
        if pat.lower() in body_lower:
            return True
    return False

bare_acks = [c for c in cr_comments if is_bare_ack(c.get('body') or '')]

if bare_acks and not cr_reviews:
    print('FAILED: only CodeRabbit artifact is a bare acknowledgement comment with no review object attached')
    print('Remediation: the plain "@coderabbitai review" command no-ops on a PR that was reviewed and then pushed to; only "@coderabbitai full review" forces a real pass.')
    sys.exit(11)

print('SUCCESS: no fake-green signals detected')
sys.exit(0)
PYEOF
exit $?