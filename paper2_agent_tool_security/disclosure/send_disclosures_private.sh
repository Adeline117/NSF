#!/usr/bin/env bash
# send_disclosures_private.sh — Send vulnerability disclosures via GitHub Security Advisories
#
# Uses the GitHub Security Advisory API for repositories that support private
# vulnerability reporting. This is the preferred method for responsible disclosure
# as it keeps the report private until the maintainer is ready.
#
# Usage:
#   ./send_disclosures_private.sh              # Live mode: creates advisories
#   ./send_disclosures_private.sh --dry-run    # Dry-run: prints API calls without executing
#
# Prerequisites:
#   - gh CLI authenticated with repo + security_advisory scopes
#   - Report files in disclosure/reports/*.md
#   - python3 available (for JSON construction)
#
# Rate limiting: 5-second delay between each advisory creation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${SCRIPT_DIR}/reports"
LOG_FILE="${SCRIPT_DIR}/send_disclosures_private.log"
BATCH_JSON="${SCRIPT_DIR}/disclosure_batch_results.json"
TODAY=$(date +%Y-%m-%d)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — no advisories will be created ==="
    echo ""
fi

# Build a lookup file: slug|repo|findings|severity
LOOKUP_FILE=$(mktemp)
python3 -c "
import json
with open('${BATCH_JSON}') as f:
    data = json.load(f)
for r in data['reports']:
    sev = r['risk_rating']
    if sev == 'critical':
        ghsa_sev = 'critical'
    elif sev == 'high':
        ghsa_sev = 'high'
    else:
        ghsa_sev = 'medium'
    print(f\"{r['slug']}|{r['repo']}|{r['total_findings']}|{ghsa_sev}\")
" > "$LOOKUP_FILE"

lookup_field() {
    local slug="$1"
    local field_num="$2"
    grep "^${slug}|" "$LOOKUP_FILE" | head -1 | cut -d'|' -f"$field_num"
}

# Counters
total=0
success=0
failed=0
skipped=0
unsupported=0

# Initialize log
cat > "$LOG_FILE" <<LOGHEADER
=== Private Security Advisory Batch Log ===
Date: ${TODAY}
Mode: $(if $DRY_RUN; then echo 'DRY RUN'; else echo 'LIVE'; fi)
============================================

LOGHEADER

REPO_COUNT=$(wc -l < "$LOOKUP_FILE" | tr -d ' ')
echo "Creating private security advisories for ${REPO_COUNT} repositories..."
echo "Log file: ${LOG_FILE}"
echo ""

for report_file in "${REPORTS_DIR}"/*.md; do
    filename=$(basename "$report_file" .md)
    total=$((total + 1))

    owner_repo=$(lookup_field "$filename" 2)
    if [[ -z "$owner_repo" ]]; then
        echo "[SKIP] #${total} ${filename} — not found in batch results"
        echo "[SKIP] ${filename} — slug not in disclosure_batch_results.json" >> "$LOG_FILE"
        skipped=$((skipped + 1))
        continue
    fi

    findings=$(lookup_field "$filename" 3)
    severity=$(lookup_field "$filename" 4)

    # Build JSON payload using python3 (handles escaping correctly)
    PAYLOAD_FILE=$(mktemp)
    python3 -c "
import json, sys

with open('${report_file}') as f:
    body = f.read()

# Truncate to GitHub API limit (65535 chars)
if len(body) > 65000:
    body = body[:65000] + '\n\n... [truncated — see full report]'

advisory = {
    'summary': 'Tool Interface Vulnerability Report — TCPI Research (${findings} findings)',
    'description': body,
    'severity': '${severity}',
    'vulnerabilities': [
        {
            'package': {
                'ecosystem': 'other',
                'name': '${owner_repo}'
            },
            'vulnerable_version_range': '<= *',
            'patched_versions': None
        }
    ],
    'cwe_ids': ['CWE-284', 'CWE-20'],
    'credits': [
        {
            'login': 'adelinewen',
            'type': 'reporter'
        }
    ]
}

with open('${PAYLOAD_FILE}', 'w') as f:
    json.dump(advisory, f)
" 2>/dev/null

    if $DRY_RUN; then
        echo "[DRY-RUN] #${total} ${owner_repo} (severity: ${severity}, findings: ${findings})"
        echo "  Step 1: Check if private vulnerability reporting is enabled"
        echo "    gh api repos/${owner_repo} --jq '.security_and_analysis.secret_scanning.status'"
        echo "  Step 2: Create security advisory"
        echo "    gh api repos/${owner_repo}/security-advisories --method POST --input <payload.json>"
        echo "  Payload summary: ${findings} findings, severity=${severity}"
        echo ""
        echo "[DRY-RUN] ${owner_repo} — severity=${severity} findings=${findings}" >> "$LOG_FILE"
        success=$((success + 1))
    else
        echo -n "[SEND] #${total} ${owner_repo} (${severity}) ... "

        RESULT=""
        if RESULT=$(gh api "repos/${owner_repo}/security-advisories" \
            --method POST \
            --input "$PAYLOAD_FILE" 2>&1); then
            ADVISORY_URL=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('html_url','(created)'))" 2>/dev/null || echo "(created)")
            echo "OK -> ${ADVISORY_URL}"
            echo "[OK] ${owner_repo} — ${ADVISORY_URL}" >> "$LOG_FILE"
            success=$((success + 1))
        else
            if echo "$RESULT" | grep -qi "not enabled\|not supported\|private vulnerability reporting\|403\|Repository does not have"; then
                echo "UNSUPPORTED (private advisories not enabled)"
                echo "[UNSUPPORTED] ${owner_repo} — private vulnerability reporting not enabled" >> "$LOG_FILE"
                unsupported=$((unsupported + 1))
            else
                echo "FAILED: ${RESULT}"
                echo "[FAIL] ${owner_repo} — ${RESULT}" >> "$LOG_FILE"
                failed=$((failed + 1))
            fi
        fi

        # Rate limiting
        if [[ $total -lt $REPO_COUNT ]]; then
            echo "  (waiting 5s for rate limiting...)"
            sleep 5
        fi
    fi

    rm -f "$PAYLOAD_FILE"
done

# Clean up
rm -f "$LOOKUP_FILE"

echo ""
echo "=== Summary ==="
echo "Total:       ${total}"
echo "Success:     ${success}"
echo "Unsupported: ${unsupported}"
echo "Failed:      ${failed}"
echo "Skipped:     ${skipped}"
echo ""
echo "Summary: total=${total} success=${success} unsupported=${unsupported} failed=${failed} skipped=${skipped}" >> "$LOG_FILE"

if [[ $unsupported -gt 0 ]]; then
    echo ""
    echo "NOTE: ${unsupported} repos do not support private security advisories."
    echo "For those repos, use send_disclosures.sh (public issue) or contact maintainers directly."
fi
