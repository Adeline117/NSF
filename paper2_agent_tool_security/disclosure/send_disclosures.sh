#!/usr/bin/env bash
# send_disclosures.sh — Batch send vulnerability disclosure reports via GitHub Issues
#
# Usage:
#   ./send_disclosures.sh              # Live mode: creates issues
#   ./send_disclosures.sh --dry-run    # Dry-run: prints commands without executing
#
# Prerequisites:
#   - gh CLI authenticated (gh auth status)
#   - Report files in disclosure/reports/*.md
#   - python3 available (for JSON parsing)
#
# Rate limiting: 5-second delay between each issue creation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${SCRIPT_DIR}/reports"
LOG_FILE="${SCRIPT_DIR}/send_disclosures.log"
NOTIFICATION_TEMPLATE="${SCRIPT_DIR}/notification_template.md"
BATCH_JSON="${SCRIPT_DIR}/disclosure_batch_results.json"
TODAY=$(date +%Y-%m-%d)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — no issues will be created ==="
    echo ""
fi

ISSUE_TITLE="[Security] Tool Interface Vulnerability Report — TCPI Research"

# Build a slug-to-repo lookup file (avoids bash 4 associative arrays for macOS compat)
LOOKUP_FILE=$(mktemp)
python3 -c "
import json
with open('${BATCH_JSON}') as f:
    data = json.load(f)
for r in data['reports']:
    print(r['slug'] + '|' + r['repo'])
" > "$LOOKUP_FILE"

lookup_repo() {
    local slug="$1"
    local result
    result=$(grep "^${slug}|" "$LOOKUP_FILE" | head -1 | cut -d'|' -f2)
    echo "$result"
}

# Counters
total=0
success=0
failed=0
skipped=0

# Initialize log
cat > "$LOG_FILE" <<LOGHEADER
=== Disclosure Batch Send Log ===
Date: ${TODAY}
Mode: $(if $DRY_RUN; then echo 'DRY RUN'; else echo 'LIVE'; fi)
=================================

LOGHEADER

REPO_COUNT=$(wc -l < "$LOOKUP_FILE" | tr -d ' ')
echo "Sending disclosure reports to ${REPO_COUNT} repositories..."
echo "Issue title: ${ISSUE_TITLE}"
echo "Log file: ${LOG_FILE}"
echo ""

for report_file in "${REPORTS_DIR}"/*.md; do
    filename=$(basename "$report_file" .md)
    total=$((total + 1))

    # Look up the owner/repo from the JSON
    owner_repo=$(lookup_repo "$filename")

    if [[ -z "$owner_repo" ]]; then
        echo "[SKIP] #${total} ${filename} — not found in batch results"
        echo "[SKIP] ${filename} — slug not in disclosure_batch_results.json" >> "$LOG_FILE"
        skipped=$((skipped + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "[DRY-RUN] #${total} ${owner_repo}"
        echo "  gh issue create \\"
        echo "    --repo \"${owner_repo}\" \\"
        echo "    --title \"${ISSUE_TITLE}\" \\"
        echo "    --body-file <(cat notification_template.md --- report) \\"
        echo "    --label \"security\""
        echo ""
        echo "[DRY-RUN] ${owner_repo} — command printed (not executed)" >> "$LOG_FILE"
        success=$((success + 1))
    else
        echo -n "[SEND] #${total} ${owner_repo} ... "

        # Build combined body: cover letter + separator + full report
        COMBINED_BODY_FILE=$(mktemp)
        cat "$NOTIFICATION_TEMPLATE" > "$COMBINED_BODY_FILE"
        echo "" >> "$COMBINED_BODY_FILE"
        echo "---" >> "$COMBINED_BODY_FILE"
        echo "" >> "$COMBINED_BODY_FILE"
        cat "$report_file" >> "$COMBINED_BODY_FILE"

        # Attempt to create the issue
        # First try with --label security; if the label doesn't exist, retry without
        ISSUE_URL=""
        if ISSUE_URL=$(gh issue create \
            --repo "${owner_repo}" \
            --title "${ISSUE_TITLE}" \
            --body-file "$COMBINED_BODY_FILE" \
            --label "security" 2>&1); then
            echo "OK -> ${ISSUE_URL}"
            echo "[OK] ${owner_repo} — ${ISSUE_URL}" >> "$LOG_FILE"
            success=$((success + 1))
        elif ISSUE_URL=$(gh issue create \
            --repo "${owner_repo}" \
            --title "${ISSUE_TITLE}" \
            --body-file "$COMBINED_BODY_FILE" 2>&1); then
            echo "OK (no label) -> ${ISSUE_URL}"
            echo "[OK] ${owner_repo} — ${ISSUE_URL} (label 'security' not available)" >> "$LOG_FILE"
            success=$((success + 1))
        else
            echo "FAILED: ${ISSUE_URL}"
            echo "[FAIL] ${owner_repo} — ${ISSUE_URL}" >> "$LOG_FILE"
            failed=$((failed + 1))
        fi

        rm -f "$COMBINED_BODY_FILE"

        # Rate limiting: 5-second delay between issue creations
        if [[ $total -lt $REPO_COUNT ]]; then
            echo "  (waiting 5s for rate limiting...)"
            sleep 5
        fi
    fi
done

# Clean up
rm -f "$LOOKUP_FILE"

echo ""
echo "=== Summary ==="
echo "Total:   ${total}"
echo "Success: ${success}"
echo "Failed:  ${failed}"
echo "Skipped: ${skipped}"
echo ""
echo "Summary: total=${total} success=${success} failed=${failed} skipped=${skipped}" >> "$LOG_FILE"

if ! $DRY_RUN && [[ $success -gt 0 ]]; then
    echo "Issue URLs logged to: ${LOG_FILE}"
    echo "Remember to update tracking.md with issue URLs and sent dates."
fi
