#!/bin/bash
# Fetch GitHub community-growth signals for the upstream repo and write them
# into the report's dated output directory.
#
# Usage: fetch_github_stats.sh <output_dir>
#
# Writes two files into <output_dir>:
#   - github_stats.json            (stars, forks, watchers, open_issues)
#   - github_contributors_count.txt (count of unique contributor logins)
#
# If `gh` is unauthenticated or an API call fails, the script logs a note and
# exits 0 (the GitHub section is simply skipped in the report) so the overall
# usage-report run never fails on this optional step.

set -u

REPO="agentic-community/mcp-gateway-registry"

OUTPUT_DIR="${1:-}"
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: output directory argument required" >&2
    echo "Usage: fetch_github_stats.sh <output_dir>" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

STATS_FILE="$OUTPUT_DIR/github_stats.json"
CONTRIBUTORS_FILE="$OUTPUT_DIR/github_contributors_count.txt"

# Star, fork, watcher, open-issue counts (single API call).
if ! gh api "repos/$REPO" \
    --jq '{stars: .stargazers_count, forks: .forks_count, watchers: .subscribers_count, open_issues: .open_issues_count}' \
    > "$STATS_FILE" 2>/dev/null; then
    echo "NOTE: gh api for repo stats failed (unauthenticated or network); skipping GitHub stats" >&2
    rm -f "$STATS_FILE"
    exit 0
fi

# Unique contributors (paginate through all pages, count unique logins).
if ! gh api --paginate "repos/$REPO/contributors" \
    --jq '.[].login' 2>/dev/null | sort -u | wc -l > "$CONTRIBUTORS_FILE"; then
    echo "NOTE: gh api for contributors failed; skipping contributor count" >&2
    rm -f "$CONTRIBUTORS_FILE"
fi

echo "Stats:"
cat "$STATS_FILE"
if [ -f "$CONTRIBUTORS_FILE" ]; then
    echo "Contributors:"
    cat "$CONTRIBUTORS_FILE"
fi
