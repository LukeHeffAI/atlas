#!/usr/bin/env bash
# git-detective-search.sh — Automated broad search for lost work
#
# Usage: bash git-detective-search.sh <keyword> [--author <name>] [--after <date>] [--before <date>] [--path <path>]
#
# Runs all major forensics searches in one pass and outputs a structured report.
# Designed to be called by Claude as a first-pass investigation tool.

set -euo pipefail

KEYWORD=""
AUTHOR=""
AFTER=""
BEFORE=""
PATH_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --author) AUTHOR="$2"; shift 2 ;;
        --after) AFTER="$2"; shift 2 ;;
        --before) BEFORE="$2"; shift 2 ;;
        --path) PATH_FILTER="$2"; shift 2 ;;
        *) KEYWORD="$1"; shift ;;
    esac
done

if [[ -z "$KEYWORD" ]]; then
    echo "Usage: bash git-detective-search.sh <keyword> [--author <name>] [--after <date>] [--before <date>] [--path <path>]"
    exit 1
fi

# Build common flags
DATE_FLAGS=""
[[ -n "$AFTER" ]] && DATE_FLAGS="$DATE_FLAGS --after=$AFTER"
[[ -n "$BEFORE" ]] && DATE_FLAGS="$DATE_FLAGS --before=$BEFORE"

AUTHOR_FLAG=""
[[ -n "$AUTHOR" ]] && AUTHOR_FLAG="--author=$AUTHOR"

PATH_ARG=""
[[ -n "$PATH_FILTER" ]] && PATH_ARG="-- $PATH_FILTER"

echo "============================================"
echo "GIT DETECTIVE — Broad Search Report"
echo "============================================"
echo "Keyword:  $KEYWORD"
[[ -n "$AUTHOR" ]] && echo "Author:   $AUTHOR"
[[ -n "$AFTER" ]] && echo "After:    $AFTER"
[[ -n "$BEFORE" ]] && echo "Before:   $BEFORE"
[[ -n "$PATH_FILTER" ]] && echo "Path:     $PATH_FILTER"
echo "============================================"
echo ""

# 1. Commit message search
echo "--- COMMIT MESSAGE MATCHES ---"
eval git log --all --oneline --grep="$KEYWORD" -i $DATE_FLAGS $AUTHOR_FLAG $PATH_ARG 2>/dev/null | head -30 || echo "(no matches)"
echo ""

# 2. Pickaxe search (code changes)
echo "--- CODE CHANGE MATCHES (pickaxe -S) ---"
eval git log --all -S "\"$KEYWORD\"" --oneline --format=\"%h %ad %an  %s\" --date=short $DATE_FLAGS $AUTHOR_FLAG $PATH_ARG 2>/dev/null | head -30 || echo "(no matches)"
echo ""

# 3. Branch name search
echo "--- MATCHING BRANCHES ---"
git branch -a 2>/dev/null | grep -i "$KEYWORD" || echo "(no matches)"
echo ""

# 4. Deleted files matching keyword
echo "--- DELETED FILES MATCHING KEYWORD ---"
git log --all --diff-filter=D --name-only --format="" $DATE_FLAGS 2>/dev/null | sort -u | grep -i "$KEYWORD" | head -20 || echo "(no matches)"
echo ""

# 5. All files ever created matching keyword
echo "--- FILES EVER CREATED MATCHING KEYWORD ---"
git log --all --pretty=format: --name-only --diff-filter=A $DATE_FLAGS 2>/dev/null | sort -u | grep -i "$KEYWORD" | head -30 || echo "(no matches)"
echo ""

# 6. Reflog search (local repos only)
echo "--- REFLOG MATCHES ---"
git reflog --all --format="%h %gd %gs" 2>/dev/null | grep -i "$KEYWORD" | head -20 || echo "(no matches or no reflog)"
echo ""

# 7. Stash search
STASH_COUNT=$(git stash list 2>/dev/null | wc -l)
if [[ $STASH_COUNT -gt 0 ]]; then
    echo "--- STASH MATCHES ($STASH_COUNT stashes) ---"
    for i in $(seq 0 $(( STASH_COUNT - 1 ))); do
        if git stash show -p "stash@{$i}" 2>/dev/null | grep -qi "$KEYWORD"; then
            echo "  MATCH: stash@{$i} — $(git stash list | sed -n "$((i+1))p")"
        fi
    done
    echo ""
else
    echo "--- STASH MATCHES ---"
    echo "(no stashes)"
    echo ""
fi

echo "============================================"
echo "Search complete. Use specific SHAs above for deeper investigation."
echo "============================================"
