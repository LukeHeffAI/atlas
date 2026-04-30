# Git Forensics Command Cheatsheet

Quick reference for all git commands used in investigations. Organised by use case.

## Table of Contents
1. [Searching Commit Messages](#searching-commit-messages)
2. [Searching Code Changes (Pickaxe)](#searching-code-changes)
3. [Filtering by Author/Date/Path](#filtering)
4. [Branch Archaeology](#branch-archaeology)
5. [Deleted File Recovery](#deleted-file-recovery)
6. [Diff and Show](#diff-and-show)
7. [Blame and Annotation](#blame-and-annotation)
8. [Reflog and Dangling Objects](#reflog-and-dangling-objects)
9. [Merge Forensics](#merge-forensics)
10. [Stash Searching](#stash-searching)
11. [Performance Tips for Large Repos](#performance-tips)
12. [Output Formatting](#output-formatting)

---

## Searching Commit Messages

```bash
# Single keyword in commit message
git log --all --oneline --grep="keyword"

# Multiple keywords (AND — all must match)
git log --all --oneline --grep="keyword1" --grep="keyword2" --all-match

# Multiple keywords (OR — any match)
git log --all --oneline --grep="keyword1" --grep="keyword2"

# Case-insensitive
git log --all --oneline --grep="keyword" -i

# Regex in commit message
git log --all --oneline --grep="feat(auth|user):" --perl-regexp
```

## Searching Code Changes

The pickaxe (`-S`) and regex (`-G`) flags are the most powerful forensics tools.

```bash
# -S: finds commits where the count of <string> changed (added or removed)
# This is the go-to for "when was this code introduced/deleted"
git log --all -S "function_name" --oneline --format="%h %ad %an %s" --date=short

# -G: finds commits where the diff matches <regex>
# More flexible but noisier — catches reformatting, moves, etc.
git log --all -G "class\s+NotificationHandler" --oneline --format="%h %ad %an %s" --date=short

# -S with --pickaxe-regex: treat the -S argument as regex
git log --all -S "notify_user|send_notification" --pickaxe-regex --oneline

# Show the actual diff for each match
git log --all -S "function_name" -p

# Limit to specific files/directories
git log --all -S "function_name" -- "src/notifications/"
```

**Key difference**: `-S` finds commits that *change the number of occurrences* of a string. `-G` finds commits whose diff *contains* the pattern. Use `-S` for "when was X added/removed" and `-G` for "when was the line matching X touched".

## Filtering

```bash
# By author
git log --all --author="Luke" --oneline --stat
git log --all --author="luke@" --oneline  # partial email match works

# By date range
git log --all --after="2024-06-01" --before="2024-12-31" --oneline

# By path (works with globs)
git log --all -- "src/api/**/*.py"
git log --all -- "*notification*"

# By diff type: A=Added, D=Deleted, M=Modified, R=Renamed, C=Copied
git log --all --diff-filter=D -- "*.py"   # find deleted Python files
git log --all --diff-filter=A -- "src/"   # find files added to src/

# Combine everything
git log --all --author="Luke" --after="2024-06-01" -S "handler" -- "src/" --oneline
```

## Branch Archaeology

```bash
# All branches (local + remote)
git branch -a

# Branches containing a specific commit
git branch -a --contains <sha>

# Branches that have been merged into main
git branch -a --merged main

# Branches NOT merged into main (likely orphaned feature work)
git branch -a --no-merged main

# Search branch names
git branch -a | grep -i "notification"

# Show the tip commit of each remote branch
git for-each-ref --sort=-committerdate refs/remotes/ \
  --format="%(refname:short) %(committerdate:short) %(authorname) %(subject)"

# Find which branch a commit came from (heuristic — not always accurate)
git name-rev <sha>

# List recently deleted branches from reflog
git reflog | grep "checkout:" | grep -oP "moving from \K\S+"  | sort -u
```

## Deleted File Recovery

```bash
# Find the commit that deleted a specific file
git log --all --diff-filter=D --summary -- "*filename*"

# Find ALL files ever deleted
git log --all --diff-filter=D --summary --format="%h %ad %s" --date=short | grep "delete mode"

# List all files that ever existed in the repo
git log --all --pretty=format: --name-only | sort -u

# Recover a deleted file at its last known state
git log --all --diff-filter=D -- "path/to/file"  # find the deleting commit
git show <deleting_commit>^:<path/to/file>         # show file just before deletion

# Find all files matching a pattern that ever existed
git log --all --pretty=format: --name-only --diff-filter=A | sort -u | grep -i "pattern"
```

## Diff and Show

```bash
# Show a specific commit's changes
git show <sha> --stat                    # summary
git show <sha> -- <file>                 # specific file only
git show <sha> -p                        # full patch

# Show a file at a specific point in time
git show <sha>:<filepath>

# Diff between two commits for a specific file
git diff <sha_a> <sha_b> -- <filepath>

# Diff with stats (file-level summary)
git diff --stat <sha_a> <sha_b>

# Diff showing only file names that changed
git diff --name-only <sha_a> <sha_b>
git diff --name-status <sha_a> <sha_b>   # also shows A/M/D/R status

# Word-level diff (useful for spotting subtle changes)
git diff --word-diff <sha_a> <sha_b> -- <file>
```

## Blame and Annotation

```bash
# Standard blame
git blame <file>
git blame <file> -L 50,80               # specific line range

# Blame at a historical point
git blame <sha> -- <file>

# Blame through renames and code movement
git blame -C <file>                      # detect moves within same commit
git blame -C -C <file>                   # detect moves from other files in same commit
git blame -C -C -C <file>               # detect moves from ANY commit

# Blame showing the original commit (before any moves)
git blame -C -C -C --show-email <file>

# Reverse blame: find when a line was removed
# (use -S pickaxe instead — more reliable)
git log -p -S "exact line content" -- <file>
```

## Reflog and Dangling Objects

```bash
# Full reflog (local actions — survives rebases and force-pushes)
git reflog --all --format="%h %gd %gs %s" | grep -i "keyword"

# Reflog for a specific branch
git reflog show <branch_name>

# Find dangling commits (orphaned by rebase, amend, branch delete)
git fsck --no-reflogs --unreachable | grep commit

# Search dangling commits for a keyword
git fsck --no-reflogs --unreachable 2>/dev/null | grep "^unreachable commit" | \
  awk '{print $3}' | xargs -I{} git log -1 --format="%H %ad %s" --date=short {} 2>/dev/null | \
  grep -i "keyword"

# Find commits reachable from reflog but not from any current branch
git log --oneline --all --reflog --not --branches --remotes

# Reflog expiry: default 90 days (reachable), 30 days (unreachable)
# Check: git config gc.reflogExpire
```

## Merge Forensics

```bash
# Find merge commits that brought a feature into main
git log --all --merges --ancestry-path <feature_commit>..<main> --oneline

# Show what happened during a specific merge
git show <merge_sha> --stat
git show <merge_sha> -m -p              # show diff against each parent

# Find the merge base (common ancestor) of two branches
git merge-base <branch_a> <branch_b>

# Show all commits in branch_a that are NOT in branch_b
git log <branch_b>..<branch_a> --oneline

# Show the merge resolution for a specific file
git show <merge_sha> -- <file>

# Check if a merge had conflicts (look for conflict markers in the diff)
git log --all --merges --format="%H %s" | while read sha msg; do
  files=$(git diff-tree --no-commit-id -r --diff-filter=U "$sha" 2>/dev/null | wc -l)
  [ "$files" -gt 0 ] && echo "$sha $msg ($files conflicts)"
done
```

## Stash Searching

```bash
# List all stashes
git stash list

# Show a specific stash's contents
git stash show -p stash@{0}

# Search all stashes for a keyword
for i in $(seq 0 $(( $(git stash list | wc -l) - 1 ))); do
  if git stash show -p "stash@{$i}" 2>/dev/null | grep -q "keyword"; then
    echo "Found in stash@{$i}: $(git stash list | sed -n "$((i+1))p")"
  fi
done
```

## Performance Tips

For repos with 10k+ commits or large working trees:

```bash
# ALWAYS constrain with path and date when possible
git log --all -S "string" --after="2024-01-01" -- "src/" --oneline

# Use --first-parent for main branch history (skips individual feature commits)
git log --first-parent main -S "string" --oneline

# Use rev-list for counting before running expensive operations
git rev-list --count --all --after="2024-01-01"  # how many commits to search?

# Parallel pickaxe: split by date range if needed
git log --all -S "str" --after="2024-01" --before="2024-07" -- "src/" &
git log --all -S "str" --after="2024-07" --before="2025-01" -- "src/" &
wait

# For very large repos, consider using git's internal pathspec magic
git log --all -- ":(glob)**/notification*"  # matches any depth
```

## Output Formatting

```bash
# Custom log format (useful for reports)
git log --format="%h %ad %an %s" --date=short

# With stats (files changed, insertions, deletions)
git log --oneline --stat

# One-liner with commit size
git log --oneline --shortstat

# JSON-ish output for scripting
git log --format='{"sha":"%h","date":"%ad","author":"%an","subject":"%s"}' --date=iso

# Graph view for understanding branch topology
git log --all --oneline --graph --decorate

# Limit output
git log -n 20 --oneline                  # last 20 commits
```
