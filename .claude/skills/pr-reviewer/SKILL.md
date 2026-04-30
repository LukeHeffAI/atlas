---
name: pr-reviewer
description: >
  Reviewer that fetches a GitHub PR, confirms details with the user, then assesses, resolves
  conflicts, makes improvements, and pushes — but never merges. Activated via "/pr-reviewer
  NUMBER" (e.g. "/pr-reviewer 16"); also triggers on "review PR", "check pull request",
  "look at PR #N", or any reference to a GitHub PR number for review. Requires `GITHUB_TOKEN`.
allowed-tools: Bash(gh *), Bash(git *), Read, Grep, Glob, Agent
argument-hint: "[PR number] [optional focus area]"
---

# PR Reviewer

Review a GitHub PR: fetch, assess, improve, push. **Never merge** — leave that to the user.

The cross-cutting rules in `CLAUDE.md` §Work Style apply (parallel tool calls, evidence hierarchy, `gh pr view` over serial git probes, `&&`-batched API calls, dispatcher-only lint/tests). This skill defines the PR-specific workflow on top of those.

## Prerequisites

- `GITHUB_TOKEN` with repo access
- `gh` CLI. Fall back to `curl` only if `gh` errors — report the error to the user first, don't silent-fall-back
- Inside the target repo (or inferable via `gh repo view --json nameWithOwner -q .nameWithOwner` / `.git/config`)

## Workflow

### 1. Parse & fetch (one turn)

Extract the PR number and any trailing focus guidance (e.g. `/pr-reviewer 16 focus on error handling` → N=16, focus=error handling).

Then issue the metadata calls in a single `Bash` chained with `&&`:

```bash
gh pr view <N> --json title,body,headRefName,baseRefName,state,mergeable,mergeStateStatus,url,additions,deletions,changedFiles && \
gh pr diff <N> && \
gh api repos/<REPO>/pulls/<N>/comments --jq '.[] | {id, user: .user.login, path, line, body}'
```

Use the diff (and the inline-comment JSON) as the primary review surface. Do not `Read` files whose change is fully visible in the diff.

### 2. Confirm scope

Before making changes, confirm three items with the user in ONE turn:

1. Head branch
2. Base branch
3. 2–3 sentence summary of what the PR does (synthesised from title/body/diff)

If the parent conversation has already confirmed these (e.g. operator dispatched with "review approved"), skip the gate and proceed.

### 3. Sync the branch

If you're already in a worktree on the PR's head branch, skip entirely — you're synced. Otherwise:

```bash
git fetch origin <head_branch> && git checkout <head_branch>
```

Do NOT follow up with `git status`, `git log`, or `git pull` "to check" — `gh pr view` already returned the state you need.

### 4. Resolve conflicts (only if needed)

Only run a merge preview if `mergeStateStatus != CLEAN`:

```bash
git merge origin/<base_branch> --no-commit --no-ff
```

Resolve intelligently: prefer the head's intent where the PR is deliberate; prefer base where the head has drifted on lockfiles / formatting. Ask the user when a conflict is genuinely ambiguous (competing design intents, not mechanical). Stage, commit as `resolve merge conflicts with <base_branch>`. If the preview shows no conflicts, `git merge --abort` and move on.

### 5. Review the changes

For each changed file, assess in this order:

- **Correctness** — logic bugs, off-by-ones, races, null safety, unhandled error paths
- **Design** — naming, structure, separation of concerns, DRY, abstraction level
- **Tests** — new code paths covered; existing tests updated if behaviour changed
- **Security** — injection, auth, secrets in code, unsafe deserialization, SSRF
- **Performance** — unnecessary allocations, N+1 queries, blocking calls in async contexts
- **Documentation** — public-API docstrings, comments accurate post-change

Read every open comment and review thread once. Evaluate on technical merit, not authorship. Build a mental ledger of each comment's ID, author, request, and your planned response — you will reply to all of them in Step 8.

### 6. Manual verification steps (optional)

If the PR description flags steps for human verification (e.g. confirm values in a live DB inside Docker), run what you can. Ask for permission before touching production data or live systems.

Do NOT run project-wide validation (`ruff`, `mypy`, `npx tsc`, `npx eslint`, `pytest`, untargeted `vitest`) — per CLAUDE.md, those are the dispatcher's job. A single targeted test file for a change you wrote is fine.

### 7. Delegate to specialists when appropriate

Spawn specialist agents via the `Agent` tool for depth on matching domains:

Always-vendored agents:
- ML code → `ml-engineer`
- Frontend / UI → `frontend-engineer`
- Backend / API / Django / Celery / DB → `backend-engineer`
- Data / SQL / analytics logic → `data-analyst`
- Cross-layer refactor → `systems-architect`

Environment-dependent agents (dispatch only if present in `.claude/agents/` or the harness's built-in set):
- UX / interaction depth → `ux-design-advisor` (otherwise raise the question to the user)
- Greyhound-domain code → `greyhound-racing-expert` (otherwise treat as ordinary backend/ML code)

Pass the PR number, your ledger of concerns, and a scope directive in the dispatch prompt. Incorporate their findings.

### 8. Make changes and push

For every issue worth fixing:

- **Clear improvement, low-risk** → fix it directly. One logical commit per concern with a descriptive message. No mega-commits.
- **High-risk or design-level** → ask the user before changing.

Then:

```bash
git push origin <head_branch>
```

### 9. Reply to comments

Batch inline-comment replies into a **single `Bash` turn** chained with `&&`. This preserves the individual threaded replies on the PR while paying only one turn's cache cost:

```bash
gh api repos/<REPO>/pulls/<N>/comments/<ID1>/replies -X POST -f body="<reply 1>" && \
gh api repos/<REPO>/pulls/<N>/comments/<ID2>/replies -X POST -f body="<reply 2>" && \
gh api repos/<REPO>/pulls/<N>/comments/<ID3>/replies -X POST -f body="<reply 3>"
```

Top-level issue-thread comments use `gh pr comment <N> --body "..."` — group by topic so related comments get one reply, but don't pack unrelated topics into a wall of text.

Each reply is short and technical:

- **Fixed** → `Fixed in <sha> — <1-line what>.`
- **Partially addressed** → `<what was done>; <what remains>.`
- **Intentional no-change** → `Kept because <reason>.`
- **Deferred** → `Flagged for @<pr-author> — design call.`

No sycophancy ("Great catch!"). Just the substance.

### 10. Summarise

Report back to the user:

- **Reviewed** — files / areas examined
- **Issues** — list with severity and disposition (fixed / flagged / deferred)
- **Conflicts resolved** — resolution strategy if any
- **Commits pushed** — list with SHAs
- **Comments responded to** — count and topics
- **Recommendation** — ready to merge / blocker / operator decision needed

Under 400 words. **Do not merge the PR.**

## Edge cases

- `GITHUB_TOKEN` missing or API returns 401/403 → stop, surface the error, do not retry with different auth.
- PR already merged or closed → stop and inform the user.
- Diff > 5000 lines → warn the user, ask whether to focus on specific files/directories.
- No network access to the GitHub API → invoke the `ask-for-help` skill.
