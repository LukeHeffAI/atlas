# Building Guide — Skills, Agents, and Hooks

Reference material for constructing new Claude Code artifacts. Read this when you need
to build a skill, agent, or hook as part of a workflow optimization proposal.

---

## Building Skills

When creating a new skill, follow these standards. The goal is skills that are
genuinely excellent — not boilerplate, not generic, but the kind of tooling that
makes the user wonder how they ever worked without it.

### SKILL.md Structure

```
---
name: [kebab-case-name]
description: >
  [Pushy, comprehensive description. Include what it does AND when to trigger.
  Err on the side of triggering too often rather than too rarely. List specific
  phrases, contexts, and adjacent tasks that should activate this skill.]
---

# [Name]

[Identity statement — who is the model when using this skill? Make it aspirational
and specific. Not "you write code" but "you are potentially the most meticulous and
creative [domain] specialist the world has ever seen."]

[Core instructions — what to do, how to think about it, why it matters]

[Output format — what the deliverable looks like]

[Edge cases and gotchas — things that commonly go wrong]

[Examples — at least one concrete input/output pair]
```

### Quality Bar

Every skill you create should:
- Have a description that's slightly "pushy" about when to trigger — undertriggering
  is worse than overtriggering
- Open with an aspirational identity statement that primes excellence
- Explain the *why* behind every instruction, not just the *what*
- Include at least one concrete example
- Handle edge cases explicitly
- Be under 500 lines (use reference files for overflow)

### Bundled Resources

If the skill needs supporting files:
```
skill-name/
├── SKILL.md
├── scripts/       # Executable helpers for deterministic tasks
├── references/    # Additional docs loaded as needed
├── templates/     # Reusable output templates
└── assets/        # Static files (icons, fonts, etc.)
```

Reference these clearly from SKILL.md with guidance on when to read them.

---

## Building Agents

Agents live in `.claude/agents/` as markdown files. Each agent is a persona with
a specific operational mandate.

### Agent File Structure

```markdown
# [Agent Name]

## Identity
[Who is this agent? What makes them exceptional at their specific job?
Be vivid and aspirational — "You are the most disciplined and eagle-eyed
code reviewer in the industry" not "You review code."]

## Mandate
[What is this agent responsible for? What decisions can it make autonomously
vs. what should it escalate?]

## Operating Principles
[How does this agent think? What does it prioritize? What does it refuse to
compromise on?]

## Input
[What does this agent receive? What context does it need?]

## Output
[What does this agent produce? What format? What quality bar?]

## Boundaries
[What is explicitly NOT this agent's job? Where does it hand off?]
```

### Quality Bar

Agents should:
- Have a vivid, specific identity — not generic
- Be scoped tightly enough to be excellent at one thing
- Know their boundaries and escalation paths
- Complement existing agents without overlap

---

## Building Hooks

Hooks are automation triggers that run at specific points in the workflow.
They can be Claude Code hooks (in `.claude/hooks/`) or git hooks.

### Claude Code Hooks

These fire on specific Claude Code events. Define them in `.claude/hooks.json`
or equivalent configuration:

```json
{
  "hooks": {
    "pre-commit": {
      "description": "What this hook checks before committing",
      "action": "Script or command to run",
      "fail_behavior": "block | warn"
    },
    "post-tool-use": {
      "description": "What this hook does after a tool is used",
      "action": "Script or command to run"
    }
  }
}
```

### Git Hooks

Standard git hooks in `.git/hooks/` or managed via a framework like `pre-commit`.
These are good for:
- Linting and formatting enforcement
- Commit message standardisation
- Test execution before push
- Secret detection

### Quality Bar

Hooks should:
- Be fast — they run in the critical path, so keep execution time minimal
- Be clear about whether they block or warn
- Have good error messages that explain *what's wrong and how to fix it*
- Not be annoying — if a hook fires too often on false positives, it'll get disabled
