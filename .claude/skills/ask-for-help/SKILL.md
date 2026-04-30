---
name: ask-for-help
description: >
  Behavioral guardrail that stops Claude from guessing when it lacks access to external resources
  (DBs, APIs, remote services, schedulers, logs, running processes, deployed environments).
  Trigger on permission errors, network errors, connection refused, timeouts, speculative output
  about what a system "probably" contains, or debugging that depends on runtime state Claude
  cannot observe (Celery queues, container logs, deployed config, env vars on remote hosts).
  Overrides the instinct to guess.
---

# Ask For Help — Stop Guessing, Start Asking

**Work Style.** `CLAUDE.md` §Work Style applies — batch up to 3 questions per ask (one round-trip beats three), structured input over prose, terse output, trust the dispatcher's stated facts.

## The Problem This Solves

When Claude can't access a resource (DB, API, scheduler, remote service, log file, deployed environment), its defaults are:

1. Guess what the resource contains and proceed on assumptions
2. Retry the same failing command with minor variations
3. Write speculative code based on imagined state
4. Generate plausible-sounding but fabricated error analyses
5. Spiral through multiple failed attempts without pausing

**This wastes the user's time and produces unreliable output.** The user is sitting right there and can get the information in seconds.

## Core Rules

### The 2-Strike Rule

If you attempt to access a resource and fail:

- **Strike 1**: try ONE reasonable alternative (different path, different command, check for a local copy).
- **Strike 2**: if that fails, **STOP IMMEDIATELY** and ask the user.

Do NOT try a third approach. Do NOT speculate about contents.

### UX-First Help Requests

**The #1 rule: make it effortless for the user to respond.** The user should answer with a tap wherever possible — not by typing prose.

#### Use structured input (multi-choice) when the answer can be a choice

Use `ask_user_input_v0` (or equivalent) for:

- **Diagnosing environment**: "What environment is this running in?" → `Local dev` / `Docker` / `Cloud` / `CI`
- **Choosing next steps**: "Can't reach the DB. How proceed?" → `I'll paste output` / `Skip DB for now` / `Use mock data` / `Let me fix it`
- **Confirming assumptions**: "API on port 8000?" → `Yes` / `No, different port` / `Not sure`
- **Gathering context**: "Which DB engine?" → `PostgreSQL` / `MySQL` / `SQLite` / `MongoDB`

Ask up to 3 questions at once — batching is encouraged. One interaction beats slow back-and-forth.

**Example — proactive context gathering:**

```
Q1: "Where is this deployed?"           [Local Docker, AWS ECS, GCP Cloud Run, Bare metal/VM]
Q2: "Can you access the server?"        [Yes, No, Partial access only]
Q3: "Is the database reachable from your machine?"  [Yes, No, Not sure]
```

#### When free-text is needed (logs, query results, errors, config contents)

1. **Triage with a structured question first**: "I need the application logs. Can you access them?" → `Yes, I'll paste` / `I can run a command` / `No access`
2. **Then provide a specific, copy-pasteable command** if they say yes:
   > Paste the output of:
   > ```
   > docker logs celery-worker --tail 50
   > ```
3. **Always provide the exact command.** Never say "check the logs" without specifying which logs and how.

### Never Do These

- **Never fabricate resource contents.** No "the DB probably has a users table with id, name, email…" without evidence.
- **Never silently assume.** If you must assume to proceed, flag it explicitly: "⚠️ ASSUMPTION: API returns JSON with a `results` key. Please confirm."
- **Never retry more than once.** Two attempts max, then ask.
- **Never diagnose without data.** "The issue is probably X" → replace with "I'd need to see X to diagnose — could you run [specific command]?"
- **Never ask open-ended questions when closed ones will do.** "What DB are you using?" is worse than `PostgreSQL` / `MySQL` / `SQLite` / `Other`.
- **Never dump a wall of questions in prose.** Convert to structured input.

## Recognising When You're Spinning

These patterns in your own behaviour mean **stop immediately and ask**:

**Self-correction / second-guessing (the #1 missed trigger):**

- You write "No, wait..." / "Hmm, actually..." / "On second thought..."
- You re-interpret user intent: "Maybe the user meant..." / "Perhaps they wanted..."
- You argue with yourself: "Well, it could be X, but it might also be Y..."
- You hedge mid-action: "Let me reconsider..." / "That doesn't seem right..."
- You narrate uncertainty: "I'm not sure if..." / "This might not be..."

If any of these appear, that is the signal. Ask the user — they resolve the ambiguity in seconds.

**Speculation and fabrication:**

- Paragraphs starting with "likely", "probably", "presumably", "I would expect"
- Generating mock / example data to "illustrate" what a system might return
- Writing error handling for errors you haven't seen
- Reverse-engineering system state from code alone instead of observing it

**Looping and retrying:**

- 2+ different approaches to the same resource
- "Let me try another approach" for the third time
- Tweaking a command slightly and re-running, hoping for a different result

**Rule: the moment you feel uncertain about what the user wants or what a system contains, ask.**

## Proactive Asking

Don't wait until you fail. If a task **will obviously require** access you don't have (production DB, running service, remote server), ask upfront before writing any code. Use structured input to gather everything in one round-trip:

```
Q1: "Needs DB access. Can you run queries?"  [Yes, direct | Yes, via tool | No, have recent dump | No access]
Q2: "Env config — where?"                     [I'll paste | .env I can share | Secrets manager | Not sure]
Q3: "App logs I should see?"                  [Yes, I'll grab | CloudWatch/Datadog/etc. | No logs | Not sure]
```

## Batching

If you need multiple pieces of information, ask for all of them at once. 3 tappable questions in 5 seconds beats 3 separate free-text questions across 3 messages.

## How to Format Your Ask

### Preferred: structured input

Use `ask_user_input_v0` with 1–3 questions, 2–4 options each. Precede it with 1–2 sentences explaining what you hit and why you need help.

### Fallback: when you need raw output

```
🔍 **I need your help to proceed.**

I can't access [specific resource] from here because [brief reason].

Could you please run the following and paste the output?

```bash
[exact command(s)]
```

[Optional: "While waiting, I'll continue working on [other part that doesn't need this info]."]
```

The "while waiting" note matters — if other work doesn't block on the missing info, do it in parallel.

### Anti-pattern: the wall of questions

**DON'T:**
> I need some information. What DB engine? What host and port? Is it in Docker or native? Do you have psql? Can you run a query? What's the schema for orders? Also, PostgreSQL version?

**DO:** Structured input for the categorical questions (engine, deployment, access level). Based on answers, follow up with one targeted command if still needed.
