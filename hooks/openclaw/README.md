# OpenClaw hooks

Three lifecycle hooks that connect OpenClaw to Jarvis-Memory. They run on OpenClaw's hook runtime (TypeScript, transpiled to JavaScript) and write to the Jarvis REST API on the same machine.

## The three hooks

| Hook | Fires on | What it does |
|---|---|---|
| **`jarvis-start-guard/`** | agent bootstrap / session start | Resolves `group_id`, calls `wake_up`, loads the latest handoff, injects a short context block into the session. |
| **`jarvis-handoff-enforcer/`** | `command:new`, `command:reset`, `session:compact:before` | Writes a `[HANDOFF]` episode via `/api/v2/session/handoff` before the session gets reset or compacted. |
| **`mem0-extractor/`** | `session:compact:before` | Scrapes the transcript for memorable content and calls `/api/v1/add` (legacy endpoint, still routed through the v2 pipeline). |

## Source-of-truth note

These hooks lived in `brain/projects/astack/hooks/` until 2026-04-23, which was an accident of initial project layout. They're memory-related infrastructure, so they belong in this repo (jarvis-memory) where they can be versioned alongside the REST/MCP contract they depend on. Astack (the productized install bundle) pulls them from here at bundle-build time.

## Known drift (cleaned up incrementally)

- Current hooks call `/api/v1/add` and hand-rolled HTTP; a follow-up refactor will route them through the centralized `jarvis_memory.handoff` module to pick up the strict `group_id` contract + idempotency keys for free.
- Codex's long-term proposal (Phase 3 of `2026-04-23-jarvis-handoff-contract-implementation-plan.md`) consolidates all three into a single `jarvis-memory-bridge` hook. That consolidation is deferred until the current ports have run alongside the new REST contract for a week or two.
