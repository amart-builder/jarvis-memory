# OpenClaw hooks

Three lifecycle hooks that connect OpenClaw to Jarvis-Memory. They run on OpenClaw's hook runtime (TypeScript, transpiled to JavaScript) and write to the Jarvis REST API on the same machine.

## The three hooks

| Hook | Fires on | What it does |
|---|---|---|
| **`jarvis-start-guard/`** | `agent:bootstrap`, `message:sent` | Calls `/api/v2/handoff/latest` for each watched `group_id`, detects cross-system handoffs from Claude Code / CLI / REST, injects a warning block into the agent's bootstrap files, and fires a START-GUARD VIOLATION if the first assistant reply doesn't acknowledge the handoff. |
| **`jarvis-handoff-enforcer/`** | `command:new`, `command:reset`, `session:compact:before` | Writes a snapshot + retrievable `[HANDOFF]` episode via `POST /api/v2/session/handoff`. Uses a deterministic `idempotency_key` so retries and duplicate triggers collapse server-side. |
| **`mem0-extractor/`** | `session:compact:before` | Posts the recent transcript to `/api/v1/add` for Haiku-based fact extraction. Tags writes with a `group_id` (default `system`) so extracted facts route into a project-scoped bucket instead of the legacy `system` fallback. |

## Contract compliance (v1.1+)

All three hooks comply with `HANDOFF_CONTRACT.md`:

- **Strict `group_id`** — `jarvis-handoff-enforcer` infers it from the transcript; `mem0-extractor` reads `MEM0_GROUP_ID` env var (default `system`); `jarvis-start-guard` watches a list of `group_id`s.
- **Server-side idempotency** — `jarvis-handoff-enforcer` sends a deterministic `idempotency_key` derived from `sha256(trigger + sessionKey + notes)`. Retrying the same logical event collapses to one write.
- **`session_key` + `source` tagging** — every write carries both, so later analysis can correlate across surfaces and filter by origin.
- **Bearer auth support** — reads `JARVIS_API_BEARER_TOKEN` and sends `Authorization: Bearer <token>` when set. Unset means localhost allow-all (normal local dev).

## Configuration (env vars)

| Var | Default | Used by |
|---|---|---|
| `JARVIS_MEMORY_API` | `http://localhost:3500` | all |
| `JARVIS_API_BEARER_TOKEN` | *(empty)* | all — adds `Authorization` header when set |
| `JARVIS_DEVICE_ID` | `openclaw` | handoff-enforcer — labels the device on the Episode |
| `JARVIS_HOOK_MAX_MESSAGES` | `24` | handoff-enforcer — transcript window |
| `JARVIS_HOOK_DRY_RUN` | *(off)* | handoff-enforcer — log payload, skip POST |
| `JARVIS_START_GUARD_GROUP_IDS` | *(default list)* | start-guard — comma-separated `group_id`s to watch |
| `JARVIS_START_GUARD_LOOKBACK_HOURS` | `2` | start-guard — how far back to consider a handoff "recent" |
| `JARVIS_START_GUARD_MAX_WARNINGS` | `3` | start-guard — cap on injected warnings per bootstrap |
| `JARVIS_START_GUARD_STATE_FILE` | `~/.openclaw/jarvis-start-guard-state.json` | start-guard — local state for pending-acknowledgement tracking |
| `MEM0_URL` | `http://localhost:3500` | mem0-extractor |
| `MEM0_USER_ID` | `user` | mem0-extractor — identity tag for extracted facts |
| `MEM0_GROUP_ID` | `system` | mem0-extractor — `group_id` at top level per the contract |
| `MEM0_MAX_MESSAGES` | `40` | mem0-extractor — transcript window |

## Source-of-truth note

These hooks lived in `brain/projects/astack/hooks/` until 2026-04-23, which was an accident of initial project layout. They're memory-related infrastructure, so they belong in this repo (jarvis-memory) where they can be versioned alongside the REST/MCP contract they depend on. Astack (the productized install bundle) now pulls them from here at bundle-build time.

## Future consolidation

Codex's long-term proposal (Phase 3 of `2026-04-23-jarvis-handoff-contract-implementation-plan.md`) consolidates all three hooks into a single `jarvis-memory-bridge`. That's deferred until the current three-hook setup has run against the new contract for a couple of weeks — the shadow-mode rollout pattern lets us compare behavior side-by-side before flipping.
