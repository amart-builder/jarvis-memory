# Handoff Contract

> **Status:** normative — v1.1.0 and later.
> **Scope:** what every agent, hook, and script MUST do when reading from or writing to Jarvis-Memory so that cross-surface handoffs work reliably.

---

## Why this document exists

The #1 reliability bug in the pre-v1.1 Jarvis-Memory wasn't "no memory" — it was **memory going to the wrong place.** A decision made on a MacBook session could silently land in `group_id="system"` instead of `group_id="navi"`, and the next session (OpenClaw on the Mini) would never find it. That's worse than forgetting because it creates false confidence.

This contract closes that gap by making the rules explicit and enforcing them at the API boundary.

---

## The seven rules

### 1. `group_id` is required on every write.

```
POST /api/v2/save_episode      → 422 if group_id missing
POST /api/v2/session/handoff   → 422 if group_id missing
POST /api/v2/session/save_state → 422 if group_id missing
POST /api/v1/add               → top-level group_id preferred;
                                  metadata.group_id fallback;
                                  'system' with WARN if both absent
MCP session_handoff            → {error: "group_id is required"} if missing
MCP save_state                  → uses DEFAULT_GROUP_ID only if caller explicitly omitted
```

Empty strings and whitespace-only values are rejected as if missing.

### 2. Handoff writes are atomic: snapshot + Episode together.

A call to `/api/v2/session/handoff` (or MCP `session_handoff`) always writes:

- A `Snapshot` node linked to the `Session` via `[:HAS_SNAPSHOT]`.
- A retrievable `Episode` node with `memory_type='handoff'`, linked to the session via `[:PRODUCED_HANDOFF]`.

Both live in Neo4j and are findable together. If either write fails, the call surfaces the error rather than half-writing.

### 3. `latest_handoff(group_id)` is the canonical "pick up where we left off" read path.

```
GET  /api/v2/handoff/latest?group_id=X[&max_age_hours=72]
MCP  latest_handoff(group_id="X", max_age_hours=72)
```

Returns the most recent `Episode` with `memory_type='handoff'` for the given group, within the age window. Agents should call this at session start before diving into new work.

### 4. Idempotency keys prevent duplicate handoffs.

```
{"task": "...", "group_id": "X", "idempotency_key": "my-hook-abc123", ...}
```

If a caller supplies `idempotency_key` and a handoff with the same key exists in the last hour **for the same `group_id`**, the new write is a no-op and the response has `idempotent_hit=true` with the *existing* `snapshot_id` + `episode_id`. This means hooks can fire twice, retries can be safe, and nothing corrupts the timeline.

> **Scoping note:** dedupe is by `group_id`, not `session_id`. Each successful handoff ends its session, so a retry resolves to a *new* session before the check runs — session-scoped dedupe would miss prior hits. `group_id` is caller-stable across retries, so it's the right key.

**Recommendation for hooks:** derive the key from something deterministic per logical event. Good: `f"precompact-{session_id}-{datetime.now():%Y%m%d%H}"`. Bad: `uuid.uuid4()` (every retry becomes a new handoff).

### 5. `session_key` is stored on the Episode for cross-surface correlation.

Optional but recommended. A hook fires on a Claude Code session (`session_id=abc`); an OpenClaw hook fires on the next-day pickup (`session_id=def`). If both pass the same `session_key` (e.g. the user's project + day combo), a later analysis can correlate "this thread of work happened in these sessions across these machines."

### 6. Every write is tagged with its source.

Implicit for REST/MCP (the middleware tags requests as `source="rest"` or `source="mcp"`), explicit for hooks and scripts via the `source` parameter. When you see a memory and want to know where it came from, look at the `source` field on the Episode.

Recommended values: `rest`, `mcp`, `cli`, `hook`, `cron`, `openclaw-bridge`, and your hook-specific names (e.g. `precompact-hook`).

### 7. `system` is for system-level memory, not a fallback.

Prior behavior: "no group_id resolved → write to `system`." That's what created the pollution bug. New rule: **never pick `system` implicitly**. If the resolver can't determine the right group_id, the caller should surface the ambiguity — not silently bucket everything into `system`.

Legitimate `system` writes: Atlas infrastructure changes, cross-project protocol updates, MEMORY_PROTOCOL edits. Illegitimate: "the agent was working on something but I couldn't tell what project."

---

## Error taxonomy

| Condition | REST response | MCP response |
|---|---|---|
| Missing required field (Pydantic) | `422` with field details | tool-level error |
| Empty `group_id` after validation | `400 {"detail": "group_id is required ..."}` | `{"error": "..."}` |
| No session exists and `create_session_if_missing=False` | `500` with ValueError | tool error |
| Neo4j unreachable | `500` | tool error |
| Bearer token required but missing | `401 {"detail": "Missing bearer token..."}` | (n/a — stdio transport) |
| Bearer token required but wrong | `401 {"detail": "Invalid bearer token."}` | (n/a) |
| `latest_handoff` with no match in window | `404` | `{"found": false}` |

---

## Bearer auth (v1.1+)

REST API has optional bearer-token auth, designed to fix the ship-gate bug where a non-loopback bind was fully writable by anyone who could reach the port.

**Rules:**
- If `JARVIS_API_BEARER_TOKEN` is set → every non-exempt request needs `Authorization: Bearer <token>`.
- If unset and bound to `127.0.0.1`/`localhost`/`::1` → allow-all (normal local dev).
- If unset and bound to anything else → allow-all **with a startup warning logged**. `JARVIS_REQUIRE_AUTH=1` flips this to a startup refusal for strict deployments.
- Exempt paths: `/health`, `/docs`, `/openapi.json`, `/redoc`.

**Rotate a token:** set the new value in `.env`, restart the API, update every client. No graceful-rotation dance; each client either has the right token or gets 401.

---

## Canonical `group_id` recommendations

For users who are adopting the contract in a fresh project:

- **One group_id per project.** Use short, kebab-case slugs: `navi`, `atlas-system`, `foundry`, `hello-world`.
- **Bots and shared services get their own:** `gravity-bot`, `jarvis`, `sentinel-ci`.
- **Only use `system`** for genuinely system-level memory (protocol docs, cross-project rules). Not for "I don't know what project."
- **Don't use the hostname** as a group_id — cross-device continuity is the point, and hostnames break it.

For Alex's specific layout, see `brain/MEMORY_PROTOCOL.md` in the Atlas workspace.

---

## Operator checklist — "did the handoff happen?"

```bash
# The one-shot answer:
jarvis handoff latest --group <your-group-id>

# Output: "Latest handoff for <group> / created_at: ... / content: ..."
# Exit 0 if found, 1 if no handoff within 72h.

# Broader check:
jarvis status                         # totals across all groups
jarvis groups                         # episodes + sessions per group
jarvis sessions --group <your-group>  # recent sessions for a project
```

When something went sideways:

```bash
# Is my group polluting 'system'?
jarvis groups --json | jq '.groups[] | select(.group_id == "system")'

# Was my handoff written with the idempotency key?
jarvis handoff latest --group navi --json | jq .  # look for snapshot_id, episode_id
```

---

## Migration notes for existing integrations

### Legacy hooks writing to `/api/v1/add`

- **Before:** hooks POSTed to `/api/v1/add` with `group_id` buried in `metadata.group_id`. Missing `metadata.group_id` silently dropped writes into `system`.
- **After v1.1:** top-level `group_id` is preferred; `metadata.group_id` still works as fallback; missing both writes to `system` with a WARN log.
- **Required action:** update hooks to pass `group_id` at top level. Existing behavior keeps working, but the WARN log should surface any remaining drift.

### The pre-v1.1 REST `/api/v2/session/save_state` and `/session/handoff` were broken.

They called a non-existent `SessionManager.save_state()` method and would AttributeError at runtime. Most callers used MCP or direct Cypher instead. **v1.1 fixes this** and routes both through the `jarvis_memory.handoff` module — writes are correct, idempotent, and create the expected snapshot + Episode pair.

### MCP `session_handoff` always wrote to `DEFAULT_GROUP_ID` regardless of request.

Fixed in v1.1. Callers must now pass `group_id` explicitly.

### MCP tool count changed from 27 → 29.

Added: `latest_handoff`, `list_groups`. The parity test (`tests/test_mcp_parity.py`) was updated accordingly.

---

## Related docs

- [`CLIENT_INSTALL.md`](CLIENT_INSTALL.md) — installing Jarvis-Memory end-to-end.
- [`README.md`](README.md) — architecture overview, operations catalog.
- [`hooks/openclaw/README.md`](hooks/openclaw/README.md) — OpenClaw hook surface.
- `brain/projects/atlas-system/plans/2026-04-23-jarvis-handoff-contract-implementation-plan.md` — the Codex-proposed plan that this contract implements (the first slice; the bridge consolidation is a follow-up).
