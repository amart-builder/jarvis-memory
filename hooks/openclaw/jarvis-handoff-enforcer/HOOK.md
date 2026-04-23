---
name: jarvis-handoff-enforcer
description: "Write an automatic Jarvis handoff snapshot before OpenClaw compaction and on session resets, so cross-system continuity does not depend on manual memory writes."
homepage: https://docs.openclaw.ai/automation/hooks
metadata:
  {
    "openclaw": {
      "emoji": "🔗",
      "events": ["session:compact:before", "command:new", "command:reset"],
      "requires": {}
    }
  }
---

# Jarvis Handoff Enforcer

Writes a structured snapshot into jarvis-memory at the safest automatic OpenClaw boundaries we can currently verify:

- `session:compact:before`
- `command:new`
- `command:reset`

## Purpose

This hook hardens OpenClaw ↔ Claude handoffs by:

1. Extracting the recent user/assistant transcript
2. Skipping heartbeat-only noise
3. Inferring the best `group_id` from known project keywords
4. Writing a structured continuity snapshot to Jarvis via `POST /api/v1/add`
5. Deduplicating repeated writes from the same session/trigger

## Notes

- Defaults to `group_id=system` when project inference is weak
- Uses the live Jarvis REST API at `http://localhost:3500`
- Supports dry-run verification with `JARVIS_HOOK_DRY_RUN=1`
