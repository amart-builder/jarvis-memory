---
name: jarvis-start-guard
description: "Inject a loud bootstrap warning when recent Claude/OpenClaw cross-system context exists, and track whether the next assistant reply acknowledged it."
homepage: https://docs.openclaw.ai/automation/hooks
metadata:
  {
    "openclaw": {
      "emoji": "🚨",
      "events": ["agent:bootstrap", "message:sent"],
      "requires": {}
    }
  }
---

# Jarvis Start Guard

Hardens session starts when recent cross-system context exists.

## What it does

1. Checks for recent non-OpenClaw memories from Jarvis at session bootstrap
2. Prepends a loud warning block into the session bootstrap files when recent context exists
3. Records guard state locally for auditing
4. Tracks whether the next assistant reply appears to acknowledge the handoff

## Notes

- Uses the live Jarvis REST API at `http://localhost:3500`
- Recent means the last 2 hours by default
- State is written to `~/.openclaw/jarvis-start-guard-state.json`
- Supports dry-run verification with `JARVIS_START_GUARD_DRY_RUN=1`
