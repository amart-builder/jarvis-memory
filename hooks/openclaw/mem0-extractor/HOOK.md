---
name: mem0-extractor
description: "Auto-extract durable facts into mem0 before compaction so nothing is lost when context is summarized away."
homepage: https://docs.openclaw.ai/automation/hooks
metadata:
  {
    "openclaw": {
      "emoji": "🧠",
      "events": ["session:compact:before"],
      "requires": {}
    }
  }
---

# mem0 Extractor

Fires on `session:compact:before` — the moment right before OpenClaw compresses
the session history. Sends the last N messages to the local mem0 API
(`http://localhost:3500`) for LLM-powered fact extraction, deduplication, and
persistent storage in Neo4j + ChromaDB.

## Why This Hook Exists

OpenClaw's LCM compaction summarizes long sessions — but granular facts
(person introductions, decisions, project state) can be lossy after compression.
This hook captures those facts *before* they're summarized, storing them in
mem0 where they're available for semantic search across all future sessions.

## What It Does

1. Takes the last 40 messages from the session history (≈ full recent context)
2. POSTs them to the jarvis-memory `/api/v1/add` endpoint as user `alex`
3. jarvis-memory.s LLM extractor pulls structured facts, deduplicates, and stores in Neo4j + ChromaDB
4. Logs a summary of what was stored

## Configuration

- `MEM0_URL`: mem0 API base URL (default: `http://localhost:3500`)
- `MEM0_USER_ID`: mem0 user namespace (default: `alex`)
- `MEM0_MAX_MESSAGES`: How many recent messages to send (default: `40`)

## Cost

~$0.002/compaction using claude-haiku-4-5. Expected ~5-10 compactions/day = $0.01-0.02/day.
