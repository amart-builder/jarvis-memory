"""Claude hooks for jarvis-memory — automatic memory injection and saving.

These hooks solve the 50-80% MCP tool invocation reliability problem by
making memory operations happen automatically instead of relying on the
agent to voluntarily call tools.

Three hooks:
  - SessionStart: Auto-inject relevant memory context into the conversation
  - Stop: Auto-save session summary to the knowledge graph
  - PreCompact: Save critical context before context window compaction
"""
