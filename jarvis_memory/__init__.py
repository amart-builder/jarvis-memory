"""jarvis-memory — Graphiti-based memory system for Jarvis agent fleet.

Forked from Graphiti (Apache 2.0) with MemClawz algorithms ported in:
- Composite relevance scoring (semantic × recency × importance × access)
- Memory type classification (heuristic + LLM fallback)
- 8-state lifecycle management with transition validation
- 3-tier compaction engine (session → daily → weekly)

Designed for Claude hooks integration (SessionStart/Stop/PreCompact).
"""

__version__ = "0.1.0"
