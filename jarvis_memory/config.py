"""jarvis-memory configuration.

All settings are loaded from environment variables with sensible defaults.
"""
from __future__ import annotations

import os

# --- Neo4j / Graphiti ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# --- Anthropic (for LLM classifier + enrichment) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLASSIFIER_MODEL = os.getenv("JARVIS_CLASSIFIER_MODEL", "claude-sonnet-4-20250514")

# --- Composite Scoring Weights ---
W_SEMANTIC = float(os.getenv("JARVIS_W_SEMANTIC", "0.50"))
W_RECENCY = float(os.getenv("JARVIS_W_RECENCY", "0.30"))
W_IMPORTANCE = float(os.getenv("JARVIS_W_IMPORTANCE", "0.20"))
HALF_LIFE_DAYS = float(os.getenv("JARVIS_HALF_LIFE_DAYS", "90.0"))

# --- Hybrid Search ---
HYBRID_ALPHA = float(os.getenv("JARVIS_HYBRID_ALPHA", "0.7"))  # 70% semantic, 30% keyword

# --- Compaction ---
COMPACTION_DEDUP_DAILY = float(os.getenv("JARVIS_DEDUP_DAILY", "0.88"))
COMPACTION_DEDUP_WEEKLY = float(os.getenv("JARVIS_DEDUP_WEEKLY", "0.92"))
COMPACTION_SESSION_MAX = int(os.getenv("JARVIS_SESSION_MAX_MEMORIES", "50"))

# --- Lifecycle ---
STALE_THRESHOLD_DAYS = int(os.getenv("JARVIS_STALE_THRESHOLD_DAYS", "30"))

# --- API Server ---
API_HOST = os.getenv("JARVIS_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("JARVIS_API_PORT", "3500"))

# --- Group IDs (project isolation) ---
DEFAULT_GROUP_ID = os.getenv("JARVIS_GROUP_ID", "jarvis-global")

# --- Device Identity ---
DEVICE_ID = os.getenv("JARVIS_DEVICE_ID", "unknown")  # "macbook-pro" or "mac-mini"

# --- Conversation Persistence ---
EPISODE_AUTO_RECORD = os.getenv("JARVIS_EPISODE_AUTO_RECORD", "true").lower() == "true"
EPISODE_MIN_LENGTH = int(os.getenv("JARVIS_EPISODE_MIN_LENGTH", "50"))
MAX_EPISODES_PER_SESSION = int(os.getenv("JARVIS_MAX_EPISODES_PER_SESSION", "100"))
SESSION_CHAIN_DEPTH = int(os.getenv("JARVIS_SESSION_CHAIN_DEPTH", "3"))
SNAPSHOT_MAX_SIZE = int(os.getenv("JARVIS_SNAPSHOT_MAX_SIZE", "5000"))

# --- ChromaDB (semantic search sidecar) ---
CHROMADB_PATH = os.getenv("JARVIS_CHROMADB_PATH", os.path.expanduser("~/Atlas/jarvis-memory/chromadb"))
CHROMADB_COLLECTION = os.getenv("JARVIS_CHROMADB_COLLECTION", "jarvis_memories")
EMBEDDING_MODEL = os.getenv("JARVIS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Token Budgets (wake_up) ---
WAKE_UP_LAYER1_MAX_ITEMS = int(os.getenv("JARVIS_WAKE_UP_MAX_ITEMS", "15"))
WAKE_UP_LAYER1_MAX_TOKENS = int(os.getenv("JARVIS_WAKE_UP_MAX_TOKENS", "500"))

# --- Room Detection ---
ROOM_AUTO_DETECT = os.getenv("JARVIS_ROOM_AUTO_DETECT", "true").lower() == "true"
