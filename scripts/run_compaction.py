#!/usr/bin/env python3
"""Jarvis compaction runner — invoked by LaunchAgents on a schedule.

Usage:
    run_compaction.py --tier daily       # runs daily_digest (24h lookback)
    run_compaction.py --tier weekly      # runs weekly_merge (7d lookback)
    run_compaction.py --tier daily --group-id navi   # optional: scope to one project

Exits 0 on success, non-zero on failure so launchd can retry or alert.
Writes structured log to stderr (captured by launchd to ~/Atlas/brain/logs/).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the package is importable when invoked via symlink or absolute path.
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent  # jarvis-memory/
sys.path.insert(0, str(PACKAGE_ROOT))

# Load .env so NEO4J_* and JARVIS_* config are available.
env_file = PACKAGE_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split("#", 1)[0].strip()
        os.environ.setdefault(key.strip(), val)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
# Downgrade the Neo4j driver's cosmetic notifications. On a fresh DB (no
# writes yet for :Page, typed edges, compaction_daily_run property, etc.)
# the server emits WARNING-level "does not exist" hints for every label /
# type / property we legitimately touch. They're informational — the
# queries return correct empty results — but they drown the compaction
# log. Set to DEBUG so `launchd` + developers can still see them with
# `JARVIS_COMPACT_DEBUG=1` (handled below).
if os.environ.get("JARVIS_COMPACT_DEBUG") != "1":
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

log = logging.getLogger("jarvis-compact")


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis compaction runner")
    parser.add_argument(
        "--tier", required=True, choices=["daily", "weekly"], help="Compaction tier"
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help="Optional group_id to scope compaction (default: all groups)",
    )
    args = parser.parse_args()

    try:
        from jarvis_memory.compaction import CompactionEngine
        from jarvis_memory.embeddings import EmbeddingStore
    except ImportError as e:
        log.error("Failed to import jarvis_memory package: %s", e)
        return 2

    # Optional embedding store for semantic dedup
    embedding_store = None
    try:
        embedding_store = EmbeddingStore()
        if not embedding_store.health_check():
            log.warning("ChromaDB not healthy — compaction will fall back to hash-only dedup")
            embedding_store = None
    except Exception as e:
        log.warning("EmbeddingStore init failed (%s) — using hash-only dedup", e)

    engine = CompactionEngine(embedding_store=embedding_store)
    try:
        if args.tier == "daily":
            log.info("Starting daily_digest (group_id=%s)", args.group_id or "<ALL>")
            result = engine.daily_digest(group_id=args.group_id)
            # Run 3 dream-cycle (read-only hygiene): citation audit,
            # orphan report, stale-edge reconciliation. Ran after the
            # daily dedup so a failed dedup still surfaces the cycle
            # report for review. Each phase has a ≤ 2 min budget.
            try:
                log.info("Starting dream-cycle phases")
                dream_report = engine.run_dream_cycle()
                log.info("Dream-cycle result: %s", json.dumps(dream_report, default=str))
                if isinstance(result, dict):
                    result["dream_cycle"] = dream_report
            except Exception as dc_err:  # never fail the daily cron on hygiene errors
                log.warning("dream-cycle failed (non-fatal): %s", dc_err)
                if isinstance(result, dict):
                    result["dream_cycle_error"] = str(dc_err)
        else:
            log.info("Starting weekly_merge (group_id=%s)", args.group_id or "<ALL>")
            result = engine.weekly_merge(group_id=args.group_id)

        log.info("Compaction result: %s", json.dumps(result, default=str))
        # Also emit to stdout so launchd's stdout log captures it for parsing
        print(json.dumps({"tier": args.tier, "group_id": args.group_id, "result": result}, default=str))
        return 0
    except Exception as e:
        log.exception("Compaction failed: %s", e)
        return 1
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
