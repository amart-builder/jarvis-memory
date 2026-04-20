#!/usr/bin/env python3
"""Backfill existing Neo4j memories with v2 metadata + ChromaDB embeddings.

One-time migration script for jarvis-memory v2.

For each memory in Neo4j:
  1. Detect room via keyword scoring
  2. Map hall from memory_type
  3. Set valid_from = created_at (sensible default)
  4. Write room, hall, valid_from properties to Neo4j node
  5. Embed in ChromaDB

Usage:
  python scripts/backfill_v2.py              # Dry run (preview only)
  python scripts/backfill_v2.py --execute    # Actually run the backfill
"""
from __future__ import annotations

import sys
import os

# Add parent dir to path so we can import jarvis_memory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis_memory.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from jarvis_memory.rooms import detect_room, get_hall
from jarvis_memory.embeddings import EmbeddingStore


def main():
    dry_run = "--execute" not in sys.argv

    if dry_run:
        print("=== DRY RUN (use --execute to actually backfill) ===\n")
    else:
        print("=== EXECUTING BACKFILL ===\n")

    # Connect to Neo4j
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Init ChromaDB
    store = EmbeddingStore()
    if not store.health_check():
        print("WARNING: ChromaDB not available. Will only update Neo4j properties.")
        store = None

    # Fetch all active memories
    with driver.session() as db:
        result = db.run("""
            MATCH (n)
            WHERE (n:EntityNode OR n:EpisodicNode OR n:Episode OR n:Entity)
              AND coalesce(n.lifecycle_status, 'active') IN ['active', 'confirmed']
            RETURN n.uuid AS uuid,
                   coalesce(n.content, n.name, n.summary, n.fact, '') AS text,
                   coalesce(n.group_id, 'jarvis-global') AS group_id,
                   coalesce(n.memory_type, n.episode_type, 'fact') AS memory_type,
                   n.created_at AS created_at,
                   n.room AS existing_room
            ORDER BY n.created_at ASC
        """)
        memories = [dict(r) for r in result]

    print(f"Found {len(memories)} active memories to backfill.\n")

    updated = 0
    embedded = 0
    skipped = 0

    for i, mem in enumerate(memories, 1):
        uid = mem["uuid"]
        text = mem["text"]
        group_id = mem["group_id"]
        memory_type = mem["memory_type"]
        created_at = mem["created_at"]

        if not text or len(text.strip()) < 10:
            skipped += 1
            continue

        room = detect_room(text, group_id)
        hall = get_hall(memory_type)

        status = f"[{i}/{len(memories)}] {uid[:8]}... → room={room}, hall={hall}, type={memory_type}"

        if dry_run:
            print(f"  PREVIEW: {status}")
            print(f"           text: {text[:80]}...")
            updated += 1
            continue

        # Update Neo4j node properties
        try:
            with driver.session() as db:
                db.run(
                    """
                    MATCH (n) WHERE n.uuid = $uuid
                    SET n.room = $room,
                        n.hall = $hall,
                        n.valid_from = CASE
                            WHEN n.valid_from IS NULL AND n.created_at IS NOT NULL
                            THEN n.created_at
                            ELSE n.valid_from
                        END
                    """,
                    uuid=uid,
                    room=room,
                    hall=hall,
                )
            updated += 1
        except Exception as e:
            print(f"  ERROR updating Neo4j for {uid}: {e}")
            continue

        # Embed in ChromaDB
        if store:
            metadata = {
                "wing": group_id,
                "room": room,
                "hall": hall,
                "memory_type": memory_type,
            }
            if created_at:
                metadata["created_at"] = str(created_at)

            ok = store.embed_and_store(uid, text, metadata)
            if ok:
                embedded += 1

        print(f"  OK: {status}")

    print(f"\n=== BACKFILL {'PREVIEW' if dry_run else 'COMPLETE'} ===")
    print(f"  Memories found:    {len(memories)}")
    print(f"  Updated (Neo4j):   {updated}")
    print(f"  Embedded (Chroma): {embedded}")
    print(f"  Skipped (too short): {skipped}")

    if dry_run:
        print(f"\nRun with --execute to apply changes.")

    driver.close()


if __name__ == "__main__":
    main()
