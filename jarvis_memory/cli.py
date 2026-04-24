"""`jarvis` CLI — inspection + ops for Jarvis-Memory.

Designed to answer "did the handoff happen?" without grepping logs, and
give operators a small set of read-mostly commands for common debugging.

Commands:
    jarvis status                         overall health + counts
    jarvis groups                         list every group_id with counts
    jarvis handoff latest --group X       show most recent [HANDOFF] for a group
    jarvis handoff write --group X --task "what" [--next-steps ...]
                                          write a handoff from the CLI
    jarvis wake-up --group X              show what an agent would see on resume
    jarvis sessions --group X [--limit N] list recent sessions for a group

Respects:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD — connects directly, no REST.

Useful pipes:
    jarvis groups --json | jq '.groups[] | select(.episode_count < 5)'
    jarvis handoff latest --group navi --json | jq .content
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

from . import handoff as handoff_module


# ── Output helpers ─────────────────────────────────────────────────────


def _emit(payload: Any, *, as_json: bool, fallback_text: str | None = None) -> None:
    if as_json:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    elif fallback_text is not None:
        print(fallback_text)
    else:
        # Pretty-print dict / list without JSON.
        if isinstance(payload, dict):
            for k, v in payload.items():
                print(f"  {k}: {v}")
        elif isinstance(payload, list):
            for item in payload:
                print(f"  - {item}")
        else:
            print(payload)


def _get_driver():
    """Open a Neo4j driver from env vars. Exits 2 on connection failure."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("ERROR: neo4j Python driver not installed.", file=sys.stderr)
        sys.exit(2)

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "neo4j")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        print(
            f"ERROR: Neo4j not reachable at {uri} ({e.__class__.__name__}: {e}).\n"
            f"Check NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in your environment.",
            file=sys.stderr,
        )
        sys.exit(2)


# ── Command handlers ───────────────────────────────────────────────────


def cmd_status(args) -> int:
    driver = _get_driver()
    try:
        groups = handoff_module.list_groups(driver)
        with driver.session() as db:
            totals = db.run(
                """
                RETURN
                  count { MATCH (e:Episode) RETURN e }  AS episodes,
                  count { MATCH (s:Session) RETURN s }  AS sessions,
                  count { MATCH (e:Episode {memory_type: 'handoff'}) RETURN e } AS handoffs
                """,
            ).single()
        payload = {
            "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            "episodes_total": totals["episodes"],
            "sessions_total": totals["sessions"],
            "handoffs_total": totals["handoffs"],
            "groups": len(groups),
            "top_groups": groups[:5],
        }
    finally:
        driver.close()

    if args.json:
        _emit(payload, as_json=True)
    else:
        print(f"jarvis-memory @ {payload['neo4j_uri']}")
        print(f"  episodes: {payload['episodes_total']:>8,}")
        print(f"  sessions: {payload['sessions_total']:>8,}")
        print(f"  handoffs: {payload['handoffs_total']:>8,}")
        print(f"  groups:   {payload['groups']:>8}")
        if payload["top_groups"]:
            print("\nTop 5 groups by episode count:")
            for g in payload["top_groups"]:
                latest = g["latest_episode_at"] or "(none)"
                print(
                    f"  {g['group_id']:<25}  eps={g['episode_count']:>5}  "
                    f"sess={g['session_count']:>3}  last={latest}"
                )
    return 0


def cmd_groups(args) -> int:
    driver = _get_driver()
    try:
        groups = handoff_module.list_groups(driver)
    finally:
        driver.close()

    if args.json:
        _emit({"groups": groups, "count": len(groups)}, as_json=True)
    elif not groups:
        print("(no groups)")
    else:
        print(f"{'group_id':<30}  {'episodes':>10}  {'sessions':>10}  latest_episode_at")
        print("-" * 80)
        for g in groups:
            latest = g["latest_episode_at"] or "—"
            print(
                f"{g['group_id']:<30}  {g['episode_count']:>10}  "
                f"{g['session_count']:>10}  {latest}"
            )
    return 0


def cmd_handoff_latest(args) -> int:
    driver = _get_driver()
    try:
        result = handoff_module.get_latest_handoff(
            driver, group_id=args.group, max_age_hours=args.max_age_hours,
        )
    except handoff_module.GroupIDRequired as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        driver.close()

    if result is None:
        if args.json:
            _emit({"found": False, "group_id": args.group}, as_json=True)
        else:
            print(f"(no handoff within {args.max_age_hours}h for group_id={args.group!r})")
        return 1  # non-zero so shell pipelines can react

    if args.json:
        _emit(result, as_json=True)
    else:
        print(f"Latest handoff for {result['group_id']}")
        print(f"  created_at: {result['created_at']}")
        print(f"  session:    {result['session_id']}")
        print(f"  device:     {result['device']}")
        print(f"  source:     {result['source']}")
        if result.get("session_key"):
            print(f"  session_key: {result['session_key']}")
        print()
        print(result["content"])
    return 0


def cmd_handoff_write(args) -> int:
    driver = _get_driver()
    try:
        result = handoff_module.save_handoff(
            driver,
            group_id=args.group,
            task=args.task,
            next_steps=args.next_step or [],
            notes=args.notes or "",
            device=args.device or os.environ.get("JARVIS_DEVICE_ID", "cli"),
            idempotency_key=args.idempotency_key,
            session_key=args.session_key,
            source="cli",
        )
    except handoff_module.GroupIDRequired as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        driver.close()

    if args.json:
        _emit(
            {
                "handoff_ready": True,
                "group_id": result.group_id,
                "session_id": result.session_id,
                "snapshot_id": result.snapshot_id,
                "episode_id": result.episode_id,
                "idempotent_hit": result.idempotent_hit,
            },
            as_json=True,
        )
    else:
        marker = "(idempotent hit — no new rows)" if result.idempotent_hit else "(new)"
        print(f"✓ Handoff written {marker}")
        print(f"  group_id:    {result.group_id}")
        print(f"  session:     {result.session_id}")
        print(f"  snapshot:    {result.snapshot_id}")
        print(f"  episode:     {result.episode_id}")
    return 0


def cmd_wake_up(args) -> int:
    try:
        from .embeddings import EmbeddingStore
        from .wake_up import wake_up as do_wake_up
    except ImportError as e:
        print(f"ERROR: import failed — {e}", file=sys.stderr)
        return 2
    driver = _get_driver()
    try:
        store = EmbeddingStore()
        payload = do_wake_up(store, driver, args.group)
    finally:
        driver.close()

    if args.json:
        _emit(payload, as_json=True)
    else:
        # wake_up returns a structured dict; show headline keys.
        print(f"wake_up for {args.group}")
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, (list, dict)):
                    print(f"  {k}: <{type(v).__name__} len={len(v) if hasattr(v, '__len__') else '?'}>")
                else:
                    print(f"  {k}: {v}")
    return 0


def cmd_sessions(args) -> int:
    from .conversation import SessionManager

    driver = _get_driver()
    try:
        sm = SessionManager(driver=driver)
        sessions = sm.list_sessions(args.group, limit=args.limit)
    finally:
        driver.close()

    if args.json:
        _emit({"sessions": sessions, "count": len(sessions)}, as_json=True)
    elif not sessions:
        print(f"(no sessions for group_id={args.group!r})")
    else:
        print(f"{'created_at':<30}  {'device':<15}  {'status':<12}  uuid")
        print("-" * 95)
        for s in sessions:
            print(
                f"{str(s.get('created_at', '')):<30}  "
                f"{str(s.get('device', '')):<15}  "
                f"{str(s.get('status', '')):<12}  "
                f"{s.get('uuid', '')}"
            )
    return 0


# ── Entry point ────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="jarvis",
        description="Jarvis-Memory inspection + ops CLI.",
    )
    p.add_argument("--json", action="store_true", help="Output JSON instead of human text.")

    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # status
    sp = sub.add_parser("status", help="Overall health + counts.")
    sp.set_defaults(func=cmd_status)

    # groups
    sp = sub.add_parser("groups", help="List every group_id with counts.")
    sp.set_defaults(func=cmd_groups)

    # handoff (with subcommands)
    h = sub.add_parser("handoff", help="Handoff operations.")
    hsub = h.add_subparsers(dest="handoff_cmd", required=True, metavar="<subcommand>")

    hl = hsub.add_parser("latest", help="Show most recent [HANDOFF] for a group.")
    hl.add_argument("--group", required=True)
    hl.add_argument("--max-age-hours", type=int, default=72)
    hl.set_defaults(func=cmd_handoff_latest)

    hw = hsub.add_parser("write", help="Write a handoff from the CLI.")
    hw.add_argument("--group", required=True)
    hw.add_argument("--task", required=True)
    hw.add_argument("--next-step", action="append", help="Repeat for multiple next-steps.")
    hw.add_argument("--notes", default="")
    hw.add_argument("--device", default=None)
    hw.add_argument("--idempotency-key", default=None)
    hw.add_argument("--session-key", default=None)
    hw.set_defaults(func=cmd_handoff_write)

    # wake-up
    sp = sub.add_parser("wake-up", help="Show what an agent would see on resume.")
    sp.add_argument("--group", required=True)
    sp.set_defaults(func=cmd_wake_up)

    # sessions
    sp = sub.add_parser("sessions", help="List recent sessions for a group.")
    sp.add_argument("--group", required=True)
    sp.add_argument("--limit", type=int, default=10)
    sp.set_defaults(func=cmd_sessions)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
