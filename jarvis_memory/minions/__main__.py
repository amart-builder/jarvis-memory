"""CLI entry point for the Minions queue.

Commands:
  python -m jarvis_memory.minions worker --queue default --concurrency 1
    Run the worker daemon. Blocks until SIGTERM / Ctrl-C.

  python -m jarvis_memory.minions submit NAME --params '{...}' [--parent-id X]
                                                [--idempotency-key KEY]
                                                [--timeout 60]
    Enqueue a job. Prints the assigned job_id.

  python -m jarvis_memory.minions list [--status S] [--queue Q] [--limit N]
    List jobs.

  python -m jarvis_memory.minions get ID
    Show a job row as JSON.

  python -m jarvis_memory.minions cancel ID [--no-cascade]
    Cancel a job and (by default) its descendants.

  python -m jarvis_memory.minions stall-sweep
    Print any stalled claimed jobs (lease expired past STALL_MULTIPLIER).

All commands accept ``--db-path`` to override the default location
``data/minions.sqlite``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import Any, Optional

from .queue import MinionQueue
from .types import Job


def _format_job(job: Job) -> dict[str, Any]:
    return dataclasses.asdict(job)


def _parse_params(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: --params must be valid JSON: {exc}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(parsed, dict):
        print("error: --params must be a JSON object", file=sys.stderr)
        sys.exit(2)
    return parsed


def _cmd_submit(args: argparse.Namespace) -> int:
    q = MinionQueue(args.db_path)
    try:
        params = _parse_params(args.params)
        job_id = q.submit(
            args.name,
            params,
            queue=args.queue,
            priority=args.priority,
            parent_id=args.parent_id,
            idempotency_key=args.idempotency_key,
            timeout_seconds=args.timeout,
            max_attempts=args.max_attempts,
            trusted=args.trusted,
        )
        print(job_id)
        return 0
    finally:
        q.close()


def _cmd_list(args: argparse.Namespace) -> int:
    q = MinionQueue(args.db_path)
    try:
        jobs = q.list(status=args.status, queue=args.queue, limit=args.limit)
        for job in jobs:
            print(json.dumps({
                "id": job.id,
                "name": job.name,
                "status": job.status,
                "priority": job.priority,
                "attempts": f"{job.attempts}/{job.max_attempts}",
                "created_at": job.created_at,
            }))
        return 0
    finally:
        q.close()


def _cmd_get(args: argparse.Namespace) -> int:
    q = MinionQueue(args.db_path)
    try:
        job = q.get(args.job_id)
        if job is None:
            print(f"error: job {args.job_id!r} not found", file=sys.stderr)
            return 1
        print(json.dumps(_format_job(job), indent=2, default=str))
        return 0
    finally:
        q.close()


def _cmd_cancel(args: argparse.Namespace) -> int:
    q = MinionQueue(args.db_path)
    try:
        n = q.cancel(args.job_id, cascade=not args.no_cascade)
        print(json.dumps({"cancelled": n, "job_id": args.job_id}))
        return 0
    finally:
        q.close()


def _cmd_stall_sweep(args: argparse.Namespace) -> int:
    q = MinionQueue(args.db_path)
    try:
        stalled = q.stall_sweep()
        for job in stalled:
            print(json.dumps({
                "id": job.id,
                "name": job.name,
                "claimed_at": job.claimed_at,
                "worker_id": job.worker_id,
                "timeout_seconds": job.timeout_seconds,
            }))
        return 0
    finally:
        q.close()


def _cmd_worker(args: argparse.Namespace) -> int:
    # Auto-register built-in handlers. Import is side-effectful.
    from .handlers import builtin  # noqa: F401

    # Import shell handler only if the env gate is on.
    if os.environ.get("GBRAIN_ALLOW_SHELL_JOBS") == "1":
        try:
            from .handlers import shell  # noqa: F401
            logging.info("Shell handler registered (GBRAIN_ALLOW_SHELL_JOBS=1)")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Shell handler registration failed: %s", exc)

    from .worker import MinionWorker

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    worker = MinionWorker(
        args.db_path,
        queue=args.queue,
        concurrency=args.concurrency,
    )
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis_memory.minions", description=__doc__.splitlines()[0])
    p.add_argument(
        "--db-path",
        default=os.environ.get("MINIONS_DB_PATH", "data/minions.sqlite"),
        help="Path to the SQLite queue file",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # worker
    wp = sub.add_parser("worker", help="Run the worker daemon")
    wp.add_argument("--queue", default="default")
    wp.add_argument("--concurrency", type=int, default=1)
    wp.set_defaults(func=_cmd_worker)

    # submit
    sp = sub.add_parser("submit", help="Enqueue a job")
    sp.add_argument("name", help="Handler name")
    sp.add_argument("--params", default="{}", help="JSON-encoded params dict")
    sp.add_argument("--queue", default="default")
    sp.add_argument("--priority", type=int, default=0)
    sp.add_argument("--parent-id", default=None)
    sp.add_argument("--idempotency-key", default=None)
    sp.add_argument("--timeout", type=int, default=60, dest="timeout")
    sp.add_argument("--max-attempts", type=int, default=3)
    sp.add_argument("--trusted", action="store_true")
    sp.set_defaults(func=_cmd_submit)

    # list
    lp = sub.add_parser("list", help="List jobs")
    lp.add_argument("--status", default=None)
    lp.add_argument("--queue", default=None)
    lp.add_argument("--limit", type=int, default=50)
    lp.set_defaults(func=_cmd_list)

    # get
    gp = sub.add_parser("get", help="Show a job as JSON")
    gp.add_argument("job_id")
    gp.set_defaults(func=_cmd_get)

    # cancel
    cp = sub.add_parser("cancel", help="Cancel a job (cascades by default)")
    cp.add_argument("job_id")
    cp.add_argument("--no-cascade", action="store_true", help="Do not cancel descendants")
    cp.set_defaults(func=_cmd_cancel)

    # stall-sweep
    ss = sub.add_parser("stall-sweep", help="Print stalled claimed jobs")
    ss.set_defaults(func=_cmd_stall_sweep)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
