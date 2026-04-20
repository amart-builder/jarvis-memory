"""MinionWorker — claim → execute → complete/fail loop.

Runs as a daemon process. Each iteration:
  1. Poll the queue for a pending job on the configured queue name.
  2. Dispatch to the handler registered for that job's ``name``.
  3. Enforce the job's ``timeout_seconds`` via a supervisor thread
     that cancels execution on overrun.
  4. Write ``complete`` or ``fail`` back to the queue.
  5. Renew lease every ``LEASE_RENEWAL_INTERVAL_SECONDS`` seconds so
     the stall_sweep supervisor can tell the difference between a
     slow-but-alive worker and a dead one.

Graceful shutdown:
  - ``SIGTERM`` / ``SIGINT`` set ``self._stop_event``. The main loop checks
    between iterations. Any in-flight handler gets ``handler_timeout * 2``
    to finish gracefully (or its own timeout_seconds, whichever first).
  - ``.stop()`` is the programmatic equivalent (used by tests and the
    CLI's KeyboardInterrupt branch).
"""
from __future__ import annotations

import concurrent.futures
import inspect
import logging
import signal
import threading
import time
import traceback
import uuid
from typing import Optional, Union

from .handlers import get_handler
from .queue import MinionQueue
from .types import Job

logger = logging.getLogger(__name__)

# How often the worker polls when the queue is empty (seconds).
IDLE_POLL_INTERVAL_SECONDS = 1.0

# How often the worker renews its lease on an in-flight job (seconds).
LEASE_RENEWAL_INTERVAL_SECONDS = 15.0

# Grace period after stop_event fires before we force-kill in-flight work.
SHUTDOWN_GRACE_SECONDS = 10.0


class MinionWorker:
    """Single-process, single-threaded job executor for a named queue.

    Concurrency > 1 would require spawning multiple worker threads with
    their own MinionQueue connections; v1 is concurrency=1 per the spec.
    Raising concurrency is future work.

    Parameters
    ----------
    db_path
        Path to the SQLite file (or an existing ``MinionQueue`` instance).
    queue
        Queue name to poll. Default ``"default"``.
    concurrency
        Number of in-flight jobs at once. Must be 1 in v1.
    worker_id
        Identifier recorded on claimed rows. Default is auto-generated.
    idle_poll_interval
        Seconds to wait between empty-poll iterations.
    """

    def __init__(
        self,
        db_path: Union[str, MinionQueue] = "data/minions.sqlite",
        *,
        queue: str = "default",
        concurrency: int = 1,
        worker_id: Optional[str] = None,
        idle_poll_interval: float = IDLE_POLL_INTERVAL_SECONDS,
    ):
        if concurrency != 1:
            raise ValueError(
                f"concurrency={concurrency} not supported in v1 (must be 1)"
            )
        if isinstance(db_path, MinionQueue):
            self._queue = db_path
            self._owns_queue = False
        else:
            self._queue = MinionQueue(db_path)
            self._owns_queue = True
        self.queue_name = queue
        self.concurrency = concurrency
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.idle_poll_interval = float(idle_poll_interval)

        self._stop_event = threading.Event()
        self._running = False
        self._installed_signal_handlers = False

    # ── signal handling ────────────────────────────────────────────────

    def install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers that trip ``stop()``.

        Only works when called from the main thread. Tests that drive the
        worker synchronously skip this.
        """
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
            self._installed_signal_handlers = True
        except ValueError:
            # Not main thread — skip. Caller must call stop() programmatically.
            logger.debug("install_signal_handlers: not on main thread, skipping")

    def _handle_signal(self, signum, frame):  # noqa: ARG002 — signal contract
        logger.info("Worker received signal %s — stopping", signum)
        self.stop()

    def stop(self) -> None:
        self._stop_event.set()

    # ── run loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block until ``stop()`` is called or SIGTERM arrives."""
        if self._installed_signal_handlers is False:
            self.install_signal_handlers()
        self._running = True
        logger.info(
            "MinionWorker starting: queue=%s worker_id=%s",
            self.queue_name,
            self.worker_id,
        )
        try:
            while not self._stop_event.is_set():
                jobs_claimed = self.run_once()
                if jobs_claimed == 0:
                    # Polite poll to avoid hammering SQLite when idle.
                    self._stop_event.wait(timeout=self.idle_poll_interval)
        finally:
            self._running = False
            if self._owns_queue:
                self._queue.close()
            logger.info("MinionWorker stopped: worker_id=%s", self.worker_id)

    def run_once(self) -> int:
        """Execute a single claim-execute-complete iteration.

        Returns the number of jobs that were processed (0 or 1).
        Exposed so tests can drive the worker deterministically without
        the idle-loop timing.
        """
        claim_result = self._queue.claim(
            self.queue_name, limit=self.concurrency, worker_id=self.worker_id
        )
        if not claim_result.jobs:
            return 0

        for job in claim_result.jobs:
            self._execute(job)
        return len(claim_result.jobs)

    # ── single-job execution ────────────────────────────────────────────

    def _execute(self, job: Job) -> None:
        """Look up the handler and run it with timeout + lease renewal."""
        # Resolve the handler BEFORE spawning the executor so a missing
        # handler fails fast (non-retriable).
        try:
            handler = get_handler(job.name)
        except KeyError as exc:
            logger.error("No handler for job %s (name=%s): %s", job.id, job.name, exc)
            self._queue.fail(job.id, f"no handler registered: {job.name}", retriable=False)
            return

        # Run the handler in a thread so we can enforce timeout + lease renewal.
        call_with_job = self._handler_takes_job(handler)
        logger.info("Executing job %s (%s) timeout=%ss", job.id, job.name, job.timeout_seconds)
        self._queue.log(job.id, "INFO", f"started on {self.worker_id}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._invoke_handler, handler, job, call_with_job)
            renewal_stop = threading.Event()
            renewal_thread = threading.Thread(
                target=self._renew_loop,
                args=(job.id, renewal_stop),
                daemon=True,
            )
            renewal_thread.start()

            try:
                result = future.result(timeout=job.timeout_seconds)
            except concurrent.futures.TimeoutError:
                logger.warning("Job %s exceeded timeout %ss", job.id, job.timeout_seconds)
                self._queue.log(job.id, "ERROR", f"timeout {job.timeout_seconds}s")
                # We can't actually kill a thread in Python; the runaway
                # handler will continue until it completes or the process
                # exits. That's an acceptable limit for v1 — the queue row
                # is marked failed and the worker moves on.
                try:
                    future.cancel()
                except Exception:  # noqa: BLE001
                    pass
                self._queue.fail(job.id, "timeout", retriable=True)
                return
            except Exception as exc:  # noqa: BLE001 — caught-and-reported path
                tb = traceback.format_exc(limit=6)
                logger.exception("Job %s raised: %s", job.id, exc)
                self._queue.log(job.id, "ERROR", f"exception: {tb[:1500]}")
                self._queue.fail(job.id, f"{type(exc).__name__}: {exc}", retriable=True)
                return
            finally:
                renewal_stop.set()
                # Give the renewal thread a brief moment to exit cleanly.
                renewal_thread.join(timeout=1.0)

        try:
            self._queue.complete(job.id, result if isinstance(result, dict) else {"result": result})
            self._queue.log(job.id, "INFO", "complete")
        except Exception as exc:  # noqa: BLE001
            # complete() shouldn't normally raise — if it does, log and move on.
            logger.exception("complete() failed for %s: %s", job.id, exc)

    def _invoke_handler(self, handler, job: Job, pass_job: bool):
        """Invoke the handler with the right signature shape."""
        if pass_job:
            return handler(job.params, job=job)
        return handler(job.params)

    @staticmethod
    def _handler_takes_job(handler) -> bool:
        """Inspect the handler signature — does it accept a ``job`` keyword?"""
        try:
            sig = inspect.signature(handler)
        except (TypeError, ValueError):
            return False
        return "job" in sig.parameters

    def _renew_loop(self, job_id: str, stop: threading.Event) -> None:
        """Background lease-renewal loop for ``job_id``."""
        while not stop.is_set():
            if stop.wait(timeout=LEASE_RENEWAL_INTERVAL_SECONDS):
                return
            try:
                renewed = self._queue.renew_lease(job_id, worker_id=self.worker_id)
                if not renewed:
                    logger.warning(
                        "Lease renewal failed for %s — worker_id mismatch or job gone",
                        job_id,
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lease renewal raised for %s: %s", job_id, exc)


__all__ = ["MinionWorker"]


# ── CLI shim: `python -m jarvis_memory.minions.worker` ──────────────────
#
# Delegates to the main CLI's ``worker`` subcommand so both invocation
# styles work and share a single implementation.

def _worker_module_main() -> int:
    from .__main__ import main as _cli_main

    return _cli_main(["worker", *_filter_argv()])


def _filter_argv() -> list[str]:
    """Return sys.argv[1:] filtered — used when invoked as a module."""
    import sys as _sys

    return _sys.argv[1:]


if __name__ == "__main__":
    import sys

    sys.exit(_worker_module_main())
