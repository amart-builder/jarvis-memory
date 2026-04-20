"""Minions handler registry.

Maps job names to callables. ``register_handler`` refuses protected names
unless ``allow_protected=True`` is passed explicitly — this guards
against a handler module registering itself as ``shell`` at import time.

A handler is any callable with the signature ``(params: dict) -> dict``.
Workers pass in the job's ``params`` dict and persist whatever dict the
handler returns as the job's ``result``.

Handlers can also accept a ``job`` keyword argument if they need the full
``Job`` context (e.g., for ``parent_id`` lookups). The worker inspects
``inspect.signature`` to decide which form to call.

This module intentionally does NOT import any handler modules at load
time — the built-in echo / compact_* handlers live in ``builtin.py`` and
register themselves on import. The shell handler lives in ``shell.py``
and is gated; callers must import it explicitly.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from .protected_names import PROTECTED_JOB_NAMES, is_protected_job_name

logger = logging.getLogger(__name__)

HandlerFn = Callable[..., dict]


# Registry is module-global. Guarded by a lock so register/unregister/get are
# safe across the worker thread, lease-renewal thread, and any test thread.
_lock = threading.Lock()
_registry: dict[str, HandlerFn] = {}


class HandlerRegistrationError(RuntimeError):
    """Raised when a handler cannot be registered (protected name, duplicate, etc.)."""


def register_handler(
    name: str,
    fn: HandlerFn,
    *,
    allow_protected: bool = False,
    overwrite: bool = False,
) -> None:
    """Register a job handler under ``name``.

    Raises ``HandlerRegistrationError`` if:
      - ``name`` is empty or non-string.
      - ``fn`` is not callable.
      - ``name`` is a protected name and ``allow_protected=False``.
      - ``name`` is already registered and ``overwrite=False``.

    Callers that need to register a protected-name handler (e.g. the shell
    handler) must pass ``allow_protected=True`` — making the intent explicit
    in source.
    """
    if not isinstance(name, str) or not name.strip():
        raise HandlerRegistrationError("handler name must be a non-empty string")
    if not callable(fn):
        raise HandlerRegistrationError(f"handler {name!r} must be callable")

    normalized_key = name.strip()
    if is_protected_job_name(normalized_key) and not allow_protected:
        raise HandlerRegistrationError(
            f"{normalized_key!r} is a protected job name "
            f"(set allow_protected=True to register deliberately). "
            f"Protected set: {sorted(PROTECTED_JOB_NAMES)}"
        )

    with _lock:
        if normalized_key in _registry and not overwrite:
            raise HandlerRegistrationError(
                f"handler {normalized_key!r} already registered "
                "(pass overwrite=True to replace)"
            )
        _registry[normalized_key] = fn
    logger.debug("Registered handler %r", normalized_key)


def unregister_handler(name: str) -> bool:
    """Remove a handler from the registry. Returns ``True`` if removed."""
    with _lock:
        return _registry.pop(name, None) is not None


def get_handler(name: str) -> HandlerFn:
    """Look up a handler by exact name. Raises ``KeyError`` if absent.

    Exact match only — ``name`` normalization is a registration-time concern;
    workers must pass the exact name the job was submitted with.
    """
    with _lock:
        try:
            return _registry[name]
        except KeyError as exc:
            raise KeyError(
                f"no handler registered for job name {name!r}. "
                f"Registered: {sorted(_registry.keys())}"
            ) from exc


def list_handlers() -> list[str]:
    """Return a sorted list of registered handler names."""
    with _lock:
        return sorted(_registry.keys())


def _clear_registry_for_tests() -> None:
    """Test-only helper: empty the registry. Not part of the public API."""
    with _lock:
        _registry.clear()


__all__ = [
    "HandlerFn",
    "HandlerRegistrationError",
    "PROTECTED_JOB_NAMES",
    "is_protected_job_name",
    "register_handler",
    "unregister_handler",
    "get_handler",
    "list_handlers",
]
