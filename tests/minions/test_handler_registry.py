"""Handler registry tests.

Validates registration, lookup, duplicate-name guard, and the
protected-name refusal that guards ``shell``, ``system``, etc.
"""
from __future__ import annotations

import pytest

from jarvis_memory.minions.handlers import (
    HandlerRegistrationError,
    get_handler,
    list_handlers,
    register_handler,
    unregister_handler,
)


def _noop(params):
    return {"ok": True}


class TestRegisterAndGet:
    def test_register_then_get(self):
        register_handler("custom_job", _noop)
        fn = get_handler("custom_job")
        assert fn is _noop

    def test_register_returns_none(self):
        # Arguably returns None; we just assert no exception.
        result = register_handler("another", _noop)
        assert result is None

    def test_list_handlers_sorted(self):
        register_handler("b_job", _noop)
        register_handler("a_job", _noop)
        names = list_handlers()
        assert names == sorted(names)
        assert "a_job" in names
        assert "b_job" in names

    def test_unregister_returns_true_when_removed(self):
        register_handler("temp_job", _noop)
        assert unregister_handler("temp_job") is True
        assert unregister_handler("temp_job") is False


class TestBadInputs:
    def test_empty_name_refused(self):
        with pytest.raises(HandlerRegistrationError):
            register_handler("", _noop)

    def test_whitespace_only_name_refused(self):
        with pytest.raises(HandlerRegistrationError):
            register_handler("   ", _noop)

    def test_non_callable_refused(self):
        with pytest.raises(HandlerRegistrationError):
            register_handler("x", "not-a-callable")  # type: ignore[arg-type]

    def test_get_missing_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_handler("never-registered")


class TestDuplicateName:
    def test_duplicate_refused_by_default(self):
        register_handler("dup_test", _noop)
        with pytest.raises(HandlerRegistrationError, match="already registered"):
            register_handler("dup_test", _noop)

    def test_overwrite_flag_allows_replace(self):
        register_handler("dup_over", _noop)
        register_handler("dup_over", _noop, overwrite=True)
        assert get_handler("dup_over") is _noop


class TestProtectedNames:
    def test_shell_refused_by_default(self):
        with pytest.raises(HandlerRegistrationError, match="protected job name"):
            register_handler("shell", _noop)

    def test_system_refused(self):
        with pytest.raises(HandlerRegistrationError):
            register_handler("system", _noop)

    def test_eval_refused(self):
        with pytest.raises(HandlerRegistrationError):
            register_handler("eval", _noop)

    def test_allow_protected_lets_shell_register(self):
        register_handler("shell", _noop, allow_protected=True)
        assert get_handler("shell") is _noop

    def test_whitespace_bypass_refused(self):
        with pytest.raises(HandlerRegistrationError, match="protected"):
            register_handler("  shell  ", _noop)

    def test_case_bypass_refused(self):
        with pytest.raises(HandlerRegistrationError, match="protected"):
            register_handler("SHELL", _noop)
