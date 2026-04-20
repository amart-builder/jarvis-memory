"""Protected-name module tests — whitespace-safe, case-safe, separator-normalized."""
from __future__ import annotations

import pytest

from jarvis_memory.minions.handlers.protected_names import (
    PROTECTED_JOB_NAMES,
    is_protected_job_name,
)


class TestConstantShape:
    def test_protected_is_frozenset(self):
        assert isinstance(PROTECTED_JOB_NAMES, frozenset)

    def test_core_names_present(self):
        assert "shell" in PROTECTED_JOB_NAMES
        assert "system" in PROTECTED_JOB_NAMES
        assert "eval" in PROTECTED_JOB_NAMES


class TestExactMatch:
    def test_exact_match_true(self):
        assert is_protected_job_name("shell") is True

    def test_non_protected_false(self):
        assert is_protected_job_name("my_custom_job") is False

    def test_empty_string_false(self):
        assert is_protected_job_name("") is False

    def test_none_returns_false(self):
        assert is_protected_job_name(None) is False  # type: ignore[arg-type]


class TestWhitespaceBypass:
    def test_leading_whitespace(self):
        assert is_protected_job_name(" shell") is True

    def test_trailing_whitespace(self):
        assert is_protected_job_name("shell ") is True

    def test_both_sides(self):
        assert is_protected_job_name("  shell  ") is True

    def test_tabs(self):
        assert is_protected_job_name("\tshell\t") is True

    def test_newlines(self):
        assert is_protected_job_name("\nshell\n") is True


class TestCaseBypass:
    def test_uppercase(self):
        assert is_protected_job_name("SHELL") is True

    def test_mixed_case(self):
        assert is_protected_job_name("Shell") is True

    def test_camel(self):
        assert is_protected_job_name("ShElL") is True


class TestSeparatorNormalization:
    def test_underscore_matches_hyphen(self):
        # "shell-exec" is protected. "shell_exec" should also match.
        assert is_protected_job_name("shell_exec") is True

    def test_space_matches_hyphen(self):
        assert is_protected_job_name("shell exec") is True

    def test_multiple_separators_normalized(self):
        assert is_protected_job_name("shell--exec") is True
        assert is_protected_job_name("shell  exec") is True
        assert is_protected_job_name("shell_-exec") is True


class TestNonProtectedDespiteShape:
    def test_similar_but_distinct_name_not_protected(self):
        assert is_protected_job_name("shell_wrapper") is False
        assert is_protected_job_name("my-shell-thing") is False
