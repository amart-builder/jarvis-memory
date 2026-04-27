"""Unit tests for scripts/run_parallel_judge.py.

Covers prompt-template selection (must match the official judge byte-for-
byte for grading parity) and the per-row failure path. The
ThreadPoolExecutor dispatch and OpenAI calls are validated by an
end-to-end live run against the targeted-validation harness.
"""
from __future__ import annotations

import pytest


# ── build_anscheck_prompt: must match official judge templates ──────


def test_build_anscheck_prompt_picks_temporal_template():
    from scripts.run_parallel_judge import build_anscheck_prompt
    out = build_anscheck_prompt(
        qtype="temporal-reasoning",
        question="Q",
        answer="A",
        response="R",
        abstention=False,
    )
    # Off-by-one tolerance is the temporal template's distinguishing rule.
    assert "off-by-one errors" in out
    assert "Question: Q" in out
    assert "Correct Answer: A" in out
    assert "Model Response: R" in out


def test_build_anscheck_prompt_ku_template_allows_previous_info():
    from scripts.run_parallel_judge import build_anscheck_prompt
    out = build_anscheck_prompt(
        qtype="knowledge-update",
        question="Q", answer="A", response="R", abstention=False,
    )
    assert "previous information along with an updated answer" in out


def test_build_anscheck_prompt_ss_user_template_subset_rule():
    from scripts.run_parallel_judge import build_anscheck_prompt
    out = build_anscheck_prompt(
        qtype="single-session-user",
        question="Q", answer="A", response="R", abstention=False,
    )
    assert "subset of the information required" in out
    assert "off-by-one errors" not in out  # SS template lacks temporal rule


def test_build_anscheck_prompt_ms_uses_same_template_as_ss_user():
    """The official judge maps multi-session AND SS-assistant onto the
    same template as SS-user (line 27 of evaluate_qa.py). Mirror that."""
    from scripts.run_parallel_judge import build_anscheck_prompt
    a = build_anscheck_prompt(qtype="single-session-user", question="Q", answer="A", response="R", abstention=False)
    b = build_anscheck_prompt(qtype="multi-session", question="Q", answer="A", response="R", abstention=False)
    c = build_anscheck_prompt(qtype="single-session-assistant", question="Q", answer="A", response="R", abstention=False)
    assert a == b == c


def test_build_anscheck_prompt_preference_uses_rubric_wording():
    from scripts.run_parallel_judge import build_anscheck_prompt
    out = build_anscheck_prompt(
        qtype="single-session-preference",
        question="Q", answer="A", response="R", abstention=False,
    )
    assert "Rubric:" in out
    assert "personal information correctly" in out


def test_build_anscheck_prompt_abstention_uses_unanswerable_template():
    """For ``_abs`` qids the judge asks 'did model correctly identify
    the question as unanswerable' instead of 'is the answer correct'."""
    from scripts.run_parallel_judge import build_anscheck_prompt
    out = build_anscheck_prompt(
        qtype="single-session-user",  # qtype ignored when abstention=True
        question="Q", answer="explanation", response="R", abstention=True,
    )
    assert "unanswerable" in out
    assert "Explanation:" in out
    # NOT the "is the model response correct" framing
    assert "Is the model response correct" not in out


def test_build_anscheck_prompt_unknown_qtype_raises():
    from scripts.run_parallel_judge import build_anscheck_prompt
    with pytest.raises(NotImplementedError):
        build_anscheck_prompt(
            qtype="not-a-real-category",
            question="Q", answer="A", response="R", abstention=False,
        )


# ── grade_entry: per-row error handling ────────────────────────────


def test_grade_entry_unknown_qid_returns_label_false_with_error_field():
    """An entry whose qid isn't in the reference data fails open with an
    autoeval_error field — does NOT raise, does NOT crash the run."""
    from scripts.run_parallel_judge import grade_entry
    entry = {"question_id": "ghost", "hypothesis": "x"}
    out = grade_entry(
        client=None,  # never reached because of the early-return
        model="gpt-4o-2024-08-06",
        entry=entry,
        qid_to_qtype={"q1": "single-session-user"},
        qid_to_qdata={"q1": {"question": "?", "answer": "!"}},
    )
    assert out["autoeval_label"] == {"model": "gpt-4o-2024-08-06", "label": False}
    assert "autoeval_error" in out
    assert "missing_or_unknown_qid" in out["autoeval_error"]


def test_grade_entry_yes_response_labels_true(monkeypatch):
    """Happy path: a "Yes" reply from the judge → label True."""
    from scripts import run_parallel_judge as judge

    monkeypatch.setattr(judge, "_judge_one", lambda *a, **kw: "Yes")

    entry = {"question_id": "q1", "hypothesis": "answer"}
    out = judge.grade_entry(
        client=None, model="gpt-4o-2024-08-06", entry=entry,
        qid_to_qtype={"q1": "single-session-user"},
        qid_to_qdata={"q1": {"question": "Q?", "answer": "A!"}},
    )
    assert out["autoeval_label"] == {"model": "gpt-4o-2024-08-06", "label": True}


def test_grade_entry_no_response_labels_false(monkeypatch):
    from scripts import run_parallel_judge as judge

    monkeypatch.setattr(judge, "_judge_one", lambda *a, **kw: "No")

    entry = {"question_id": "q1", "hypothesis": "wrong answer"}
    out = judge.grade_entry(
        client=None, model="gpt-4o-2024-08-06", entry=entry,
        qid_to_qtype={"q1": "multi-session"},
        qid_to_qdata={"q1": {"question": "Q?", "answer": "A!"}},
    )
    assert out["autoeval_label"]["label"] is False
    assert "autoeval_error" not in out  # not an error — judge said no


def test_grade_entry_judge_call_failure_fails_open(monkeypatch):
    """If the OpenAI call raises, the row gets label=False + autoeval_error."""
    from scripts import run_parallel_judge as judge

    def boom(*a, **kw):
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(judge, "_judge_one", boom)

    entry = {"question_id": "q1", "hypothesis": "answer"}
    out = judge.grade_entry(
        client=None, model="gpt-4o-2024-08-06", entry=entry,
        qid_to_qtype={"q1": "single-session-user"},
        qid_to_qdata={"q1": {"question": "Q?", "answer": "A!"}},
    )
    assert out["autoeval_label"]["label"] is False
    assert "autoeval_error" in out
    assert "RuntimeError" in out["autoeval_error"]
    assert "simulated outage" in out["autoeval_error"]


def test_grade_entry_handles_abstention_qid_pattern():
    """qids ending in `_abs` route through the abstention prompt."""
    from scripts import run_parallel_judge as judge

    captured: dict = {}

    def spy_judge(client, model, prompt):
        captured["prompt"] = prompt
        return "yes"

    import unittest.mock as m
    with m.patch.object(judge, "_judge_one", spy_judge):
        out = judge.grade_entry(
            client=None, model="gpt-4o-2024-08-06",
            entry={"question_id": "q5_abs", "hypothesis": "I don't have enough info"},
            qid_to_qtype={"q5_abs": "single-session-user"},
            qid_to_qdata={"q5_abs": {"question": "Q?", "answer": "explanation"}},
        )
    assert "unanswerable" in captured["prompt"]
    assert out["autoeval_label"]["label"] is True
