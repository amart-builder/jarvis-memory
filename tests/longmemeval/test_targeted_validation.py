"""Unit tests for scripts/run_targeted_validation.py.

Covers the pure helpers — adapter+judge subprocess invocations are
validated end-to-end by the live targeted run, not unit tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── parse_baseline_eval ───────────────────────────────────────────────


def _write_eval(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_parse_baseline_eval_separates_right_and_wrong(tmp_path):
    from scripts.run_targeted_validation import parse_baseline_eval
    p = tmp_path / "eval.jsonl"
    _write_eval(p, [
        {"question_id": "q1", "predicted_category": "multi-session",
         "autoeval_label": {"label": True}},
        {"question_id": "q2", "predicted_category": "multi-session",
         "autoeval_label": {"label": False}},
        {"question_id": "q3", "predicted_category": "temporal-reasoning",
         "autoeval_label": {"label": False}},
    ])
    wrongs, rights = parse_baseline_eval(p)
    assert wrongs == {"q2": "multi-session", "q3": "temporal-reasoning"}
    assert rights == {"q1": "multi-session"}


def test_parse_baseline_eval_filters_by_category(tmp_path):
    from scripts.run_targeted_validation import parse_baseline_eval
    p = tmp_path / "eval.jsonl"
    _write_eval(p, [
        {"question_id": "q1", "predicted_category": "multi-session",
         "autoeval_label": {"label": False}},
        {"question_id": "q2", "predicted_category": "temporal-reasoning",
         "autoeval_label": {"label": False}},
        {"question_id": "q3", "predicted_category": "single-session-user",
         "autoeval_label": {"label": True}},
    ])
    wrongs, rights = parse_baseline_eval(p, category_filter={"multi-session"})
    assert wrongs == {"q1": "multi-session"}
    assert rights == {}  # q3 was right but not in filter


def test_parse_baseline_eval_tolerates_blank_lines(tmp_path):
    from scripts.run_targeted_validation import parse_baseline_eval
    p = tmp_path / "eval.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1", "predicted_category": "x",
                    "autoeval_label": {"label": True}}) + "\n"
        + "\n\n"
        + json.dumps({"question_id": "q2", "predicted_category": "x",
                      "autoeval_label": {"label": False}}) + "\n"
    )
    wrongs, rights = parse_baseline_eval(p)
    assert "q2" in wrongs
    assert "q1" in rights


def test_parse_baseline_eval_handles_missing_label(tmp_path):
    """A row with no autoeval_label is treated as wrong (default False)."""
    from scripts.run_targeted_validation import parse_baseline_eval
    p = tmp_path / "eval.jsonl"
    _write_eval(p, [
        {"question_id": "q1", "predicted_category": "x"},  # no label
    ])
    wrongs, rights = parse_baseline_eval(p)
    assert wrongs == {"q1": "x"}
    assert rights == {}


# ── write_subset_dataset ──────────────────────────────────────────────


def test_write_subset_dataset_keeps_only_target_qids(tmp_path):
    from scripts.run_targeted_validation import write_subset_dataset
    full = tmp_path / "full.json"
    full.write_text(json.dumps([
        {"question_id": "q1", "question": "a"},
        {"question_id": "q2", "question": "b"},
        {"question_id": "q3", "question": "c"},
    ]))
    out = tmp_path / "sub.json"
    n = write_subset_dataset(full, {"q1", "q3"}, out)
    assert n == 2
    sub = json.loads(out.read_text())
    assert sorted(q["question_id"] for q in sub) == ["q1", "q3"]


def test_write_subset_dataset_empty_keep_set_writes_empty_array(tmp_path):
    from scripts.run_targeted_validation import write_subset_dataset
    full = tmp_path / "full.json"
    full.write_text(json.dumps([{"question_id": "q1"}]))
    out = tmp_path / "sub.json"
    n = write_subset_dataset(full, set(), out)
    assert n == 0
    assert json.loads(out.read_text()) == []


# ── parse_judge_output ────────────────────────────────────────────────


def test_parse_judge_output_returns_label_dict(tmp_path):
    from scripts.run_targeted_validation import parse_judge_output
    p = tmp_path / "judge.jsonl"
    _write_eval(p, [
        {"question_id": "q1", "autoeval_label": {"label": True}},
        {"question_id": "q2", "autoeval_label": {"label": False}},
    ])
    out = parse_judge_output(p)
    assert out == {"q1": True, "q2": False}


# ── print_delta_report ────────────────────────────────────────────────


def test_print_delta_report_classifies_buckets(capsys):
    from scripts.run_targeted_validation import print_delta_report
    wrongs = {"w1": "multi-session", "w2": "multi-session", "w3": "temporal-reasoning"}
    rights = {"r1": "multi-session", "r2": "single-session-user", "r3": "knowledge-update"}
    sampled = {"r1", "r2", "r3"}
    new_labels = {
        "w1": True,    # fixed
        "w2": False,   # still wrong
        "w3": True,    # fixed
        "r1": True,    # stable
        "r2": False,   # regressed
        "r3": True,    # stable
    }
    out = print_delta_report(
        wrongs=wrongs,
        rights=rights,
        sampled_rights=sampled,
        new_labels=new_labels,
    )
    assert sorted(out["fixed"]) == ["w1", "w3"]
    assert out["still_wrong"] == ["w2"]
    assert out["regressed"] == ["r2"]
    assert sorted(out["stable"]) == ["r1", "r3"]

    # Stdout content sanity
    captured = capsys.readouterr().out
    assert "FIXED" in captured
    assert "REGRESSED" in captured
    assert "r2" in captured  # regression name surfaced for human attention


def test_print_delta_report_handles_missing_qid_in_new_labels(capsys):
    """If the targeted run errored out on a question, we shouldn't crash."""
    from scripts.run_targeted_validation import print_delta_report
    wrongs = {"w1": "multi-session"}
    rights = {"r1": "multi-session"}
    sampled = {"r1"}
    new_labels = {"r1": True}  # w1 missing from re-run
    out = print_delta_report(
        wrongs=wrongs,
        rights=rights,
        sampled_rights=sampled,
        new_labels=new_labels,
    )
    # w1 not in either fixed or still-wrong since it had no new label.
    assert out["fixed"] == []
    assert out["still_wrong"] == []
    assert out["stable"] == ["r1"]
