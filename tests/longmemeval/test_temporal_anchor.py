"""Unit tests for ``scripts/longmemeval/temporal_anchor.py`` — Stage 4D.

The temporal anchor is a ported OMEGA technique that's the largest
delta between OMEGA's 95.4% and our Stage-1.5 93.4% on gpt-4.1.
Failure-mode analysis showed that for date-anchored questions
("in March", "last weekend", "two weeks ago"), gold sessions are
RETRIEVED but ranked low — these helpers push them to the top.

Tests pin:
  * Each pattern in ``infer_temporal_range_anchored`` (last weekday,
    "N weeks ago", "between X and Y", "last/past N units", "in Month YYYY")
  * Each pattern in ``resolve_relative_dates`` (date-keyword expansion
    used by the embedding side of expansion)
  * ``expand_query`` composition (counting + dates + entities,
    additive to the original query, no-op when no signals fire)
  * ``hit_in_temporal_window`` boundary behavior + malformed-input
    safety
"""
from __future__ import annotations

from datetime import datetime

import pytest


# ── infer_temporal_range_anchored ─────────────────────────────────────


@pytest.fixture
def anchor_iso() -> str:
    """Friday 2024-03-15 12:00. Picked so weekday math is non-trivial."""
    return "2024-03-15T12:00:00"


def test_infer_anchor_last_monday_resolves_to_previous_monday(anchor_iso):
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored("what did I do last Monday", anchor_iso)
    assert out is not None
    start, end = out
    # Anchor is Friday March 15 2024. "Last Monday" = March 11.
    assert start.startswith("2024-03-09")  # Mar 11 minus 2 days = Mar 9
    assert end.startswith("2024-03-13")    # Mar 11 plus 2 days = Mar 13


def test_infer_anchor_last_weekend_returns_window(anchor_iso):
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored("did anything happen last weekend", anchor_iso)
    assert out is not None
    start, _end = out
    # Most recent Saturday before Friday Mar 15 = Saturday Mar 9.
    # Buffer of 2 days each side → Mar 7 onward.
    assert "2024-03-07" in start


def test_infer_anchor_two_weeks_ago_centers_on_past_date(anchor_iso):
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored("how many bakes two weeks ago", anchor_iso)
    assert out is not None
    start, end = out
    # 14 days before Mar 15 = Mar 1. Buffer = max(14*0.25, 3) = 3.5 days.
    assert "2024-02" in start  # late February
    assert "2024-03-05" in end  # roughly Mar 5


def test_infer_anchor_word_number_n_weeks_ago(anchor_iso):
    """"three weeks ago" should parse as 3 (word_to_num)."""
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored("three weeks ago", anchor_iso)
    assert out is not None
    start, end = out
    # Center: Mar 15 minus 21 days = Feb 23.
    assert "2024-02" in start


def test_infer_anchor_between_dates_returns_explicit_window(anchor_iso):
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored(
        "what happened between 2024-01-15 and 2024-02-20",
        anchor_iso,
    )
    assert out is not None
    start, end = out
    # Window buffered by 1 day each side.
    assert "2024-01-14" in start
    assert "2024-02-21" in end


def test_infer_anchor_past_two_weeks_returns_anchor_minus_window(anchor_iso):
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored(
        "how many bakes in the past two weeks", anchor_iso,
    )
    assert out is not None
    start, end = out
    # Should run from anchor minus 14 days back, ending at anchor.
    assert "2024-03-15" in end
    assert "2024-02-29" in start or "2024-03-01" in start


def test_infer_anchor_in_month_year_explicit(anchor_iso):
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored("activities in January 2024", anchor_iso)
    assert out is not None
    start, end = out
    # January 2024 → window with 1-day buffer each side: Dec 31 2023 → Feb 2 2024.
    assert "2023-12-31" in start
    assert "2024-02-02" in end


def test_infer_anchor_in_month_only_returns_none(anchor_iso):
    """"in March" without explicit year is handled by resolve_relative_dates,
    NOT by the range filter (matches OMEGA exactly — line 681 of their script
    requires explicit year)."""
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored("how many runs in March", anchor_iso)
    assert out is None


def test_infer_anchor_no_temporal_signal_returns_none(anchor_iso):
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    assert infer_temporal_range_anchored("what is my favorite color", anchor_iso) is None


def test_infer_anchor_malformed_anchor_returns_none():
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    # Garbage anchor — should fall through both parsers and return None.
    assert infer_temporal_range_anchored("last Monday", "not a date") is None


def test_infer_anchor_handles_longmemeval_native_format():
    """LongMemEval stores dates as ``YYYY/MM/DD (Day) HH:MM`` — must parse."""
    from scripts.longmemeval.temporal_anchor import infer_temporal_range_anchored
    out = infer_temporal_range_anchored(
        "yesterday I went running",
        "2024/03/15 (Fri) 12:00",
    )
    # "yesterday" isn't an infer_anchor pattern (it's only a resolve_dates one),
    # so it returns None — but the parser should not crash.
    assert out is None


# ── resolve_relative_dates ────────────────────────────────────────────


def test_resolve_yesterday_emits_absolute_date():
    from scripts.longmemeval.temporal_anchor import resolve_relative_dates
    anchor = datetime(2024, 3, 15, 12, 0)
    out = resolve_relative_dates("yesterday I went out", anchor)
    assert any("2024-03-14" in s for s in out)


def test_resolve_in_march_no_year_picks_latest_occurrence():
    from scripts.longmemeval.temporal_anchor import resolve_relative_dates
    anchor = datetime(2024, 5, 15, 12, 0)  # Anchor in May 2024
    # "in March" — most recent March before May 2024 → March 2024.
    out = resolve_relative_dates("how many fun runs in March", anchor)
    assert any("March 2024" in s for s in out)


def test_resolve_in_march_anchor_in_february_picks_prior_year():
    from scripts.longmemeval.temporal_anchor import resolve_relative_dates
    anchor = datetime(2024, 2, 15, 12, 0)  # Anchor before March
    # "in March" with anchor in Feb — most recent March is March 2023.
    out = resolve_relative_dates("how many fun runs in March", anchor)
    assert any("March 2023" in s for s in out)


def test_resolve_no_temporal_signal_returns_empty():
    from scripts.longmemeval.temporal_anchor import resolve_relative_dates
    anchor = datetime(2024, 3, 15)
    assert resolve_relative_dates("what is my favorite food", anchor) == []


# ── expand_query ──────────────────────────────────────────────────────


def test_expand_query_counting_appends_recall_cue():
    from scripts.longmemeval.temporal_anchor import expand_query
    out = expand_query("how many bakes did I do", "2024-03-15T12:00:00")
    assert "every instance all occurrences each time" in out
    assert "how many bakes" in out  # original preserved


def test_expand_query_resolves_in_month():
    from scripts.longmemeval.temporal_anchor import expand_query
    out = expand_query("activities in March", "2024-05-15T12:00:00")
    assert "March 2024" in out


def test_expand_query_extracts_proper_nouns():
    from scripts.longmemeval.temporal_anchor import expand_query
    out = expand_query("what did Rachel say at the concert in Brooklyn",
                       "2024-03-15T12:00:00")
    assert "Rachel" in out
    assert "Brooklyn" in out


def test_expand_query_skips_common_capitalized_words():
    from scripts.longmemeval.temporal_anchor import expand_query
    out = expand_query("What was the answer", "2024-03-15T12:00:00")
    # The trailing entities shouldn't include "What" or "The".
    assert out.endswith("What was the answer") or " What" not in out.split("answer")[-1]


def test_expand_query_no_signals_returns_unchanged():
    from scripts.longmemeval.temporal_anchor import expand_query
    q = "tell me about food"
    assert expand_query(q, "2024-03-15T12:00:00") == q


def test_expand_query_no_anchor_skips_temporal_expansion():
    from scripts.longmemeval.temporal_anchor import expand_query
    out = expand_query("how many bakes in March", None)
    # Counting cue still fires. Temporal expansion does NOT (no anchor).
    assert "every instance" in out
    assert "March 2024" not in out  # no resolved date


def test_expand_query_combines_all_three_signals():
    from scripts.longmemeval.temporal_anchor import expand_query
    out = expand_query(
        "how many times did Rachel visit Brooklyn in March",
        "2024-05-15T12:00:00",
    )
    assert "every instance" in out
    assert "March 2024" in out
    assert "Rachel" in out
    assert "Brooklyn" in out


# ── hit_in_temporal_window ───────────────────────────────────────────


def test_hit_in_window_inside_returns_true():
    from scripts.longmemeval.temporal_anchor import hit_in_temporal_window
    window = ("2024-03-01T00:00:00", "2024-03-31T00:00:00")
    assert hit_in_temporal_window("2024-03-15T10:00:00", window) is True


def test_hit_in_window_outside_returns_false():
    from scripts.longmemeval.temporal_anchor import hit_in_temporal_window
    window = ("2024-03-01T00:00:00", "2024-03-31T00:00:00")
    assert hit_in_temporal_window("2024-04-15T10:00:00", window) is False
    assert hit_in_temporal_window("2024-02-15T10:00:00", window) is False


def test_hit_in_window_boundary_is_inclusive():
    from scripts.longmemeval.temporal_anchor import hit_in_temporal_window
    window = ("2024-03-01T00:00:00", "2024-03-31T00:00:00")
    assert hit_in_temporal_window("2024-03-01T00:00:00", window) is True
    assert hit_in_temporal_window("2024-03-31T00:00:00", window) is True


def test_hit_in_window_malformed_date_returns_false():
    from scripts.longmemeval.temporal_anchor import hit_in_temporal_window
    window = ("2024-03-01T00:00:00", "2024-03-31T00:00:00")
    assert hit_in_temporal_window("not a date", window) is False
    assert hit_in_temporal_window("", window) is False


def test_hit_in_window_malformed_window_returns_false():
    from scripts.longmemeval.temporal_anchor import hit_in_temporal_window
    bad = ("not a date", "also bad")
    assert hit_in_temporal_window("2024-03-15T00:00:00", bad) is False
