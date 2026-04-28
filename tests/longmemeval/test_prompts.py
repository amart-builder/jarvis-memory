"""Smoke tests for the LongMemEval prompt module.

Verifies template selection, rendering, date parsing, session
formatting — the small functional surface that the adapter relies on.
"""
from __future__ import annotations

import pytest

from scripts.longmemeval.prompts import (
    RAG_PROMPT_ENHANCED,
    RAG_PROMPT_MULTISESSION,
    RAG_PROMPT_PREFERENCE,
    RAG_PROMPT_TEMPORAL,
    RAG_PROMPT_VANILLA,
    answer_to_str,
    format_session_for_prompt,
    format_session_text,
    get_prompt_template,
    parse_longmemeval_date,
    render_prompt,
)


# ── Template selection ────────────────────────────────────────────────


def test_each_category_maps_to_a_template():
    expected = {
        "single-session-user": RAG_PROMPT_VANILLA,
        "single-session-assistant": RAG_PROMPT_VANILLA,
        "single-session-preference": RAG_PROMPT_PREFERENCE,
        "knowledge-update": RAG_PROMPT_ENHANCED,
        "multi-session": RAG_PROMPT_MULTISESSION,
        "temporal-reasoning": RAG_PROMPT_TEMPORAL,
    }
    for cat, tmpl in expected.items():
        assert get_prompt_template(cat) is tmpl, f"wrong template for {cat}"


def test_unknown_category_falls_back_to_vanilla():
    assert get_prompt_template("not-a-category") is RAG_PROMPT_VANILLA


# ── Template content sanity ───────────────────────────────────────────
# These guard against accidental edits — if a template is silently
# modified, the test breaks loud.


def test_enhanced_has_recency_rule():
    assert "LATEST date is the ONLY correct one" in RAG_PROMPT_ENHANCED
    assert "SUPERSEDED" in RAG_PROMPT_ENHANCED


def test_multisession_has_enumeration_rule():
    """AR3 — counting enumeration ships free as part of OMEGA's prompt."""
    assert "list EVERY matching item" in RAG_PROMPT_MULTISESSION
    assert "[Note #]" in RAG_PROMPT_MULTISESSION


def test_multisession_has_dedup_rule():
    assert "DEDUPLICATION" in RAG_PROMPT_MULTISESSION
    assert "merging duplicates" in RAG_PROMPT_MULTISESSION


def test_temporal_has_recollection_rule():
    """The 'RECOLLECTION ≠ ACTION' rule is OMEGA's biggest temporal lift."""
    assert "RECOLLECTION ≠ ACTION" in RAG_PROMPT_TEMPORAL


def test_preference_forces_personalization():
    assert "Generic advice" in RAG_PROMPT_PREFERENCE
    assert "WRONG" in RAG_PROMPT_PREFERENCE


def test_all_templates_have_format_anchors():
    for tmpl in (RAG_PROMPT_VANILLA, RAG_PROMPT_ENHANCED, RAG_PROMPT_MULTISESSION,
                 RAG_PROMPT_PREFERENCE, RAG_PROMPT_TEMPORAL):
        assert "{sessions}" in tmpl
        assert "{question}" in tmpl
        assert "{question_date}" in tmpl


# ── render_prompt ─────────────────────────────────────────────────────


def test_render_prompt_substitutes_fields():
    out = render_prompt(
        category="single-session-user",
        sessions="[Note 1 | Date: 2024-01-15T10:00:00]\nfoo: bar\n[End Note 1]",
        question="What did I do?",
        question_date="2024-01-20T12:00:00",
    )
    assert "What did I do?" in out
    assert "2024-01-20T12:00:00" in out
    assert "[Note 1 | Date: 2024-01-15T10:00:00]" in out


def test_render_prompt_picks_temporal_for_temporal_reasoning():
    out = render_prompt(
        category="temporal-reasoning",
        sessions="x",
        question="When?",
        question_date="d",
    )
    assert "RECOLLECTION ≠ ACTION" in out


# ── Session text formatting ───────────────────────────────────────────


def test_format_session_text_concatenates_turns():
    turns = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How are you"},
    ]
    out = format_session_text(turns)
    assert out == "user: Hi\nassistant: Hello\nuser: How are you"


def test_format_session_text_empty():
    assert format_session_text([]) == ""


def test_format_session_for_prompt_wraps_with_note_block():
    out = format_session_for_prompt("hello world", "2024-01-15T10:00:00", 3)
    assert out.startswith("[Note 3 | Date: 2024-01-15T10:00:00]\n")
    assert out.endswith("\n[End Note 3]")
    assert "hello world" in out


# ── Date parsing ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2024/01/15 (Monday) 14:30", "2024-01-15T14:30:00"),
        ("2024/12/31 (Tuesday) 09:05", "2024-12-31T09:05:00"),
        ("2024/06/01 (Saturday) 00:00", "2024-06-01T00:00:00"),
    ],
)
def test_parse_longmemeval_date_strips_weekday(raw: str, expected: str):
    assert parse_longmemeval_date(raw) == expected


def test_parse_longmemeval_date_returns_input_on_failure():
    """Bad input is returned as-is — fail open, never crash the adapter."""
    assert parse_longmemeval_date("garbage") == "garbage"


# ── Answer normalization ──────────────────────────────────────────────


def test_answer_to_str_handles_list():
    assert answer_to_str(["a", "b", "c"]) == "a, b, c"


def test_answer_to_str_handles_string():
    assert answer_to_str("plain") == "plain"


def test_answer_to_str_handles_int():
    assert answer_to_str(42) == "42"


# ── Stage 4A — two-pass MS counting prompts ─────────────────────────


def test_render_ms_extract_prompt_basic_structure():
    from scripts.longmemeval.prompts import render_ms_extract_prompt
    out = render_ms_extract_prompt(
        sessions="[Note 1] foo\n[Note 2] bar",
        question="How many bakes?",
        question_date="2026-04-27T00:00:00",
    )
    # Pass 1 must NOT ask for a total — that's pass 2's job.
    assert "Total: N" not in out
    # Must instruct the model to be liberal / not dedupe.
    assert "MAXIMALLY INCLUSIVE" in out
    assert "Do NOT compute a total" in out
    # Substitutions plumbed through.
    assert "[Note 1] foo" in out
    assert "How many bakes?" in out
    assert "2026-04-27T00:00:00" in out


def test_render_ms_count_prompt_basic_structure():
    from scripts.longmemeval.prompts import render_ms_count_prompt
    out = render_ms_count_prompt(
        sessions="[Note 1] foo",
        candidate_list="1. baked banana bread [Note 1]",
        question="How many bakes?",
        question_date="2026-04-27T00:00:00",
    )
    # Pass 2 MUST instruct the model to output Total: N.
    assert 'Total: N' in out
    # Must reference the candidate list (not just the notes).
    assert "Candidate list" in out
    assert "1. baked banana bread [Note 1]" in out
    # Original notes still visible for verification.
    assert "[Note 1] foo" in out
    # Question + date plumbed through.
    assert "How many bakes?" in out


# ── Stages 4E / 4C / 4F / 4G / 4H — surgical prompt patches ──────────
# Each test pins a specific phrase the patch added — guards against
# accidental edits and acts as documentation of "this rule is here on
# purpose, here's the failing-question pattern it targets."


def test_ms_count_inclusion_drops_planning_and_wishing():
    """Stage 4E — pass-2 should drop candidates that are plans/wishes/recollections.
    Targets 6d550036 (Nigeria project = planning), 88432d0a (ingredients I'm
    going to use this weekend = future plan)."""
    from scripts.longmemeval.prompts import RAG_PROMPT_MULTISESSION_COUNT
    # The new INCLUSION RULE must enumerate these drop conditions.
    assert "PLANNED, INTENDED, or WANTED" in RAG_PROMPT_MULTISESSION_COUNT
    assert "hypothetical / conditional" in RAG_PROMPT_MULTISESSION_COUNT
    assert "DROP" in RAG_PROMPT_MULTISESSION_COUNT


def test_ms_count_keeps_user_did_thing_with_sparse_details():
    """Stage 4E corollary — the rule must NOT over-prune. Borderline 'I did
    it' candidates with sparse details should still be kept."""
    from scripts.longmemeval.prompts import RAG_PROMPT_MULTISESSION_COUNT
    assert "KEEP borderline candidates where the user clearly DID" in RAG_PROMPT_MULTISESSION_COUNT


def test_enhanced_has_cumulative_count_rule():
    """Stage 4C — KU prompt must handle 'earlier explicit count + later
    implicit increment'. Targets f9e8c073 (3 sessions stated → 5 implied)
    and 45dc21b6 (Emma's recipes count rises across notes)."""
    from scripts.longmemeval.prompts import RAG_PROMPT_ENHANCED
    assert "CUMULATIVE" in RAG_PROMPT_ENHANCED
    assert "INCREMENT" in RAG_PROMPT_ENHANCED
    # Ordinal language rule for "my Nth time" pattern.
    assert "ORDINAL LANGUAGE" in RAG_PROMPT_ENHANCED
    assert '"my Nth time"' in RAG_PROMPT_ENHANCED
    # Later-evidence-wins disambiguation (explicit vs implicit conflict).
    assert "LATER one wins" in RAG_PROMPT_ENHANCED


def test_enhanced_has_previous_former_rule():
    """Stage 4F — 'previous/former/old' questions ask for the EARLIER
    value, NOT the latest. Targets e66b632c (previous PB 5K). Note: 'FIRST'
    deliberately excluded after reviewer flagged ambiguity (first-note vs
    first-event-referenced)."""
    from scripts.longmemeval.prompts import RAG_PROMPT_ENHANCED
    assert "PREVIOUS" in RAG_PROMPT_ENHANCED
    assert "FORMER" in RAG_PROMPT_ENHANCED
    assert "use the EARLIER value" in RAG_PROMPT_ENHANCED


def test_enhanced_previous_rule_excludes_first():
    """Reviewer fix — 'FIRST' was removed from the trigger list because it's
    ambiguous (could mean earliest note OR the first instance referenced)."""
    from scripts.longmemeval.prompts import RAG_PROMPT_ENHANCED
    # Find the PREVIOUS rule line and check FIRST is not in the trigger list.
    lines = [l for l in RAG_PROMPT_ENHANCED.split("\n") if "PREVIOUS" in l and "OLD" in l]
    assert len(lines) == 1, "PREVIOUS rule should appear exactly once"
    # The trigger list ends after ORIGINAL — FIRST was removed.
    assert "ORIGINAL QUESTIONS" in lines[0] or "ORIGINAL " in lines[0]


def test_enhanced_has_substitution_abstention_rule():
    """Stage 4G (KU) — don't substitute a similar-but-different item when
    the question asks for a specific one."""
    from scripts.longmemeval.prompts import RAG_PROMPT_ENHANCED
    assert "ABSTENTION ON QUESTION-SUBSTITUTION TRAPS" in RAG_PROMPT_ENHANCED


def test_multisession_has_substitution_abstention_rule():
    """Stage 4G (MS pass-1) — same substitution guard."""
    from scripts.longmemeval.prompts import RAG_PROMPT_MULTISESSION
    assert "ABSTENTION ON QUESTION-SUBSTITUTION TRAPS" in RAG_PROMPT_MULTISESSION


def test_ms_count_has_both_sides_rule():
    """Stage 4G (MS pass-2) — compare/save/diff questions need BOTH sides
    quantified. Targets 09ba9854_abs (asks about bus savings; notes only
    cover train)."""
    from scripts.longmemeval.prompts import RAG_PROMPT_MULTISESSION_COUNT
    assert "BOTH-SIDES RULE" in RAG_PROMPT_MULTISESSION_COUNT


def test_temporal_has_before_after_event_rule():
    """Stage 4H — 'X before/after event Y' counting questions need
    explicit date filtering. Targets a3838d2b (charity events before
    Run for the Cure)."""
    from scripts.longmemeval.prompts import RAG_PROMPT_TEMPORAL
    assert "BEFORE / AFTER A KNOWN EVENT" in RAG_PROMPT_TEMPORAL


def test_enhanced_has_cross_attribute_rule():
    """Stage 4G+ — attribute must match exactly. Targets a96c20ee_abs
    (undergrad vs thesis substitution)."""
    from scripts.longmemeval.prompts import RAG_PROMPT_ENHANCED
    assert "CROSS-ATTRIBUTE SUBSTITUTION TRAPS" in RAG_PROMPT_ENHANCED
    assert "UNDERGRAD" in RAG_PROMPT_ENHANCED


def test_multisession_has_cross_attribute_rule():
    """Stage 4G+ — same in MULTISESSION."""
    from scripts.longmemeval.prompts import RAG_PROMPT_MULTISESSION
    assert "CROSS-ATTRIBUTE SUBSTITUTION TRAPS" in RAG_PROMPT_MULTISESSION


def test_ms_count_has_cross_attribute_rule():
    """Stage 4G+ — same in MS pass-2."""
    from scripts.longmemeval.prompts import RAG_PROMPT_MULTISESSION_COUNT
    assert "CROSS-ATTRIBUTE SUBSTITUTION" in RAG_PROMPT_MULTISESSION_COUNT


def test_temporal_has_nth_occurrence_rule():
    """Stage 4H — 'the Nth time I did X' enumeration rule. Targets
    370a8ff4 (10th jog after recovering from flu)."""
    from scripts.longmemeval.prompts import RAG_PROMPT_TEMPORAL
    assert "Nth OCCURRENCE" in RAG_PROMPT_TEMPORAL


def test_ms_extract_and_count_prompts_use_same_question_keys():
    """Both passes substitute {sessions}, {question}, {question_date}.
    Pass 2 also substitutes {candidate_list}. Drift between them would
    silently break — pin the contract."""
    from scripts.longmemeval.prompts import (
        RAG_PROMPT_MULTISESSION_EXTRACT,
        RAG_PROMPT_MULTISESSION_COUNT,
    )
    extract_keys = {"sessions", "question", "question_date"}
    count_keys = {"sessions", "question", "question_date", "candidate_list"}
    # Use string.Formatter to get the actual placeholders.
    import string
    fmt = string.Formatter()
    extract_actual = {
        f for _, f, _, _ in fmt.parse(RAG_PROMPT_MULTISESSION_EXTRACT)
        if f
    }
    count_actual = {
        f for _, f, _, _ in fmt.parse(RAG_PROMPT_MULTISESSION_COUNT)
        if f
    }
    assert extract_actual == extract_keys
    assert count_actual == count_keys
