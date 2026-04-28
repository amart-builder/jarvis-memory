"""Unit tests for ``scripts/longmemeval/evidence_packet.py`` — Stage 5
v2 Phase 3.

Pins:
  - ``_split_user_turns`` parsing (multi-line user turns, role headers)
  - ``_score_user_turn`` signal scoring (dates, quantities, ordinals,
    proper nouns, relative time)
  - ``_truncate_snippet`` boundary handling
  - ``build_evidence_packet`` end-to-end (empty input, signal threshold,
    chronological ordering, max_snippets cap, format anchors)
"""
from __future__ import annotations


# ── _split_user_turns ────────────────────────────────────────────────


def test_split_user_turns_basic():
    from scripts.longmemeval.evidence_packet import _split_user_turns
    content = (
        "user: Hello\n"
        "assistant: Hi there\n"
        "user: I baked cookies on May 21\n"
        "assistant: Sounds good\n"
        "user: I also went jogging\n"
    )
    out = _split_user_turns(content)
    assert out == ["Hello", "I baked cookies on May 21", "I also went jogging"]


def test_split_user_turns_handles_multiline_user_turn():
    """A user turn that spans multiple lines (e.g., the model's
    transcript wraps long user input) should be joined as one snippet."""
    from scripts.longmemeval.evidence_packet import _split_user_turns
    content = (
        "user: I want to talk about my marathon.\n"
        "I trained for 16 weeks.\n"
        "It was on May 21 2023.\n"
        "assistant: That's impressive\n"
    )
    out = _split_user_turns(content)
    assert len(out) == 1
    assert "marathon" in out[0]
    assert "16 weeks" in out[0]
    assert "May 21 2023" in out[0]


def test_split_user_turns_skips_assistant_turns():
    from scripts.longmemeval.evidence_packet import _split_user_turns
    content = (
        "assistant: I recommend you try yoga.\n"
        "user: I prefer pilates\n"
        "assistant: That works too. Try this routine.\n"
    )
    out = _split_user_turns(content)
    assert out == ["I prefer pilates"]


def test_split_user_turns_empty_input():
    from scripts.longmemeval.evidence_packet import _split_user_turns
    assert _split_user_turns("") == []
    assert _split_user_turns("\n\n") == []


def test_split_user_turns_unknown_format_returns_empty():
    """Lines with no role: prefix yield no turns (defensive)."""
    from scripts.longmemeval.evidence_packet import _split_user_turns
    out = _split_user_turns("just some random text\nno roles here")
    assert out == []


# ── _score_user_turn ─────────────────────────────────────────────────


def test_score_user_turn_iso_date():
    from scripts.longmemeval.evidence_packet import _score_user_turn
    assert _score_user_turn("I baked on 2023-05-21.") >= 3


def test_score_user_turn_written_date():
    from scripts.longmemeval.evidence_packet import _score_user_turn
    assert _score_user_turn("I baked on March 15, 2023.") >= 3
    assert _score_user_turn("My birthday is May 21st.") >= 3
    assert _score_user_turn("I went on the 5th of June.") >= 3


def test_score_user_turn_relative_time():
    from scripts.longmemeval.evidence_packet import _score_user_turn
    assert _score_user_turn("I baked yesterday.") >= 3
    assert _score_user_turn("I went last weekend.") >= 3
    assert _score_user_turn("Two weeks ago I started.") >= 3


def test_score_user_turn_quantity_dollars():
    from scripts.longmemeval.evidence_packet import _score_user_turn
    assert _score_user_turn("I spent $50 on dinner.") >= 2
    assert _score_user_turn("It cost 100 dollars.") >= 2


def test_score_user_turn_quantity_count():
    from scripts.longmemeval.evidence_packet import _score_user_turn
    assert _score_user_turn("I attended 5 sessions this month.") >= 2
    assert _score_user_turn("I have 3 chickens.") >= 2


def test_score_user_turn_ordinal():
    from scripts.longmemeval.evidence_packet import _score_user_turn
    assert _score_user_turn("This is my 5th time.") >= 2
    assert _score_user_turn("That was the third visit.") >= 2


def test_score_user_turn_proper_nouns():
    from scripts.longmemeval.evidence_packet import _score_user_turn
    # "Rachel" and "Brooklyn" both score; common stoplist words don't.
    score_with_names = _score_user_turn("I went to Brooklyn with Rachel.")
    score_without = _score_user_turn("I went to a place with a friend.")
    assert score_with_names > score_without


def test_score_user_turn_stoplist_excluded():
    """Days, months, common articles shouldn't count as proper nouns."""
    from scripts.longmemeval.evidence_packet import _score_user_turn
    # "Monday" and "March" are in stoplist (they're caught as date signals,
    # not proper-noun signals).
    score = _score_user_turn("Monday and March are normal words.")
    # Should score for "in March" relative-time pattern (3) + maybe nothing else.
    # Important: no proper-noun bonus for Monday/March.
    assert score < 6  # would be higher if Monday+March each got proper-noun bonus


def test_score_user_turn_zero_for_signal_free_text():
    from scripts.longmemeval.evidence_packet import _score_user_turn
    assert _score_user_turn("hello") == 0
    assert _score_user_turn("i am fine") == 0


# ── _truncate_snippet ────────────────────────────────────────────────


def test_truncate_snippet_short_text_unchanged():
    from scripts.longmemeval.evidence_packet import _truncate_snippet
    assert _truncate_snippet("short") == "short"


def test_truncate_snippet_breaks_at_sentence_boundary():
    from scripts.longmemeval.evidence_packet import _truncate_snippet
    text = "This is sentence one. This is sentence two and is much much longer than the first one and keeps going on and on and on with more words."
    out = _truncate_snippet(text, max_chars=80)
    # Should end at the period after "sentence one" — not mid-word.
    assert out.endswith(".")
    assert "sentence one" in out


def test_truncate_snippet_hard_cuts_when_no_boundary():
    from scripts.longmemeval.evidence_packet import _truncate_snippet
    text = "x" * 200  # no sentence boundaries
    out = _truncate_snippet(text, max_chars=50)
    assert len(out) <= 50
    assert out.endswith("...")


# ── build_evidence_packet end-to-end ─────────────────────────────────


def _hit(idx, content, date="2023-05-15"):
    return {"content": content, "referenced_date": date, "uuid": f"u{idx}"}


def test_build_evidence_packet_empty_hits_returns_empty_string():
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    assert build_evidence_packet([], "any question") == ""


def test_build_evidence_packet_no_signal_returns_empty():
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = [_hit(1, "user: hello\nassistant: hi\n")]
    assert build_evidence_packet(hits, "any") == ""


def test_build_evidence_packet_includes_signal_user_turn():
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = [_hit(1, "user: I baked sourdough on 2023-05-21\nassistant: nice\n")]
    out = build_evidence_packet(hits, "how many bakes?")
    assert out
    assert "[High-signal evidence" in out
    assert "I baked sourdough" in out
    assert "[Note 1]" in out
    assert "2023-05-15" in out  # date prefix from referenced_date
    assert out.endswith(
        "[End of evidence packet. Full chronological notes follow — these "
        "are the ground truth.]"
    )


def test_build_evidence_packet_drops_assistant_turns():
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = [_hit(1,
        "user: nothing relevant here\n"
        "assistant: I recommend buying 5 books on May 21 in Brooklyn\n"
    )]
    # Assistant has lots of signal but should NOT contribute.
    out = build_evidence_packet(hits, "any")
    assert out == ""


def test_build_evidence_packet_chronological_ordering():
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = [
        _hit(1, "user: I baked apple pie on 2023-05-20\n", "2023-05-20"),
        _hit(2, "user: I baked sourdough on 2023-05-21\n", "2023-05-21"),
    ]
    out = build_evidence_packet(hits, "any")
    # Apple pie line must come before sourdough line.
    apple_idx = out.find("apple pie")
    sourdough_idx = out.find("sourdough")
    assert 0 < apple_idx < sourdough_idx


def test_build_evidence_packet_caps_at_max_snippets():
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = []
    for i in range(20):
        hits.append(_hit(i + 1, f"user: I baked something on 2023-05-{i+1:02d}\n"))
    out = build_evidence_packet(hits, "any", max_snippets=5)
    # Count "User:" occurrences in the packet.
    assert out.count("User:") == 5


def test_build_evidence_packet_format_has_index_anchors():
    """Note numbers in packet must match the chronological note numbering
    in the prompt below — (i+1) where i is the hit list index."""
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = [
        _hit(1, "user: I went jogging in Brooklyn on May 5\n"),
        _hit(2, "user: I bought a book on June 12 for $20\n"),
    ]
    out = build_evidence_packet(hits, "any")
    assert "[Note 1]" in out
    assert "[Note 2]" in out


def test_build_evidence_packet_signal_threshold_drops_low_score():
    """Default min_signal_score=1 — turns with no signal markers must
    be excluded even if other turns make it in."""
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = [
        _hit(1,
            "user: I went jogging in Brooklyn on May 5 with Rachel\n"
            "assistant: nice\n"
            "user: hi\n"  # zero signal — must be dropped
        ),
    ]
    out = build_evidence_packet(hits, "any")
    assert "Brooklyn" in out
    assert '"hi"' not in out


def test_build_evidence_packet_handles_missing_content_field():
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = [{"referenced_date": "2023-05-15"}]  # no "content"
    assert build_evidence_packet(hits, "any") == ""


def test_build_evidence_packet_handles_missing_date_field():
    from scripts.longmemeval.evidence_packet import build_evidence_packet
    hits = [{"content": "user: I baked on May 21\n"}]  # no date
    out = build_evidence_packet(hits, "any")
    # Should still produce a packet, just without the date prefix.
    assert "[Note 1]" in out
    assert "I baked on May 21" in out


# ── Stage 5 v2 Phase 4 — temporal two-lane partitioning ──────────────
# These pin the partition logic separately from the adapter wiring, so
# we can test it without spinning up Neo4j/Chroma.


def _two_lane_partition(hits, window):
    """Mirror the partition logic in run_longmemeval.run_one_question
    section 4. Pulled out as a helper so the test can pin behavior
    without instantiating the full adapter."""
    from scripts.longmemeval.temporal_anchor import hit_in_temporal_window
    in_window = []
    out_window = []
    for i, h in enumerate(hits):
        ref = str(h.get("referenced_date") or h.get("created_at") or "")
        if ref and hit_in_temporal_window(ref, window):
            in_window.append((i + 1, h))
        else:
            out_window.append((i + 1, h))
    return in_window, out_window


def test_temporal_two_lane_partitions_by_window():
    window = ("2023-05-15T00:00:00", "2023-05-30T00:00:00")
    hits = [
        {"referenced_date": "2023-05-10T00:00:00", "content": "x"},  # before
        {"referenced_date": "2023-05-20T00:00:00", "content": "y"},  # in
        {"referenced_date": "2023-06-01T00:00:00", "content": "z"},  # after
    ]
    in_w, out_w = _two_lane_partition(hits, window)
    assert len(in_w) == 1
    assert in_w[0][0] == 2  # original [Note 2] preserved
    assert len(out_w) == 2
    assert {idx for idx, _ in out_w} == {1, 3}


def test_temporal_two_lane_preserves_chronological_indices():
    """Each hit keeps its original (i+1) index so [Note N] references
    stay consistent with the evidence packet that runs above us."""
    window = ("2023-05-01T00:00:00", "2023-05-31T00:00:00")
    hits = [
        {"referenced_date": "2023-04-15T00:00:00", "content": "a"},
        {"referenced_date": "2023-05-15T00:00:00", "content": "b"},
        {"referenced_date": "2023-04-20T00:00:00", "content": "c"},
        {"referenced_date": "2023-05-20T00:00:00", "content": "d"},
    ]
    in_w, out_w = _two_lane_partition(hits, window)
    in_indices = [idx for idx, _ in in_w]
    out_indices = [idx for idx, _ in out_w]
    assert in_indices == [2, 4]
    assert out_indices == [1, 3]


def test_temporal_two_lane_handles_missing_dates_as_out_of_window():
    """A note with no date can't be classified as in-window; treat as
    out. Means it stays in context but doesn't get the salience boost."""
    window = ("2023-05-01T00:00:00", "2023-05-31T00:00:00")
    hits = [
        {"referenced_date": "", "content": "no date"},
        {"referenced_date": "2023-05-15T00:00:00", "content": "in"},
    ]
    in_w, out_w = _two_lane_partition(hits, window)
    assert [idx for idx, _ in in_w] == [2]
    assert [idx for idx, _ in out_w] == [1]


def test_temporal_two_lane_all_in_window_returns_empty_out():
    window = ("2023-05-01T00:00:00", "2023-05-31T00:00:00")
    hits = [
        {"referenced_date": "2023-05-10T00:00:00", "content": "a"},
        {"referenced_date": "2023-05-20T00:00:00", "content": "b"},
    ]
    in_w, out_w = _two_lane_partition(hits, window)
    assert len(in_w) == 2
    assert out_w == []


def test_temporal_two_lane_all_out_of_window_returns_empty_in():
    window = ("2023-05-01T00:00:00", "2023-05-31T00:00:00")
    hits = [
        {"referenced_date": "2023-04-10T00:00:00", "content": "a"},
        {"referenced_date": "2023-06-01T00:00:00", "content": "b"},
    ]
    in_w, out_w = _two_lane_partition(hits, window)
    assert in_w == []
    assert len(out_w) == 2
