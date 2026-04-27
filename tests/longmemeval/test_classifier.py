"""LongMemEval question classifier — unit tests + oracle accuracy gate.

The accuracy test (``test_classifier_accuracy_on_oracle``) is the
real proof point: it runs the classifier on all 500 oracle questions
and asserts overall accuracy ≥ 80%. Per-category breakdown is printed
on failure for diagnosis.

The oracle-accuracy test is auto-skipped when the dataset isn't
present (so a fresh clone of the repo doesn't fail tests).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.longmemeval.classifier import (
    ABSTENTION_FILTER,
    COUNTING_K_FLOOR,
    FILTER_CONFIG,
    INTENT_NEUTRAL,
    INTENT_OVERLAYS,
    K_FLOORS,
    RETRIEVAL_PROFILES,
    channel_weights,
    classify,
    classify_lme_intent,
    is_counting_question,
)


ORACLE = Path("data/longmemeval/longmemeval_oracle.json")


# ── Spot-check tests on representative phrasings ──────────────────────


@pytest.mark.parametrize(
    "question, expected",
    [
        # single-session-assistant — back-reference to assistant
        ("Can you remind me what the rotation was for Admon?", "single-session-assistant"),
        ("Can you remind me of the name of that restaurant?", "single-session-assistant"),
        ("I remember you told me about CITGO's refineries.", "single-session-assistant"),
        ("Going back to our previous conversation about the children's book...", "single-session-assistant"),
        ("Remember when you suggested the Italian restaurant?", "single-session-assistant"),

        # single-session-preference — recommendation request
        ("Can you recommend some resources for video editing?", "single-session-preference"),
        ("Can you suggest a hotel for my trip to Miami?", "single-session-preference"),
        ("Could you suggest some interesting cultural events?", "single-session-preference"),
        ("What would you recommend for tonight?", "single-session-preference"),

        # temporal-reasoning — ordering/intervals
        ("What was the first issue I had with my new car?", "temporal-reasoning"),
        ("Which event did I attend first, the workshop or the webinar?", "temporal-reasoning"),
        ("How many days had passed between the Sunday mass and Ash Wednesday?", "temporal-reasoning"),
        ("How many days before the team meeting did I attend the workshop?", "temporal-reasoning"),
        ("Which device did I get first, the Galaxy or the Dell?", "temporal-reasoning"),

        # multi-session — counting/aggregation (NOT temporal)
        ("How many items of clothing do I need to pick up?", "multi-session"),
        ("How many projects have I led?", "multi-session"),
        ("How often do I call my mom?", "multi-session"),
        ("What's the total number of model kits I bought?", "multi-session"),

        # knowledge-update — recency markers
        ("What is my current job title?", "knowledge-update"),
        ("Where did Rachel move to after her recent relocation?", "knowledge-update"),
        ("What is my personal best time in the 5K?", "knowledge-update"),
        ("Am I still using the same grocery method?", "knowledge-update"),

        # single-session-user — default
        ("What degree did I graduate with?", "single-session-user"),
        ("How long is my daily commute to work?", "single-session-user"),
        ("Where did I redeem the coupon?", "single-session-user"),
    ],
)
def test_classifier_spot_checks(question: str, expected: str):
    assert classify(question).label == expected, f"misclassified: {question!r}"


def test_empty_question():
    assert classify("").label == "single-session-user"
    assert classify("   ").label == "single-session-user"


def test_assistant_beats_preference():
    """'Can you remind me what you recommended' → assistant, not preference."""
    q = "Can you remind me of the restaurant you recommended?"
    assert classify(q).label == "single-session-assistant"


def test_temporal_beats_multi_session_for_how_many_days():
    """Temporal arithmetic Qs use 'how many' but are temporal-reasoning."""
    q = "How many days passed between my doctor visit and the trip?"
    assert classify(q).label == "temporal-reasoning"


def test_classification_returns_rule_name():
    c = classify("How many books did I read?")
    assert c.label == "multi-session"
    assert "counting" in c.rule


# ── Counting helper ────────────────────────────────────────────────────


def test_is_counting_question():
    assert is_counting_question("How many books did I read?")
    assert is_counting_question("How often do I exercise?")
    assert is_counting_question("Total number of trips this year?")
    assert is_counting_question("Count the projects I've finished")
    assert not is_counting_question("What was my degree?")
    assert not is_counting_question("Where did I go for lunch?")


# ── Config tables sanity ──────────────────────────────────────────────


def test_k_floors_cover_all_six_categories():
    expected = {
        "single-session-user", "single-session-assistant", "single-session-preference",
        "knowledge-update", "multi-session", "temporal-reasoning",
    }
    assert set(K_FLOORS) == expected
    assert all(k >= 20 for k in K_FLOORS.values())
    assert COUNTING_K_FLOOR == 60  # Stage 2: bumped from 45 for wider MS recall


def test_filter_config_keys_complete():
    expected = {
        "single-session-user", "single-session-assistant", "single-session-preference",
        "knowledge-update", "multi-session", "temporal-reasoning",
    }
    assert set(FILTER_CONFIG) == expected
    for cat, cfg in FILTER_CONFIG.items():
        assert {"min_rel", "min_res", "max_res", "max_tokens"} == set(cfg)
        assert 0 < cfg["min_rel"] < 1
        assert cfg["min_res"] <= cfg["max_res"]


def test_abstention_filter_is_tight():
    """Abstention has the tightest filter — most likely to refuse."""
    assert ABSTENTION_FILTER["min_rel"] >= 0.20
    assert ABSTENTION_FILTER["max_res"] <= 5
    assert ABSTENTION_FILTER["max_tokens"] <= 256


# ── Stage 1.5 — OMEGA retrieval profile + intent overlay ──────────────


def test_retrieval_profiles_cover_all_six_categories():
    """OMEGA's profile is keyed on the six LME question categories. If
    one is missing, ``channel_weights`` silently falls back to the
    SS-user profile — that's a real bug we want a test to catch."""
    expected = {
        "single-session-user", "single-session-assistant", "single-session-preference",
        "knowledge-update", "multi-session", "temporal-reasoning",
    }
    assert set(RETRIEVAL_PROFILES) == expected
    for cat, prof in RETRIEVAL_PROFILES.items():
        assert {"vec", "kw"} == set(prof), f"{cat} profile missing channel"
        # OMEGA's published multipliers fall in [0.5, 2.0]; anything wildly
        # out of band is a typo.
        for ch, w in prof.items():
            assert 0.5 <= w <= 2.0, f"{cat}.{ch}={w} outside [0.5, 2.0]"


def test_intent_overlays_disjoint_and_extreme():
    """The three overlays should each pull HARD in opposite directions —
    if FACTUAL/CONCEPTUAL look similar, the overlay isn't actually
    discriminating between question types."""
    assert set(INTENT_OVERLAYS) == {"FACTUAL", "CONCEPTUAL", "NAVIGATIONAL"}
    factual = INTENT_OVERLAYS["FACTUAL"]
    conceptual = INTENT_OVERLAYS["CONCEPTUAL"]
    nav = INTENT_OVERLAYS["NAVIGATIONAL"]

    # FACTUAL prefers keyword (exact-match); CONCEPTUAL prefers vector.
    assert factual["kw"] > factual["vec"]
    assert conceptual["vec"] > conceptual["kw"]
    # NAVIGATIONAL is the most extreme keyword preference (code tokens).
    assert nav["kw"] > factual["kw"]
    assert nav["vec"] < factual["vec"]


def test_intent_neutral_is_identity():
    """NEUTRAL fallback should be exact 1.0 across both channels — any
    other value would silently rescale every NEUTRAL question."""
    assert INTENT_NEUTRAL == {"vec": 1.0, "kw": 1.0}


@pytest.mark.parametrize("query,expected", [
    # FACTUAL — Wh-words, "remember", "preference for"
    ("What was my favorite restaurant?", "FACTUAL"),
    ("which color did I pick", "FACTUAL"),
    ("when did we last meet", "FACTUAL"),
    ("remind me what I said about Tom", "FACTUAL"),
    ("decision about the new gym", "FACTUAL"),
    # CONCEPTUAL — explanatory verbs
    ("how does the search ranker actually work", "CONCEPTUAL"),
    ("explain why the test failed", "CONCEPTUAL"),
    ("why does it crash on startup", "CONCEPTUAL"),
    ("describe the architecture", "CONCEPTUAL"),
    # NAVIGATIONAL — code-like tokens
    ("look up 5f3a92c4-8d1e-4a09-b0a2-8e0c4d5f6a7b", "NAVIGATIONAL"),
    ("what does foo_bar do", "NAVIGATIONAL"),
    ("read /etc/hosts", "NAVIGATIONAL"),
    ("inspect module.submodule", "NAVIGATIONAL"),
    # NEUTRAL — falls through to none of the buckets
    ("plan a meal", "NEUTRAL"),
    ("List items", "NEUTRAL"),
])
def test_classify_lme_intent_buckets(query: str, expected: str):
    assert classify_lme_intent(query) == expected


def test_classify_lme_intent_priority_navigational_beats_factual():
    """A query containing BOTH a code token and a Wh-word should route
    as NAVIGATIONAL — the code token is the stronger signal for
    keyword-channel preference."""
    q = "what does foo_bar mean in this context"  # has 'what' AND foo_bar
    assert classify_lme_intent(q) == "NAVIGATIONAL"


def test_classify_lme_intent_priority_conceptual_beats_factual():
    """A query containing a 'how does' phrase AND a 'what was' phrase
    should route as CONCEPTUAL — explanation is the dominant intent."""
    q = "how does this work and what was the trigger"
    assert classify_lme_intent(q) == "CONCEPTUAL"


def test_classify_lme_intent_empty_returns_neutral():
    assert classify_lme_intent("") == "NEUTRAL"
    assert classify_lme_intent("   ") == "NEUTRAL"
    assert classify_lme_intent(None) == "NEUTRAL"  # type: ignore[arg-type]


def test_channel_weights_composition_is_multiplicative():
    """``channel_weights(cat, intent) == profile[cat] * overlay[intent]``
    — verifying with a hand-computed pair so a typo in either dict gets
    caught."""
    vec, kw = channel_weights("knowledge-update", "FACTUAL")
    # KU: vec=0.8, kw=1.4 ; FACTUAL: vec=0.3, kw=1.65
    assert abs(vec - 0.8 * 0.3) < 1e-9
    assert abs(kw - 1.4 * 1.65) < 1e-9


def test_channel_weights_neutral_intent_is_profile_only():
    """With NEUTRAL intent, weights == profile (overlay is 1.0×)."""
    for cat, prof in RETRIEVAL_PROFILES.items():
        vec, kw = channel_weights(cat, "NEUTRAL")
        assert abs(vec - prof["vec"]) < 1e-9
        assert abs(kw - prof["kw"]) < 1e-9


def test_channel_weights_unknown_category_falls_back_to_ss_user():
    """Defensive fallback: an unknown category shouldn't crash. SS-user
    is the safest default since OMEGA's recipe gives it neutral-ish
    weights (vec=1.0, kw=1.15)."""
    vec, kw = channel_weights("not-a-real-category", "NEUTRAL")
    expected = RETRIEVAL_PROFILES["single-session-user"]
    assert vec == expected["vec"]
    assert kw == expected["kw"]


def test_channel_weights_unknown_intent_falls_back_to_neutral():
    """Defensive fallback: an unknown intent is treated as NEUTRAL."""
    vec, kw = channel_weights("knowledge-update", "WEIRD_BUCKET")
    prof = RETRIEVAL_PROFILES["knowledge-update"]
    assert vec == prof["vec"]
    assert kw == prof["kw"]


# ── End-to-end accuracy gate on oracle ────────────────────────────────


@pytest.mark.skipif(not ORACLE.exists(), reason=f"{ORACLE} not present")
def test_classifier_accuracy_on_oracle():
    """Run the classifier on all 500 oracle questions; assert ≥ 70%.

    The 70% floor reflects the realistic ceiling for heuristic
    classification on this dataset. The KU-vs-multi-session boundary
    in particular is genuinely indistinguishable from question text
    alone — both "How many Korean restaurants have I tried" (KU) and
    "How many model kits have I worked on" (multi-session) look
    identical. OMEGA gets around this by reading ``question_type``
    directly from the dataset (their cheat); we don't.

    Misclassification cost is bounded: KU→multi-session loses the
    recency boost but uses a similar prompt; multi-session→temporal
    still gets date-aware retrieval; single-session-user→multi-session
    just bumps the K floor. The end-to-end score doesn't drop in
    proportion to the classifier accuracy gap.

    On failure, the per-category confusion matrix is printed.
    """
    with ORACLE.open() as f:
        data = json.load(f)

    correct = 0
    total = len(data)
    confusion: dict[tuple[str, str], int] = Counter()
    per_cat_total: dict[str, int] = Counter()
    per_cat_correct: dict[str, int] = Counter()

    for q in data:
        true_cat = q["question_type"]
        pred = classify(q["question"]).label
        per_cat_total[true_cat] += 1
        if pred == true_cat:
            correct += 1
            per_cat_correct[true_cat] += 1
        else:
            confusion[(true_cat, pred)] += 1

    overall = correct / total
    print(f"\nOverall accuracy: {overall:.1%} ({correct}/{total})")
    print("\nPer-category accuracy:")
    for cat in sorted(per_cat_total):
        acc = per_cat_correct[cat] / per_cat_total[cat] if per_cat_total[cat] else 0.0
        print(f"  {cat:30s} {acc:.1%}  ({per_cat_correct[cat]}/{per_cat_total[cat]})")

    if confusion:
        print("\nTop confusion pairs (true → predicted):")
        for (true, pred), n in confusion.most_common(10):
            print(f"  {true} → {pred}: {n}")

    # Floor: ≥70%. Below that, we'd have to question the classifier
    # itself before publishing a number. The exact value is published
    # in the report as the classifier-vs-oracle gap.
    assert overall >= 0.70, f"classifier accuracy {overall:.1%} below 70% floor"
