from __future__ import annotations

from scripts.longmemeval.evidence_ledger import (
    build_evidence_ledger,
    parse_role_segments,
    should_use_evidence_ledger,
)


def test_parse_role_segments_keeps_multiline_turns():
    segments = parse_role_segments(
        "user: I spent $75 at SaveMart\n"
        "last Thursday.\n"
        "assistant: With 1% cashback, that is $0.75.\n"
    )

    assert len(segments) == 2
    assert segments[0].role == "user"
    assert "last Thursday" in segments[0].text
    assert segments[1].role == "assistant"


def test_build_evidence_ledger_selects_matching_turn_and_adjacent_answer():
    hits = [{
        "content": (
            "user: I spent $75 on groceries at SaveMart last Thursday.\n"
            "assistant: With your 1% cashback, you earned $0.75.\n"
            "user: I also bought a blue sweater yesterday.\n"
            "assistant: Nice find.\n"
        ),
        "referenced_date": "2023-05-25T12:00:00",
    }]

    ledger, n_lines = build_evidence_ledger(
        hits=hits,
        question="How much cashback did I earn at SaveMart last Thursday?",
        category="multi-session",
    )

    assert n_lines >= 2
    assert "[Note 1 | Date: 2023-05-25T12:00:00 | Evidence ledger]" in ledger
    assert "SaveMart last Thursday" in ledger
    assert "$0.75" in ledger


def test_build_evidence_ledger_is_limited_to_high_context_categories():
    hits = [{"content": "user: I like stand-up comedy on Netflix.", "referenced_date": ""}]

    ledger, n_lines = build_evidence_ledger(
        hits=hits,
        question="Can you recommend a show for me?",
        category="single-session-preference",
    )

    assert ledger == ""
    assert n_lines == 0
    assert should_use_evidence_ledger("multi-session")
    assert should_use_evidence_ledger("temporal-reasoning")
    assert should_use_evidence_ledger("knowledge-update")
    assert not should_use_evidence_ledger("single-session-user")


def test_build_evidence_ledger_caps_total_lines():
    hits = []
    for idx in range(10):
        hits.append({
            "content": (
                f"user: I attended charity event {idx} before Run for the Cure.\n"
                f"assistant: Event {idx} counts as a charity event.\n"
            ),
            "referenced_date": f"2023-05-{idx + 1:02d}T09:00:00",
        })

    ledger, n_lines = build_evidence_ledger(
        hits=hits,
        question="How many charity events did I participate in before the Run for the Cure event?",
        category="temporal-reasoning",
        max_total_lines=5,
    )

    assert n_lines == 5
    assert "charity event" in ledger


def test_build_evidence_ledger_focuses_buried_by_the_way_fact():
    hits = [
        {
            "content": (
                "user: Can you help me organize my closet and categorize my clothes? "
                "Also, by the way, I still need to pick up my dry cleaning for "
                "the navy blue blazer I wore to a meeting.\n"
                "assistant: Sure, organize by category.\n"
            ),
            "referenced_date": "2023-02-15T06:30:00",
        },
        {
            "content": (
                "user: How should I handle e-commerce returns and refunds for my store?\n"
                "assistant: Returns and refunds are important for customer service.\n"
            ),
            "referenced_date": "2023-02-15T13:21:00",
        },
    ]

    ledger, _ = build_evidence_ledger(
        hits=hits,
        question="How many items of clothing do I need to pick up or return from a store?",
        category="multi-session",
        max_total_lines=8,
    )

    assert "navy blue blazer" in ledger
    assert "e-commerce returns" not in ledger


def test_build_evidence_ledger_surfaces_frequency_updates():
    hits = [
        {
            "content": (
                "user: I go to the gym on Tuesdays, Thursdays, and Saturdays.\n"
                "assistant: Great routine.\n"
            ),
            "referenced_date": "2023-06-01T09:48:00",
        },
        {
            "content": (
                "user: I've been consistent with my gym routine - four times a week, actually.\n"
                "assistant: That's great consistency.\n"
            ),
            "referenced_date": "2023-08-15T20:17:00",
        },
    ]

    ledger, _ = build_evidence_ledger(
        hits=hits,
        question="Do I go to the gym more frequently than I did previously?",
        category="knowledge-update",
    )

    assert "Tuesdays, Thursdays, and Saturdays" in ledger
    assert "four times a week" in ledger


def test_build_evidence_ledger_expands_business_milestone_language():
    hits = [{
        "content": (
            "user: I'm looking for advice on creating a solid contract for my "
            "freelance clients. I just signed a contract with my first client today.\n"
            "assistant: Congratulations on landing your first client.\n"
        ),
        "referenced_date": "2023-03-01T02:43:00",
    }]

    ledger, _ = build_evidence_ledger(
        hits=hits,
        question="What was the significant business milestone I mentioned four weeks ago?",
        category="temporal-reasoning",
    )

    assert "signed a contract with my first client" in ledger


def test_build_evidence_ledger_rejects_clothing_without_pickup_action():
    hits = [{
        "content": "user: Give me company name ideas for a streetwear clothing brand.\n",
        "referenced_date": "2023-02-15T23:17:00",
    }]

    ledger, n_lines = build_evidence_ledger(
        hits=hits,
        question="How many items of clothing do I need to pick up or return from a store?",
        category="multi-session",
    )

    assert ledger == ""
    assert n_lines == 0


def test_build_evidence_ledger_rejects_unrelated_cashback_program():
    hits = [{
        "content": (
            "assistant: Walmart has 2% cashback on online grocery purchases.\n"
            "assistant: You earn $2 cashback on a $100 subtotal.\n"
        ),
        "referenced_date": "2023-05-26T14:39:00",
    }]

    ledger, n_lines = build_evidence_ledger(
        hits=hits,
        question="How much cashback did I earn at SaveMart last Thursday?",
        category="multi-session",
    )

    assert ledger == ""
    assert n_lines == 0


def test_build_evidence_ledger_rejects_assistant_generic_bus_fare_for_savings():
    hits = [{
        "content": (
            "user: I was told a taxi from the airport to my hotel would cost around $60.\n"
            "assistant: The Airport Limousine Bus fare is around $10-$20 depending on the route.\n"
        ),
        "referenced_date": "2023-04-12T08:00:00",
    }]

    ledger, _ = build_evidence_ledger(
        hits=hits,
        question="How much will I save by taking the bus from the airport to my hotel instead of a taxi?",
        category="multi-session",
    )

    assert "taxi from the airport to my hotel" in ledger
    assert "Airport Limousine Bus fare" not in ledger


def test_build_evidence_ledger_museum_order_filters_non_museum_events():
    hits = [
        {
            "content": (
                "user: I attended a photography workshop last month and liked it.\n"
                "assistant: Great workshop.\n"
            ),
            "referenced_date": "2023-01-19T06:30:00",
        },
        {
            "content": (
                "user: I attended a lecture series at the Museum of Contemporary Art recently.\n"
                "assistant: That sounds thought-provoking.\n"
            ),
            "referenced_date": "2023-01-22T20:21:00",
        },
        {
            "content": (
                "user: By the way, I participated in a guided tour there on February 17th.\n"
                "user: I later remembered it was the Modern Art Gallery.\n"
                "assistant: Nice gallery visit.\n"
            ),
            "referenced_date": "2023-02-20T06:37:00",
        },
    ]

    ledger, _ = build_evidence_ledger(
        hits=hits,
        question="What is the order of the six museums I visited from earliest to latest?",
        category="temporal-reasoning",
    )

    assert "Museum of Contemporary Art" in ledger
    assert "photography workshop" not in ledger
    assert "guided tour there" not in ledger
    assert "Modern Art Gallery" not in ledger
