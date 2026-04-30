from __future__ import annotations

from scripts.longmemeval.answer_scaffold import build_answer_scaffold


def test_pickup_return_scaffold_counts_different_obligations_separately():
    hits = [
        {
            "content": (
                "user: I still need to pick up my dry cleaning for the navy blue blazer.\n"
                "assistant: Good reminder.\n"
            ),
            "referenced_date": "2023-02-15T06:30:00",
        },
        {
            "content": (
                "user: I need to return some boots to Zara, actually. I got them on "
                "February 5th, but they were too small, so I exchanged them for a "
                "larger size. I just haven't had a chance to pick them up yet.\n"
            ),
            "referenced_date": "2023-02-15T11:13:00",
        },
        {
            "content": (
                "user: I just exchanged a pair of boots I got from Zara on 2/5, and "
                "I still need to pick up the new pair.\n"
            ),
            "referenced_date": "2023-02-15T16:19:00",
        },
    ]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="How many items of clothing do I need to pick up or return from a store?",
        category="multi-session",
    )

    assert rows == 3
    assert "Required count from scaffold rows: 3" in scaffold
    assert '"Total: 3"' in scaffold
    assert "| yes | return | Zara boots |" in scaffold
    assert "| yes | pickup | new larger Zara boots |" in scaffold


def test_bus_taxi_scaffold_marks_missing_bus_user_price():
    hits = [{
        "content": (
            "user: I was told that taking a taxi from the airport to my hotel would cost around $60.\n"
            "assistant: The Airport Limousine Bus fare is around $10-$20 depending on the route.\n"
            "user: I think I got the price from my friend wrong, yeah it's actually $10 "
            "to get to my hotel from the airport by train.\n"
        ),
        "referenced_date": "2023-05-20T15:31:00",
    }]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="How much will I save by taking the bus from the airport to my hotel instead of a taxi?",
        category="multi-session",
    )

    assert rows == 3
    assert "| taxi airport-to-hotel | $60 |" in scaffold
    assert "| bus airport-to-hotel | MISSING |" in scaffold
    assert "nearby non-answer: train" in scaffold
    assert "not enough information" in scaffold
    assert "$10-$20" not in scaffold


def test_museum_order_scaffold_extracts_venues_and_skips_gallery_only_rows():
    hits = [
        {
            "content": (
                "user: I visited the Science Museum's \"Space Exploration\" exhibition today. "
                "I actually attended a lectures series at the Museum of Contemporary Art recently.\n"
            ),
            "referenced_date": "2023-01-15T16:31:00",
        },
        {
            "content": (
                "user: By the way, I saw it in person today at the Metropolitan Museum of Art's "
                "\"Ancient Egyptian Artifacts\" exhibition.\n"
            ),
            "referenced_date": "2023-02-10T22:26:00",
        },
        {
            "content": (
                "user: By the way, I participated in a guided tour there on February 17th. "
                "I later remembered it was the Modern Art Gallery.\n"
            ),
            "referenced_date": "2023-02-20T06:37:00",
        },
        {
            "content": (
                "user: I'm planning to visit the Modern Art Museum again soon. By the way, "
                "I attended their guided tour of \"The Evolution of Abstract Expressionism\" today.\n"
            ),
            "referenced_date": "2023-02-20T22:50:00",
        },
        {
            "content": (
                "user: I took my niece to the Natural History Museum to see the "
                "\"Dinosaur Fossils\" exhibition today.\n"
            ),
            "referenced_date": "2023-03-04T19:42:00",
        },
    ]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="What is the order of the six museums I visited from earliest to latest?",
        category="temporal-reasoning",
    )

    assert rows == 5
    assert "Science Museum, Museum of Contemporary Art, Metropolitan Museum of Art" in scaffold
    assert "Modern Art Museum, Natural History Museum" in scaffold
    assert "Modern Art Gallery" not in scaffold


def test_museum_order_scaffold_sorts_today_before_recently_on_same_note():
    hits = [{
        "content": (
            "user: I visited the Science Museum's \"Space Exploration\" exhibition today.\n"
            "assistant: Nice visit.\n"
            "user: I attended a lecture series at the Museum of Contemporary Art recently.\n"
        ),
        "referenced_date": "2023-01-15T16:31:00",
    }]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="What is the order of the six museums I visited from earliest to latest?",
        category="temporal-reasoning",
    )

    assert rows == 2
    assert "Required order from scaffold rows: Science Museum, Museum of Contemporary Art" in scaffold
