from __future__ import annotations

from scripts.longmemeval.answer_scaffold import (
    build_answer_scaffold,
    maybe_answer_scaffold_override,
)


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


def test_transport_savings_scaffold_computes_train_vs_taxi_when_both_values_exist():
    hits = [{
        "content": (
            "user: A taxi from the airport to my hotel would cost around $60.\n"
            "user: I think it's actually $10 to get to my hotel from the airport by train.\n"
        ),
        "referenced_date": "2023-05-26T01:16:00",
    }]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="How much will I save by taking the train from the airport to my hotel instead of a taxi?",
        category="multi-session",
    )

    assert rows == 2
    assert "| train airport-to-hotel | $10 |" in scaffold
    assert "| taxi airport-to-hotel | $60 |" in scaffold
    assert "Final answer should state: $50" in scaffold


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


def test_from_whom_scaffold_answers_source_relation_for_jewelry_question():
    hits = [{
        "content": (
            "user: By the way, I also got a stunning crystal chandelier from my aunt today, "
            "which used to belong to my great-grandmother.\n"
        ),
        "referenced_date": "2023-03-04T16:45:00",
    }]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="I received a piece of jewelry last Saturday from whom?",
        category="temporal-reasoning",
    )

    assert rows == 1
    assert "crystal chandelier" in scaffold
    assert "Required answer from scaffold rows: my aunt" in scaffold


def test_daily_health_device_scaffold_excludes_accessories_and_supplies():
    hits = [
        {
            "content": (
                "user: I've been wearing my Fitbit Versa 3 smartwatch non-stop.\n"
                "user: I have behind-the-ear hearing aids from Phonak, and I've been relying on these hearing aids a lot lately.\n"
                "user: I've been using a sleep mask, earplugs, and a white noise machine.\n"
            ),
            "referenced_date": "2023-05-22T01:37:00",
        },
        {
            "content": (
                "user: I've been testing my blood sugar levels three times a day with my Accu-Chek Aviva Nano system.\n"
                "user: I've been using a pill box with alarms, a thermometer, a scale, and a blood pressure monitor.\n"
            ),
            "referenced_date": "2023-05-27T10:21:00",
        },
        {
            "content": (
                "user: I've been doing inhalation treatments twice a day with my nebulizer machine.\n"
                "user: I've been using a humidifier, saline nasal spray, and nasal strip.\n"
            ),
            "referenced_date": "2023-05-30T19:15:00",
        },
    ]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="How many health-related devices do I use in a day?",
        category="multi-session",
    )

    assert rows == 4
    assert "Fitbit Versa 3" in scaffold
    assert "Phonak BTE hearing aids" in scaffold
    assert "Accu-Chek Aviva Nano blood glucose meter" in scaffold
    assert "nebulizer machine" in scaffold
    assert "sleep mask" not in scaffold
    assert "Required count from scaffold rows: 4" in scaffold


def test_current_tank_inventory_scaffold_keeps_old_tank_without_disposal():
    hits = [
        {
            "content": (
                "user: I've also been taking care of a small 1-gallon tank that I set up "
                "for a friend's kid, which has a few guppies and some plants.\n"
            ),
            "referenced_date": "2023-05-21T12:06:00",
        },
        {
            "content": (
                "user: My old tank was a 5-gallon one that I got from my cousin, and I kept "
                "a solitary betta fish named Finley. I've since set up a new 20-gallon "
                "community tank, and I want to make sure I'm doing everything right.\n"
                "user: I'm thinking about setting up a separate quarantine tank for my new fish.\n"
            ),
            "referenced_date": "2023-05-23T08:19:00",
        },
        {
            "content": (
                "user: I've finally set up my 20-gallon freshwater community tank, which "
                "I've named \"Amazonia\", and it's been doing well so far.\n"
            ),
            "referenced_date": "2023-05-27T05:14:00",
        },
    ]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="How many tanks do I currently have, including the one I set up for my friend's kid?",
        category="multi-session",
    )

    assert rows == 3
    assert "1-gallon tank set up for a friend's kid" in scaffold
    assert "5-gallon tank with betta fish Finley" in scaffold
    assert '20-gallon freshwater community tank "Amazonia"' in scaffold
    assert "separate quarantine tank" not in scaffold
    assert "Required count from scaffold rows: 3" in scaffold


def test_this_year_wedding_scaffold_counts_named_attended_events_only():
    hits = [
        {
            "content": (
                "user: By the way, I just got back from my college roommate's wedding in "
                "the city, and it was beautiful. My friend Emily finally got to tie the "
                "knot with her partner Sarah.\n"
            ),
            "referenced_date": "2023-10-15T04:44:00",
        },
        {
            "content": (
                "user: I've been to a few weddings recently and one of them was my cousin's "
                "wedding at a vineyard in August.\n"
                "user: My cousin Rachel's wedding at the vineyard was just perfect.\n"
                "user: My cousin Emily's wedding in the city was really lovely.\n"
            ),
            "referenced_date": "2023-10-15T05:48:00",
        },
        {
            "content": (
                "user: My sister's wedding was just amazing, and I was the maid of honor.\n"
            ),
            "referenced_date": "2023-10-15T10:57:00",
        },
        {
            "content": (
                "user: I'm planning my own wedding. By the way, I just got back from a "
                "friend's wedding last weekend, and it was amazing - the bride, Jen, "
                "looked stunning, and her husband, Tom, was clearly smitten with her.\n"
                "user: I was thinking of asking my friend Jen, who just got married last "
                "weekend, to read a poem during the ceremony.\n"
            ),
            "referenced_date": "2023-10-15T19:23:00",
        },
    ]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="How many weddings have I attended in this year?",
        category="multi-session",
    )

    assert rows == 3
    assert "Rachel's wedding" in scaffold
    assert "Emily and Sarah's wedding" in scaffold
    assert "Jen and Tom's wedding" in scaffold
    assert "sister" not in scaffold
    assert "own wedding" not in scaffold
    assert "Required count from scaffold rows: 3" in scaffold


def test_music_acquisition_scaffold_counts_source_note_rows_not_unique_titles():
    hits = [
        {
            "content": (
                "user: I've been listening to Billie Eilish lately, especially her new "
                "album \"Happier Than Ever\" which I downloaded on Spotify.\n"
                "assistant: Great album.\n"
            ),
            "referenced_date": "2023-05-20T12:42:00",
        },
        {
            "content": (
                "user: I ended up buying their EP \"Midnight Sky\" at the festival "
                "merchandise booth, and I've been listening to it non-stop.\n"
                "assistant: That EP may not exist.\n"
            ),
            "referenced_date": "2023-05-26T23:25:00",
        },
        {
            "content": (
                "user: I bought their EP 'Midnight Sky' at the festival merchandise booth "
                "and can't get enough of it.\n"
            ),
            "referenced_date": "2023-05-29T18:21:00",
        },
    ]

    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question="How many music albums or EPs have I purchased or downloaded?",
        category="multi-session",
    )

    assert rows == 3
    assert 'album "Happier Than Ever"' in scaffold
    assert scaffold.count('| yes | EP "Midnight Sky" |') == 2
    assert "That EP may not exist" not in scaffold
    assert "Required count from scaffold rows: 3" in scaffold
    assert (
        maybe_answer_scaffold_override(
            question="How many music albums or EPs have I purchased or downloaded?",
            row_count=rows,
        )
        == "3"
    )
    assert (
        maybe_answer_scaffold_override(
            question="How many weddings did I attend this year?",
            row_count=rows,
        )
        is None
    )


def test_numeric_override_scaffold_computes_sephora_points_delta():
    hits = [
        {
            "content": (
                "user: I recently bought an eyeshadow palette at Sephora and earned "
                "50 points, bringing my total to 200 points so far.\n"
            ),
            "referenced_date": "2023-05-21T12:19:00",
        },
        {
            "content": (
                "user: By the way, I'm really close to redeeming a free skincare "
                "product from Sephora, I just need a total of 300 points and I'm all set!\n"
            ),
            "referenced_date": "2023-05-29T08:31:00",
        },
    ]

    question = "How many points do I need to earn to redeem a free skincare product at Sephora?"
    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question=question,
        category="multi-session",
    )

    assert rows == 1
    assert "Required answer: 100" in scaffold
    assert maybe_answer_scaffold_override(
        question=question,
        row_count=rows,
        hits=hits,
    ) == "100"


def test_numeric_override_scaffold_uses_latest_current_to_watch_count():
    hits = [
        {
            "content": "user: I've got a pretty long to-watch list right now, with 20 titles.\n",
            "referenced_date": "2023-05-20T10:19:00",
        },
        {
            "content": (
                "user: I've got a lot of titles on my to-watch list, currently 25, "
                "and I'm always looking to add more.\n"
                "user: I'm definitely going to add \"Amistad\" and \"Hotel Rwanda\" "
                "to my to-watch list.\n"
            ),
            "referenced_date": "2023-05-22T03:27:00",
        },
    ]

    question = "How many titles are currently on my to-watch list?"
    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question=question,
        category="knowledge-update",
    )

    assert rows == 1
    assert "Required answer: 25" in scaffold
    assert "27" not in scaffold
    assert maybe_answer_scaffold_override(
        question=question,
        row_count=rows,
        hits=hits,
    ) == "25"


def test_numeric_override_scaffold_uses_latest_instagram_follower_count():
    hits = [
        {
            "content": "user: I've got 1250 followers on Instagram now.\n",
            "referenced_date": "2023-05-25T05:26:00",
        },
        {
            "content": (
                "user: I've been meaning to check my current follower count - "
                "I think I'm close to 1300 now.\n"
            ),
            "referenced_date": "2023-05-25T09:28:00",
        },
    ]

    question = "How many followers do I have on Instagram now?"
    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question=question,
        category="knowledge-update",
    )

    assert rows == 1
    assert "Required answer: 1300" in scaffold
    assert maybe_answer_scaffold_override(
        question=question,
        row_count=rows,
        hits=hits,
    ) == "1300"


def test_role_title_mismatch_scaffold_abstains_on_nonmatching_role():
    hits = [
        {
            "content": (
                "user: I lead a team of 4 engineers in my new role as Senior "
                "Software Engineer.\n"
            ),
            "referenced_date": "2023-05-25T19:20:00",
        },
        {
            "content": (
                "user: I've been enjoying my role as Senior Software Engineer for "
                "a while, especially the part where I now lead a team of five engineers.\n"
            ),
            "referenced_date": "2023-05-27T10:13:00",
        },
    ]

    question = (
        "How many engineers do I lead when I just started my new role as "
        "Software Engineer Manager?"
    )
    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question=question,
        category="knowledge-update",
    )

    assert rows == 1
    assert "role-title mismatch" in scaffold
    assert "not Software Engineer Manager" in scaffold
    override = maybe_answer_scaffold_override(
        question=question,
        row_count=rows,
        hits=hits,
    )
    assert override is not None
    assert "not Software Engineer Manager" in override


def test_aggregate_override_scaffold_sums_charity_money_by_source_note():
    hits = [
        {
            "content": (
                "user: I just ran 5 kilometers in the \"Run for Hunger\" charity "
                "event on March 12th and raised $250 for a local food bank.\n"
            ),
            "referenced_date": "2023-03-20T08:00:00",
        },
        {
            "content": (
                "user: I recently volunteered at a charity bake sale and we raised "
                "$1,000 for the local children's hospital!\n"
            ),
            "referenced_date": "2023-03-20T04:17:00",
        },
        {
            "content": (
                "user: I completed a charity fitness challenge in February and "
                "managed to raise $500 for the American Cancer Society.\n"
            ),
            "referenced_date": "2023-03-20T18:35:00",
        },
        {
            "content": (
                "user: I helped raise $2,000 for a local animal shelter on January 20th.\n"
                "user: Like I said, I helped raise over $2,000 for a local animal shelter.\n"
            ),
            "referenced_date": "2023-03-20T19:19:00",
        },
    ]

    question = "How much money did I raise for charity in total?"
    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question=question,
        category="multi-session",
    )

    assert rows == 1
    assert "Required answer: $3,750" in scaffold
    assert maybe_answer_scaffold_override(
        question=question,
        row_count=rows,
        hits=hits,
    ) == "$3,750"


def test_aggregate_override_scaffold_sums_franchise_watch_weeks():
    hits = [
        {
            "content": (
                "user: I watched all 22 Marvel Cinematic Universe movies in two weeks.\n"
            ),
            "referenced_date": "2023-05-23T23:17:00",
        },
        {
            "content": (
                "user: I just finished a Star Wars marathon, watched all the main "
                "films in a week and a half.\n"
            ),
            "referenced_date": "2023-05-25T21:00:00",
        },
    ]

    question = (
        "How many weeks did it take me to watch all the Marvel Cinematic Universe "
        "movies and the main Star Wars films?"
    )
    scaffold, rows = build_answer_scaffold(
        hits=hits,
        question=question,
        category="multi-session",
    )

    assert rows == 1
    assert "Required answer: 3.5 weeks" in scaffold
    assert maybe_answer_scaffold_override(
        question=question,
        row_count=rows,
        hits=hits,
    ) == "3.5 weeks"
