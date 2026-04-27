"""wake_up.generate_layer1 must order rooms by relevance, not alphabetically.

Regression guard for the room-ordering bug where ``sorted(rooms.items())``
sorted alphabetically by room name. Highest-scored memories belong at the
top of the context buffer (LLMs read top first); alphabetical ordering
silently demoted them.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from jarvis_memory.wake_up import generate_layer1


class _FakeStore:
    """Minimal EmbeddingStore stand-in. Returns canned hits in fixed order."""

    def __init__(self, hits):
        self._hits = hits

    def health_check(self):
        return True

    def search(self, query, limit, where_filter=None):
        return self._hits


def _make_driver(content_map):
    """Build a MagicMock Neo4j driver that satisfies _fetch_content's call shape."""
    records = [
        {"uuid": uid, "text": text}
        for uid, text in content_map.items()
    ]
    db_session = MagicMock()
    db_session.run.return_value = iter(records)
    db_session.__enter__ = MagicMock(return_value=db_session)
    db_session.__exit__ = MagicMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = db_session
    return driver


def test_rooms_ordered_by_max_score_not_alphabetically():
    """Room containing the highest-scored item must appear first.

    Setup: two rooms, ``zeta`` (alphabetically last) holds the highest-scored
    item; ``alpha`` (alphabetically first) holds a lower-scored item. Under
    the old alphabetical sort, ``alpha`` would print first — which is exactly
    the bug.
    """
    hits = [
        {
            "id": "uuid-zeta-high",
            "similarity": 0.95,
            "metadata": {
                "room": "zeta",
                "hall": "decisions",
                "memory_type": "decision",
                "created_at": "2026-04-26T00:00:00+00:00",
            },
        },
        {
            "id": "uuid-alpha-low",
            "similarity": 0.40,
            "metadata": {
                "room": "alpha",
                "hall": "context",
                "memory_type": "fact",
                "created_at": "2026-04-26T00:00:00+00:00",
            },
        },
    ]
    content = {
        "uuid-zeta-high": "Picked Clerk over Auth0 for Navi.",
        "uuid-alpha-low": "Auth provider shortlist refreshed.",
    }
    driver = _make_driver(content)

    output = generate_layer1(
        store=_FakeStore(hits),
        group_id="navi",
        driver=driver,
    )

    zeta_pos = output.find("**zeta:**")
    alpha_pos = output.find("**alpha:**")

    assert zeta_pos != -1, f"expected zeta room in output, got:\n{output}"
    assert alpha_pos != -1, f"expected alpha room in output, got:\n{output}"
    assert zeta_pos < alpha_pos, (
        f"high-score room 'zeta' should appear before low-score room 'alpha', "
        f"but found zeta at {zeta_pos}, alpha at {alpha_pos}.\nOutput:\n{output}"
    )


def test_three_rooms_strict_score_ordering():
    """With three rooms, output order must match descending max-score per room."""
    hits = [
        {
            "id": "uid-mid",
            "similarity": 0.80,
            "metadata": {"room": "billing", "hall": "context", "memory_type": "fact"},
        },
        {
            "id": "uid-high",
            "similarity": 0.95,
            "metadata": {"room": "deals", "hall": "decisions", "memory_type": "decision"},
        },
        {
            "id": "uid-low",
            "similarity": 0.45,
            "metadata": {"room": "auth", "hall": "context", "memory_type": "fact"},
        },
    ]
    content = {
        "uid-mid": "Stripe webhook retry policy set.",
        "uid-high": "Closed lead with Foundry partner.",
        "uid-low": "Auth provider config reviewed.",
    }
    driver = _make_driver(content)

    output = generate_layer1(
        store=_FakeStore(hits),
        group_id="catalyst",
        driver=driver,
    )

    deals_pos = output.find("**deals:**")
    billing_pos = output.find("**billing:**")
    auth_pos = output.find("**auth:**")

    assert deals_pos < billing_pos < auth_pos, (
        f"expected deals(0.95) < billing(0.80) < auth(0.45), "
        f"got deals={deals_pos}, billing={billing_pos}, auth={auth_pos}.\n"
        f"Output:\n{output}"
    )
