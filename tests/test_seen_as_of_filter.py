"""Bi-temporal ingestion-time filter tests.

A3.3 of the v1.1 roadmap. Validates that:
  * ``filter_by_seen_as_of`` keeps only records whose ``t_created`` ≤ X
    and ``t_expired`` is NULL or > X.
  * ``filter_by_date`` (event-time) and ``filter_by_seen_as_of``
    (ingestion-time) compose without interfering.
  * ``scored_search`` accepts and applies ``seen_as_of`` end-to-end.
"""
from __future__ import annotations

from jarvis_memory.scoring import _apply_filters, scored_search
from jarvis_memory.temporal import filter_by_date, filter_by_seen_as_of


def _make_record(uuid: str, **kwargs) -> dict:
    return {"uuid": uuid, **kwargs}


# ── filter_by_seen_as_of in isolation ──────────────────────────────────


def test_seen_as_of_excludes_records_ingested_after_target():
    records = [
        _make_record("a", t_created="2026-01-01T00:00:00+00:00"),
        _make_record("b", t_created="2026-04-15T00:00:00+00:00"),
        _make_record("c", t_created="2026-04-25T00:00:00+00:00"),
    ]
    out = filter_by_seen_as_of(records, "2026-04-20T00:00:00+00:00")
    uuids = [r["uuid"] for r in out]
    assert "a" in uuids, "a was ingested before target — must remain"
    assert "b" in uuids, "b was ingested before target — must remain"
    assert "c" not in uuids, "c was ingested AFTER target — must be excluded"


def test_seen_as_of_excludes_records_expired_before_target():
    records = [
        _make_record(
            "old",
            t_created="2026-01-01T00:00:00+00:00",
            t_expired="2026-03-15T00:00:00+00:00",
        ),
        _make_record("current", t_created="2026-01-01T00:00:00+00:00"),
    ]
    out = filter_by_seen_as_of(records, "2026-04-20T00:00:00+00:00")
    uuids = [r["uuid"] for r in out]
    assert "old" not in uuids, "old expired before target — system no longer believed"
    assert "current" in uuids, "current never expired — still believed"


def test_seen_as_of_keeps_record_at_exact_t_created():
    """Boundary: a record ingested at exactly the target moment counts as believed."""
    records = [
        _make_record("boundary", t_created="2026-04-20T00:00:00+00:00"),
    ]
    out = filter_by_seen_as_of(records, "2026-04-20T00:00:00+00:00")
    assert len(out) == 1, "ingestion-time start is inclusive"


def test_seen_as_of_falls_back_to_created_at_for_legacy_data():
    """Pre-A3.1 data has no t_created. Filter should fall back to created_at."""
    records = [
        _make_record("legacy_old", created_at="2026-01-01T00:00:00+00:00"),
        _make_record("legacy_new", created_at="2026-04-25T00:00:00+00:00"),
    ]
    out = filter_by_seen_as_of(records, "2026-04-20T00:00:00+00:00")
    uuids = [r["uuid"] for r in out]
    assert "legacy_old" in uuids
    assert "legacy_new" not in uuids


def test_seen_as_of_invalid_date_returns_unfiltered():
    records = [_make_record("a", t_created="2026-01-01T00:00:00+00:00")]
    out = filter_by_seen_as_of(records, "not-a-date")
    assert out == records, "invalid input should fall through to no-op"


# ── Composition with event-time filter ─────────────────────────────────


def test_event_time_and_ingestion_time_compose_independently():
    """A record should pass both filters or neither — they're orthogonal axes."""
    records = [
        # event-time: ended Mar 15 (was true Jan 1 - Mar 15)
        # ingestion-time: ended Apr 1 (we believed it Jan 1 - Apr 1)
        _make_record(
            "stopped_then_disbelieved",
            valid_from="2026-01-01T00:00:00+00:00",
            valid_to="2026-03-15T00:00:00+00:00",
            t_created="2026-01-01T00:00:00+00:00",
            t_expired="2026-04-01T00:00:00+00:00",
        ),
        # event-time + ingestion-time both still open
        _make_record(
            "still_true_still_believed",
            valid_from="2026-01-01T00:00:00+00:00",
            t_created="2026-01-01T00:00:00+00:00",
        ),
    ]

    # "What did we believe on 2026-03-20 about the world on 2026-02-01?"
    # On 2026-02-01: stopped_then_disbelieved was still true (event ended Mar 15)
    # On 2026-03-20: we still believed it (ingestion ended Apr 1)
    # → both records keep
    out = filter_by_seen_as_of(
        filter_by_date(records, as_of="2026-02-01T00:00:00+00:00"),
        seen_as_of="2026-03-20T00:00:00+00:00",
    )
    uuids = {r["uuid"] for r in out}
    assert uuids == {"stopped_then_disbelieved", "still_true_still_believed"}

    # "What did we believe on 2026-04-15 about the world on 2026-04-15?"
    # On 2026-04-15: stopped_then_disbelieved no longer true (ended Mar 15)
    #                AND we no longer believed it (ingestion ended Apr 1)
    # → only still_true_still_believed
    out = filter_by_seen_as_of(
        filter_by_date(records, as_of="2026-04-15T00:00:00+00:00"),
        seen_as_of="2026-04-15T00:00:00+00:00",
    )
    uuids = {r["uuid"] for r in out}
    assert uuids == {"still_true_still_believed"}


# ── _apply_filters integration ─────────────────────────────────────────


def test_apply_filters_passes_seen_as_of_through():
    """Internal entrypoint must honor the new param."""
    records = [
        _make_record("ingested_early", t_created="2026-01-01T00:00:00+00:00"),
        _make_record("ingested_late", t_created="2026-04-25T00:00:00+00:00"),
    ]
    out = _apply_filters(
        records,
        group_id=None,
        room=None,
        hall=None,
        memory_type=None,
        as_of=None,
        seen_as_of="2026-04-20T00:00:00+00:00",
    )
    uuids = [r["uuid"] for r in out]
    assert uuids == ["ingested_early"]


def test_apply_filters_default_seen_as_of_is_none():
    """``seen_as_of`` is optional — omitted callers must keep working."""
    records = [_make_record("x", t_created="2026-01-01T00:00:00+00:00")]
    out = _apply_filters(
        records,
        group_id=None,
        room=None,
        hall=None,
        memory_type=None,
        as_of=None,
    )
    assert len(out) == 1


# ── End-to-end through scored_search ───────────────────────────────────


def test_scored_search_accepts_seen_as_of_param():
    """scored_search must accept seen_as_of without raising and pass it
    through. We don't need a live DB — the empty-rankings path falls
    through to the legacy composite, which we've already taught."""

    # Empty vector_search_fn → no rankings → legacy fallback path with
    # an empty result set. Test purpose: confirm the kwarg is accepted
    # and the function returns a list.
    out = scored_search(
        "test query",
        seen_as_of="2026-04-20T00:00:00+00:00",
        vector_search_fn=lambda q, n: [],
    )
    assert isinstance(out, list)
