"""Unit tests for Phase 8 observation retrieval (adapter helpers).

Pure tests — Chroma is mocked. The actual end-to-end vector recall is
exercised by the smoke test in scripts/, but the filter-shape and
formatting contract live here so they don't drift silently.
"""
from __future__ import annotations

import pytest


# ── retrieve_observations ───────────────────────────────────────────────


class _FakeChromaCollection:
    """Tiny stand-in for chromadb.Collection.

    Records the kwargs of the last query() call and returns whatever the
    test scripted via ``returns``. ``raise_on_query`` simulates a Chroma
    failure (e.g. corrupt index) so we can verify the graceful-skip path.
    """

    def __init__(self, returns: dict | None = None, raise_on_query: bool = False):
        self.returns = returns or {
            "ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]
        }
        self.raise_on_query = raise_on_query
        self.last_kwargs: dict = {}

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        if self.raise_on_query:
            raise RuntimeError("simulated Chroma failure")
        return self.returns


class TestRetrieveObservations:
    def test_returns_ranked_observations(self):
        from scripts.run_longmemeval import retrieve_observations

        col = _FakeChromaCollection(returns={
            "ids": [["o1", "o2", "o3"]],
            "documents": [[
                "- fact: user_age = 32",
                "- event: yoga_class = 5th class [2023-06-12]",
                "- update: 5K_PB = 27:45 → 26:30 [2023-07-30]",
            ]],
            # Chroma cosine distances; lower = better. similarity = 1 - d.
            "distances": [[0.10, 0.30, 0.45]],
            "metadatas": [[
                {"obs_type": "fact", "obs_key": "user_age",
                 "referenced_date": "2023-04-11",
                 "source_episode_uid": "ep_a", "group_id": "lme_q1"},
                {"obs_type": "event", "obs_key": "yoga_class",
                 "referenced_date": "2023-06-12",
                 "source_episode_uid": "ep_b", "group_id": "lme_q1"},
                {"obs_type": "update", "obs_key": "5K_PB",
                 "referenced_date": "2023-07-30",
                 "source_episode_uid": "ep_c", "group_id": "lme_q1"},
            ]],
        })
        out = retrieve_observations(
            chroma_collection=col,
            query="what is the user's age?",
            group_id="lme_q1",
            top_k=3,
        )
        assert len(out) == 3
        # Order preserved (Chroma already returns sorted by distance asc).
        assert out[0]["obs_key"] == "user_age"
        # similarity = 1 - distance
        assert out[0]["similarity"] == pytest.approx(0.90, abs=1e-6)
        assert out[1]["similarity"] == pytest.approx(0.70, abs=1e-6)
        assert out[2]["similarity"] == pytest.approx(0.55, abs=1e-6)
        # Content carries the rendered observation line for prompt use.
        assert out[0]["content"].startswith("- fact: user_age")

    def test_filters_by_group_and_observation_type(self):
        """The where-clause MUST scope by both group and memory_type."""
        from scripts.run_longmemeval import retrieve_observations

        col = _FakeChromaCollection()
        retrieve_observations(
            chroma_collection=col,
            query="anything",
            group_id="lme_q42",
            top_k=5,
        )
        where = col.last_kwargs.get("where") or {}
        # Chroma requires $and at the top level when combining filters.
        assert "$and" in where
        clauses = where["$and"]
        assert {"group_id": {"$eq": "lme_q42"}} in clauses
        assert {"memory_type": {"$eq": "observation"}} in clauses

    def test_caps_n_results_at_100(self):
        """top_k=500 should NOT actually request 500 from Chroma."""
        from scripts.run_longmemeval import retrieve_observations

        col = _FakeChromaCollection()
        retrieve_observations(
            chroma_collection=col, query="q", group_id="lme_q1", top_k=500,
        )
        assert col.last_kwargs.get("n_results") == 100

    def test_empty_query_returns_empty_no_call(self):
        from scripts.run_longmemeval import retrieve_observations

        col = _FakeChromaCollection()
        out = retrieve_observations(
            chroma_collection=col, query="", group_id="lme_q1", top_k=5,
        )
        assert out == []
        assert col.last_kwargs == {}

    def test_zero_top_k_returns_empty_no_call(self):
        from scripts.run_longmemeval import retrieve_observations

        col = _FakeChromaCollection()
        out = retrieve_observations(
            chroma_collection=col, query="q", group_id="lme_q1", top_k=0,
        )
        assert out == []
        assert col.last_kwargs == {}

    def test_chroma_failure_returns_empty_not_raise(self):
        """Phase 8 is additive — retrieval must fail gracefully."""
        from scripts.run_longmemeval import retrieve_observations

        col = _FakeChromaCollection(raise_on_query=True)
        out = retrieve_observations(
            chroma_collection=col, query="q", group_id="lme_q1", top_k=5,
        )
        assert out == []  # No exception raised

    def test_handles_missing_metadata_fields(self):
        """Real Chroma returns None for missing metadatas; don't crash."""
        from scripts.run_longmemeval import retrieve_observations

        col = _FakeChromaCollection(returns={
            "ids": [["o1"]],
            "documents": [["- fact: k = v"]],
            "distances": [[0.2]],
            "metadatas": [[None]],
        })
        out = retrieve_observations(
            chroma_collection=col, query="q", group_id="lme_q1", top_k=5,
        )
        assert len(out) == 1
        assert out[0]["obs_type"] == ""  # default for missing
        assert out[0]["obs_key"] == ""


# ── format_observations_block ───────────────────────────────────────────


class TestFormatObservationsBlock:
    def test_renders_compact_block(self):
        from scripts.run_longmemeval import format_observations_block

        block = format_observations_block([
            {"content": "- fact: user_age = 32 [2023-04-11]"},
            {"content": "- event: yoga_class = 5th class [2023-06-12]"},
            {"content": "- update: 5K_PB = 27:45 → 26:30 [2023-07-30]"},
        ])
        assert block.startswith("[Structured evidence")
        assert block.endswith("[End structured evidence]")
        # All three observations preserved verbatim
        assert "user_age" in block
        assert "yoga_class" in block
        assert "5K_PB" in block

    def test_empty_list_returns_empty_string(self):
        from scripts.run_longmemeval import format_observations_block

        assert format_observations_block([]) == ""

    def test_skips_empty_content(self):
        from scripts.run_longmemeval import format_observations_block

        block = format_observations_block([
            {"content": ""},
            {"content": "   "},
            {"content": "- fact: k = v"},
        ])
        # The valid entry survives, empties are dropped
        assert "- fact: k = v" in block
        # Block starts/ends with the brackets; only one observation line in middle
        body = block.split("\n")
        assert any("- fact: k = v" in ln for ln in body)
        # No double-blank lines from the dropped empties
        assert "\n\n\n" not in block

    def test_all_empty_content_returns_empty_string(self):
        from scripts.run_longmemeval import format_observations_block

        assert format_observations_block([{"content": ""}, {"content": "  "}]) == ""


# ── Cross-channel isolation (regression guard) ──────────────────────────


class TestRawSessionChannelExcludesObservations:
    """When Phase 8 is on, raw-session retrieval and observation retrieval
    share a Chroma collection. The ONLY thing that keeps observations
    out of [Note N] blocks is the `memory_type` filter. This test pins
    that the filter shape is correct — if it ever drops back to
    `where={"group_id": group_id}` only, observations will pollute the
    raw-session results and the 104q baseline will silently regress.
    """

    def test_vector_search_fn_filters_by_session_summary_type(self):
        """The retrieve_with_omega_recipe inner closure MUST scope by
        memory_type. We verify by spying on chroma_collection.query."""
        from scripts.run_longmemeval import retrieve_with_omega_recipe

        # Chroma stub that records every call's where-clause.
        class _SpyCollection:
            def __init__(self):
                self.where_clauses: list = []

            def query(self, **kwargs):
                self.where_clauses.append(kwargs.get("where"))
                return {
                    "ids": [[]], "documents": [[]],
                    "distances": [[]], "metadatas": [[]],
                }

        # Driver stub — retrieve_with_omega_recipe also runs Cypher;
        # return empty rows so it falls through to the vector channel.
        class _NoopSession:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def run(self, *_a, **_k):
                return iter([])

        class _NoopDriver:
            def session(self): return _NoopSession()

        # EmbeddingStore stub — never used since vec channel is fed by
        # vector_search_fn, but scored_search expects something with a
        # search() shape. Return empty.
        class _NoopEmbed:
            def search(self, *_a, **_k): return []
            def health_check(self): return True

        spy = _SpyCollection()
        # This will hit several internal calls; we just want to confirm
        # every where-clause includes memory_type=session_summary.
        try:
            retrieve_with_omega_recipe(
                query="some test query",
                group_id="lme_test_q",
                category="single-session-user",
                counting=False,
                driver=_NoopDriver(),
                embedding_store=_NoopEmbed(),
                chroma_collection=spy,
                question_date="2023-06-12",
            )
        except Exception:
            # We don't care if the full pipeline succeeds with empty
            # data — only that the where-clause was correct on every
            # vector-channel call we made.
            pass

        # At least one query was made
        assert spy.where_clauses, "no chroma queries were issued"
        # Every query must scope to session_summary type
        for where in spy.where_clauses:
            assert where is not None, "missing where clause"
            # Accept either compound shape with $eq, OR a future shape
            # that explicitly excludes observation type.
            if "$and" in where:
                clauses = where["$and"]
                assert {"memory_type": {"$eq": "session_summary"}} in clauses, (
                    f"raw-session channel missing memory_type filter: {where}"
                )
            else:
                # If filter is bare-key shorthand, must include memory_type
                assert where.get("memory_type") == "session_summary", (
                    f"raw-session channel missing memory_type filter: {where}"
                )
