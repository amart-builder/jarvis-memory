"""Unit tests for Phase 8 observation extraction.

Tests are pure (no network) — the live model call is exercised only by
the smoke test in scripts/. These cover the prompt-build + parse-response
contract that everything downstream depends on.
"""
from __future__ import annotations

import pytest

from scripts.longmemeval.extract import (
    EXTRACTION_PROMPT_TEMPLATE,
    Observation,
    build_extraction_prompt,
    parse_extraction_response,
)


# ── prompt ──────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_includes_session_text_and_date(self):
        prompt = build_extraction_prompt(
            session_text="user: hello\nassistant: hi",
            session_date="2023-06-12",
        )
        assert "user: hello" in prompt
        assert "assistant: hi" in prompt
        assert "2023-06-12" in prompt

    def test_includes_strict_rules(self):
        prompt = build_extraction_prompt("text", "2023-01-01")
        # Verbatim discipline must appear or extraction quality collapses
        assert "VERBATIM" in prompt
        assert "Do NOT infer" in prompt
        # Schema fields must all appear
        for f in ["type", "key", "value", "date", "details"]:
            assert f in prompt
        # Type vocabulary must be locked
        for t in ["event", "fact", "preference", "update"]:
            assert t in prompt

    def test_handles_unknown_session_date(self):
        # Some haystack rows have empty haystack_dates entries
        prompt = build_extraction_prompt("user: x", "")
        assert "unknown" in prompt


# ── response parser ─────────────────────────────────────────────────────


class TestParseResponse:
    def test_clean_json(self):
        raw = '''{
            "observations": [
                {"type": "event", "key": "yoga_class", "value": "5th class",
                 "date": "2023-06-12", "details": "vinyasa"}
            ]
        }'''
        obs = parse_extraction_response(raw)
        assert len(obs) == 1
        assert obs[0].type == "event"
        assert obs[0].key == "yoga_class"
        assert obs[0].value == "5th class"
        assert obs[0].date == "2023-06-12"
        assert obs[0].details == "vinyasa"

    def test_multiple_types(self):
        raw = '''{"observations": [
            {"type": "event", "key": "a", "value": "1", "date": null, "details": ""},
            {"type": "fact", "key": "b", "value": "2", "date": null, "details": ""},
            {"type": "preference", "key": "c", "value": "3", "date": null, "details": ""},
            {"type": "update", "key": "d", "value": "4", "date": null, "details": ""}
        ]}'''
        obs = parse_extraction_response(raw)
        assert len(obs) == 4
        assert {o.type for o in obs} == {"event", "fact", "preference", "update"}

    def test_strips_markdown_code_fence(self):
        raw = '''```json
        {"observations": [{"type": "fact", "key": "k", "value": "v",
                           "date": null, "details": ""}]}
        ```'''
        obs = parse_extraction_response(raw)
        assert len(obs) == 1
        assert obs[0].key == "k"

    def test_handles_prose_around_json(self):
        raw = '''Sure! Here is the extraction:
        {"observations": [{"type": "fact", "key": "k", "value": "v",
                           "date": null, "details": ""}]}
        Hope that helps!'''
        obs = parse_extraction_response(raw)
        assert len(obs) == 1

    def test_empty_response(self):
        assert parse_extraction_response("") == []
        assert parse_extraction_response("   ") == []
        assert parse_extraction_response("not json at all") == []

    def test_drops_invalid_type(self):
        raw = '''{"observations": [
            {"type": "garbage", "key": "k", "value": "v", "date": null, "details": ""},
            {"type": "fact", "key": "k2", "value": "v2", "date": null, "details": ""}
        ]}'''
        obs = parse_extraction_response(raw)
        assert len(obs) == 1
        assert obs[0].key == "k2"

    def test_drops_missing_key(self):
        raw = '''{"observations": [
            {"type": "fact", "key": "", "value": "v", "date": null, "details": ""}
        ]}'''
        obs = parse_extraction_response(raw)
        assert obs == []

    def test_drops_missing_value(self):
        raw = '''{"observations": [
            {"type": "fact", "key": "k", "value": "", "date": null, "details": ""},
            {"type": "fact", "key": "k2", "value": null, "date": null, "details": ""}
        ]}'''
        obs = parse_extraction_response(raw)
        assert obs == []

    def test_invalid_date_becomes_null(self):
        # Model sometimes gives "June 2023" or "2023" — we coerce to null,
        # not raise, since the value itself is still useful.
        raw = '''{"observations": [
            {"type": "event", "key": "k", "value": "v",
             "date": "June 2023", "details": ""}
        ]}'''
        obs = parse_extraction_response(raw)
        assert len(obs) == 1
        assert obs[0].date is None

    def test_truncates_long_fields(self):
        long_val = "X" * 500
        raw = f'''{{"observations": [
            {{"type": "fact", "key": "{long_val}",
              "value": "{long_val}", "date": null,
              "details": "{long_val}"}}
        ]}}'''
        obs = parse_extraction_response(raw)
        assert len(obs) == 1
        # Truncated to caps in extract.py
        assert len(obs[0].key) <= 60
        assert len(obs[0].value) <= 200
        assert len(obs[0].details) <= 80

    def test_caps_at_30_observations(self):
        rows = [
            f'{{"type": "fact", "key": "k{i}", "value": "v{i}", '
            f'"date": null, "details": ""}}'
            for i in range(50)
        ]
        raw = '{"observations": [' + ",".join(rows) + ']}'
        obs = parse_extraction_response(raw)
        assert len(obs) == 30  # _MAX_OBSERVATIONS_PER_SESSION

    def test_observations_field_must_be_list(self):
        raw = '{"observations": "not a list"}'
        assert parse_extraction_response(raw) == []

    def test_skips_non_dict_rows(self):
        raw = '''{"observations": [
            "stringy row",
            {"type": "fact", "key": "k", "value": "v", "date": null, "details": ""},
            42,
            null
        ]}'''
        obs = parse_extraction_response(raw)
        assert len(obs) == 1


# ── render_line ─────────────────────────────────────────────────────────


class TestRenderLine:
    def test_with_date_and_details(self):
        o = Observation(type="event", key="yoga_class",
                        value="5th class", date="2023-06-12",
                        details="vinyasa style")
        s = o.render_line()
        assert s == "- event: yoga_class = 5th class [2023-06-12] — vinyasa style"

    def test_without_date(self):
        o = Observation(type="fact", key="user_age", value="32",
                        date=None, details="")
        s = o.render_line()
        assert s == "- fact: user_age = 32"

    def test_with_date_no_details(self):
        o = Observation(type="update", key="5K_PB",
                        value="27:45 → 26:30", date="2023-07-30",
                        details="")
        s = o.render_line()
        assert s == "- update: 5K_PB = 27:45 → 26:30 [2023-07-30]"


# ── extract_observations (live-call wrapper) ────────────────────────────


class _FakeResponse:
    def __init__(self, content: str):
        msg = type("M", (), {"content": content})
        choice = type("C", (), {"message": msg})
        self.choices = [choice]


class _FakeOpenAIClient:
    def __init__(self, response_content: str = "", raise_on_call: bool = False):
        self.response_content = response_content
        self.raise_on_call = raise_on_call
        self.last_kwargs: dict = {}

        client = self  # closure for chat.completions.create

        class _Completions:
            def create(self_inner, **kwargs):
                client.last_kwargs = kwargs
                if client.raise_on_call:
                    raise RuntimeError("simulated API failure")
                return _FakeResponse(client.response_content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class TestExtractObservations:
    def test_happy_path_uses_model_and_seed(self):
        from scripts.longmemeval.extract import extract_observations

        client = _FakeOpenAIClient(
            response_content='{"observations": ['
            '{"type": "fact", "key": "k", "value": "v", "date": null, "details": ""}'
            ']}'
        )
        obs = extract_observations(
            session_text="user: hi\nassistant: hello",
            session_date="2023-06-12",
            client=client,
            model="gpt-4o-mini",
        )
        assert len(obs) == 1
        assert obs[0].key == "k"
        # Verify deterministic args propagated
        assert client.last_kwargs["model"] == "gpt-4o-mini"
        assert client.last_kwargs["temperature"] == 0
        assert client.last_kwargs["seed"] == 42

    def test_empty_session_text_returns_empty_no_call(self):
        from scripts.longmemeval.extract import extract_observations

        client = _FakeOpenAIClient(response_content="should not be called")
        obs = extract_observations(
            session_text="",
            session_date="2023-06-12",
            client=client,
        )
        assert obs == []
        assert client.last_kwargs == {}  # never called

    def test_api_failure_returns_empty_not_raise(self):
        """Phase 8 must be additive — extraction failures are graceful."""
        from scripts.longmemeval.extract import extract_observations

        client = _FakeOpenAIClient(raise_on_call=True)
        obs = extract_observations(
            session_text="user: hi",
            session_date="2023-06-12",
            client=client,
        )
        assert obs == []  # No exception raised


# ── extract_observations_batch (parallel) ───────────────────────────────


class TestExtractObservationsBatch:
    def test_returns_per_session_map(self):
        from scripts.longmemeval.extract import extract_observations_batch

        client = _FakeOpenAIClient(
            response_content='{"observations": ['
            '{"type": "fact", "key": "k", "value": "v", "date": null, "details": ""}'
            ']}'
        )
        sessions = [
            ("sess_a", "user: hi", "2023-01-01"),
            ("sess_b", "user: hello", "2023-02-01"),
            ("sess_c", "user: hey", "2023-03-01"),
        ]
        out = extract_observations_batch(sessions=sessions, client=client, max_workers=2)
        assert set(out.keys()) == {"sess_a", "sess_b", "sess_c"}
        for sid in ("sess_a", "sess_b", "sess_c"):
            assert len(out[sid]) == 1
            assert out[sid][0].key == "k"

    def test_empty_input(self):
        from scripts.longmemeval.extract import extract_observations_batch

        out = extract_observations_batch(sessions=[], client=_FakeOpenAIClient())
        assert out == {}

    def test_empty_session_text_skips_call(self):
        from scripts.longmemeval.extract import extract_observations_batch

        client = _FakeOpenAIClient(response_content="should not be called")
        out = extract_observations_batch(
            sessions=[("sess_x", "", "2023-01-01")],
            client=client,
        )
        assert out == {"sess_x": []}


# ── ExtractionCache (sqlite) ────────────────────────────────────────────


class TestExtractionCache:
    def _new_cache(self, tmp_path):
        from scripts.longmemeval.extract import ExtractionCache
        return ExtractionCache(tmp_path / "cache.db")

    def test_init_creates_db(self, tmp_path):
        cache = self._new_cache(tmp_path)
        assert (tmp_path / "cache.db").exists()
        # Empty cache → all misses
        assert cache.hits == 0
        assert cache.misses == 0

    def test_make_key_deterministic_and_input_sensitive(self):
        from scripts.longmemeval.extract import ExtractionCache
        k1 = ExtractionCache.make_key(
            session_text="user: hi", session_date="2023-01-01",
            prompt_version="v1", model="gpt-4o-mini",
        )
        k2 = ExtractionCache.make_key(
            session_text="user: hi", session_date="2023-01-01",
            prompt_version="v1", model="gpt-4o-mini",
        )
        assert k1 == k2  # deterministic
        # Each input field affects the key
        for differs in [
            {"session_text": "user: bye"},
            {"session_date": "2024-01-01"},
            {"prompt_version": "v2"},
            {"model": "gpt-4o"},
        ]:
            kw = dict(session_text="user: hi", session_date="2023-01-01",
                      prompt_version="v1", model="gpt-4o-mini")
            kw.update(differs)
            assert ExtractionCache.make_key(**kw) != k1

    def test_get_miss_returns_none(self, tmp_path):
        cache = self._new_cache(tmp_path)
        assert cache.get("nonexistent_key") is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_put_then_get_roundtrips_observations(self, tmp_path):
        from scripts.longmemeval.extract import Observation
        cache = self._new_cache(tmp_path)
        observations = [
            Observation(type="event", key="yoga", value="5th class",
                        date="2023-06-12", details="vinyasa"),
            Observation(type="fact", key="age", value="32",
                        date=None, details=""),
        ]
        cache.put(
            "key1", observations,
            model="gpt-4o-mini", prompt_version="v1", session_date="2023-06-12",
        )
        out = cache.get("key1")
        assert out is not None
        assert len(out) == 2
        assert out[0].type == "event"
        assert out[0].key == "yoga"
        assert out[0].value == "5th class"
        assert out[0].date == "2023-06-12"
        assert out[0].details == "vinyasa"
        assert out[1].type == "fact"
        assert out[1].date is None
        assert out[1].details == ""
        assert cache.hits == 1

    def test_extract_observations_uses_cache_on_hit(self, tmp_path):
        """Cache hit must skip the OpenAI call entirely."""
        from scripts.longmemeval.extract import (
            ExtractionCache, Observation, extract_observations,
        )
        cache = ExtractionCache(tmp_path / "cache.db")
        # Pre-populate
        from scripts.longmemeval.extract import PROMPT_VERSION
        key = ExtractionCache.make_key(
            session_text="user: hi", session_date="2023-06-12",
            prompt_version=PROMPT_VERSION, model="gpt-4o-mini",
        )
        cache.put(
            key,
            [Observation(type="fact", key="cached", value="yes", date=None, details="")],
            model="gpt-4o-mini", prompt_version=PROMPT_VERSION,
            session_date="2023-06-12",
        )

        client = _FakeOpenAIClient(response_content="SHOULD NOT BE CALLED")
        obs = extract_observations(
            session_text="user: hi", session_date="2023-06-12",
            client=client, model="gpt-4o-mini", cache=cache,
        )
        assert len(obs) == 1
        assert obs[0].key == "cached"
        assert client.last_kwargs == {}  # never called

    def test_extract_observations_misses_then_writes(self, tmp_path):
        """Cache miss runs OpenAI, then stores result."""
        from scripts.longmemeval.extract import (
            ExtractionCache, extract_observations,
        )
        cache = ExtractionCache(tmp_path / "cache.db")
        client = _FakeOpenAIClient(
            response_content='{"observations": [{"type": "fact",'
            ' "key": "from_api", "value": "v", "date": null, "details": ""}]}'
        )
        # First call: miss → OpenAI → cache
        obs1 = extract_observations(
            session_text="user: hello", session_date="2023-07-01",
            client=client, model="gpt-4o-mini", cache=cache,
        )
        assert len(obs1) == 1
        assert obs1[0].key == "from_api"
        assert client.last_kwargs.get("model") == "gpt-4o-mini"

        # Second call with a NEW client that would error if hit:
        client2 = _FakeOpenAIClient(raise_on_call=True)
        obs2 = extract_observations(
            session_text="user: hello", session_date="2023-07-01",
            client=client2, model="gpt-4o-mini", cache=cache,
        )
        assert len(obs2) == 1
        assert obs2[0].key == "from_api"
        # Returned observations match byte-for-byte after roundtrip
        assert obs2[0].to_dict() == obs1[0].to_dict()
        assert cache.hits == 1
        assert cache.misses == 1

    def test_different_prompt_versions_dont_collide(self, tmp_path):
        """Bumping PROMPT_VERSION must invalidate prior cache entries."""
        from scripts.longmemeval.extract import ExtractionCache
        cache = ExtractionCache(tmp_path / "cache.db")
        k_v1 = ExtractionCache.make_key(
            session_text="x", session_date="2023-01-01",
            prompt_version="v1", model="gpt-4o-mini",
        )
        k_v2 = ExtractionCache.make_key(
            session_text="x", session_date="2023-01-01",
            prompt_version="v2", model="gpt-4o-mini",
        )
        assert k_v1 != k_v2
        # Storing under v1 must not satisfy a v2 lookup
        cache.put(k_v1, [], model="gpt-4o-mini",
                  prompt_version="v1", session_date="2023-01-01")
        assert cache.get(k_v2) is None

    def test_batch_uses_cache(self, tmp_path):
        """Cache propagates through extract_observations_batch."""
        from scripts.longmemeval.extract import (
            ExtractionCache, extract_observations_batch,
        )
        cache = ExtractionCache(tmp_path / "cache.db")
        client = _FakeOpenAIClient(
            response_content='{"observations": [{"type": "fact",'
            ' "key": "k", "value": "v", "date": null, "details": ""}]}'
        )
        sessions = [
            ("sa", "user: a", "2023-01-01"),
            ("sb", "user: b", "2023-02-01"),
        ]
        # Run 1 — both miss
        out1 = extract_observations_batch(
            sessions=sessions, client=client, max_workers=2, cache=cache,
        )
        assert cache.misses == 2
        assert cache.hits == 0
        assert all(len(out1[s]) == 1 for s in ("sa", "sb"))

        # Run 2 — both hit; OpenAI must NOT be called
        client2 = _FakeOpenAIClient(raise_on_call=True)
        out2 = extract_observations_batch(
            sessions=sessions, client=client2, max_workers=2, cache=cache,
        )
        assert cache.hits == 2
        assert all(len(out2[s]) == 1 for s in ("sa", "sb"))
        # Returned observations are equivalent
        for s in ("sa", "sb"):
            assert out1[s][0].to_dict() == out2[s][0].to_dict()

    def test_corrupt_entry_treated_as_miss(self, tmp_path):
        """A corrupted JSON value must not crash — fall back to miss."""
        from scripts.longmemeval.extract import ExtractionCache
        import sqlite3
        cache = ExtractionCache(tmp_path / "cache.db")
        # Inject corrupt row directly
        with sqlite3.connect(tmp_path / "cache.db") as conn:
            conn.execute(
                """INSERT INTO extractions
                   (cache_key, observations_json, n_observations,
                    model, prompt_version, session_date)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("bad_key", "{not valid json", 0,
                 "gpt-4o-mini", "v1", "2023-01-01"),
            )
            conn.commit()
        assert cache.get("bad_key") is None  # treated as miss, no crash

    def test_concurrent_batch_safe(self, tmp_path):
        """Batch with workers > 1 must not deadlock or corrupt the cache."""
        from scripts.longmemeval.extract import (
            ExtractionCache, extract_observations_batch,
        )
        cache = ExtractionCache(tmp_path / "cache.db")
        client = _FakeOpenAIClient(
            response_content='{"observations": [{"type": "fact",'
            ' "key": "k", "value": "v", "date": null, "details": ""}]}'
        )
        sessions = [(f"s{i}", f"user: msg-{i}", "2023-01-01") for i in range(20)]
        out = extract_observations_batch(
            sessions=sessions, client=client, max_workers=4, cache=cache,
        )
        assert len(out) == 20
        assert all(len(v) == 1 for v in out.values())
        assert cache.misses == 20
        # Re-run — all 20 should hit
        out2 = extract_observations_batch(
            sessions=sessions, client=_FakeOpenAIClient(raise_on_call=True),
            max_workers=4, cache=cache,
        )
        assert len(out2) == 20
        assert cache.hits == 20

    def test_empty_extraction_is_cached_when_api_succeeds(self, tmp_path):
        """API-success with zero observations MUST be cached.

        Without this, re-runs hit OpenAI for those sessions, and OpenAI
        is non-deterministic enough at temperature=0 that a re-call
        could produce a different result — leaking variance into
        benchmark scores. This is the load-bearing accuracy claim.
        """
        from scripts.longmemeval.extract import ExtractionCache, extract_observations
        cache = ExtractionCache(tmp_path / "cache.db")
        client = _FakeOpenAIClient(response_content='{"observations": []}')
        obs1 = extract_observations(
            session_text="user: hi only",
            session_date="2023-01-01",
            client=client, cache=cache,
        )
        assert obs1 == []
        assert cache.misses == 1
        # Second call MUST hit cache, not re-call OpenAI
        client2 = _FakeOpenAIClient(raise_on_call=True)
        obs2 = extract_observations(
            session_text="user: hi only",
            session_date="2023-01-01",
            client=client2, cache=cache,
        )
        assert obs2 == []
        assert cache.hits == 1
        assert client2.last_kwargs == {}  # never called

    def test_api_error_is_NOT_cached(self, tmp_path):
        """API errors should retry on next run, not serve stale-error from cache."""
        from scripts.longmemeval.extract import ExtractionCache, extract_observations
        cache = ExtractionCache(tmp_path / "cache.db")
        # First call: API errors → empty list returned, cache NOT populated
        client_err = _FakeOpenAIClient(raise_on_call=True)
        obs_err = extract_observations(
            session_text="user: x", session_date="2023-01-01",
            client=client_err, cache=cache,
        )
        assert obs_err == []
        # Cache must be empty — second call must hit OpenAI again
        client_ok = _FakeOpenAIClient(
            response_content='{"observations": [{"type": "fact",'
            ' "key": "from_retry", "value": "v", "date": null, "details": ""}]}'
        )
        obs_retry = extract_observations(
            session_text="user: x", session_date="2023-01-01",
            client=client_ok, cache=cache,
        )
        assert len(obs_retry) == 1
        assert obs_retry[0].key == "from_retry"
        # The retry SHOULD have populated the cache
        assert cache.misses == 2  # both calls were misses
        assert cache.hits == 0

    def test_cache_put_failure_does_not_break_extraction(self, tmp_path):
        """If cache.put() raises, extraction must still return observations."""
        from scripts.longmemeval.extract import ExtractionCache, extract_observations

        class _BrokenCache(ExtractionCache):
            def put(self, *a, **kw):
                raise RuntimeError("simulated cache write failure")

        cache = _BrokenCache(tmp_path / "cache.db")
        client = _FakeOpenAIClient(
            response_content='{"observations": [{"type": "fact",'
            ' "key": "k", "value": "v", "date": null, "details": ""}]}'
        )
        obs = extract_observations(
            session_text="user: x", session_date="2023-01-01",
            client=client, cache=cache,
        )
        # Observations still returned even though cache.put() raised
        assert len(obs) == 1
        assert obs[0].key == "k"

    def test_observation_field_equality_after_roundtrip(self, tmp_path):
        """Frozen-dataclass equality should hold after cache roundtrip."""
        from scripts.longmemeval.extract import (
            ExtractionCache, extract_observations,
        )
        cache = ExtractionCache(tmp_path / "cache.db")
        client = _FakeOpenAIClient(
            response_content='{"observations": [{"type": "event",'
            ' "key": "yoga", "value": "5th class", "date": "2023-06-12",'
            ' "details": "vinyasa"}]}'
        )
        obs1 = extract_observations(
            session_text="user: did yoga", session_date="2023-06-12",
            client=client, cache=cache,
        )
        # Re-run hits cache
        obs2 = extract_observations(
            session_text="user: did yoga", session_date="2023-06-12",
            client=_FakeOpenAIClient(raise_on_call=True), cache=cache,
        )
        # Frozen dataclass equality — every field must match
        assert obs1 == obs2
        assert obs1[0] == obs2[0]
