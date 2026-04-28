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
