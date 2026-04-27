"""Unit tests for the LongMemEval adapter — testable pieces only.

The adapter has integration paths (Neo4j, Chroma, LLM) we don't mock —
those are validated by the live --validate run, not unit tests. This
module covers:
  - resume / JSONL parsing
  - stratified sampling
  - AR2 seed-broadening behavior (via monkey-patch verification)
  - Stage 0: gold-session retrieval diagnostics + summary JSON
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Resume: load_done_question_ids ────────────────────────────────────


def test_load_done_question_ids_empty_when_missing(tmp_path):
    from scripts.run_longmemeval import load_done_question_ids
    assert load_done_question_ids(tmp_path / "nope.jsonl") == set()


def test_load_done_question_ids_reads_existing(tmp_path):
    from scripts.run_longmemeval import load_done_question_ids
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "a"}) + "\n"
        + json.dumps({"question_id": "q2", "hypothesis": "b"}) + "\n"
    )
    assert load_done_question_ids(p) == {"q1", "q2"}


def test_load_done_question_ids_tolerates_blank_lines(tmp_path):
    from scripts.run_longmemeval import load_done_question_ids
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1"}) + "\n"
        + "\n"
        + json.dumps({"question_id": "q2"}) + "\n"
    )
    assert load_done_question_ids(p) == {"q1", "q2"}


def test_load_done_question_ids_tolerates_bad_line(tmp_path):
    from scripts.run_longmemeval import load_done_question_ids
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1"}) + "\n"
        + "{bad json\n"
        + json.dumps({"question_id": "q2"}) + "\n"
    )
    # q1 is read; the bad line is skipped; q2 is read.
    assert load_done_question_ids(p) == {"q1", "q2"}


def test_load_done_question_ids_skips_errored_rows(tmp_path):
    """Errored rows must NOT count as done — they need to be retried."""
    from scripts.run_longmemeval import load_done_question_ids
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "ok"}) + "\n"
        + json.dumps({"question_id": "q2", "hypothesis": "", "error": "Traceback..."}) + "\n"
        + json.dumps({"question_id": "q3", "hypothesis": "ok"}) + "\n"
    )
    # q2 was an error row — gets retried, NOT in done.
    assert load_done_question_ids(p) == {"q1", "q3"}


# ── stratified_subset ─────────────────────────────────────────────────


def test_stratified_subset_picks_n_per_category():
    from scripts.run_longmemeval import stratified_subset
    data = []
    for cat in ("temporal-reasoning", "multi-session", "knowledge-update",
                "single-session-user", "single-session-assistant",
                "single-session-preference"):
        for i in range(5):
            data.append({"question_id": f"{cat}_{i}", "question_type": cat})
    out = stratified_subset(data, n_per_cat=2)
    # 6 cats × 2 = 12 questions
    assert len(out) == 12
    cats = [q["question_type"] for q in out]
    for cat in ("temporal-reasoning", "multi-session", "knowledge-update",
                "single-session-user", "single-session-assistant",
                "single-session-preference"):
        assert cats.count(cat) == 2


def test_stratified_subset_includes_abstention():
    """_abs questions form their own bucket so validation hits abstention."""
    from scripts.run_longmemeval import stratified_subset
    data = [
        {"question_id": f"q_{i}", "question_type": "multi-session"} for i in range(3)
    ] + [
        {"question_id": f"abs_{i}_abs", "question_type": "multi-session"} for i in range(3)
    ]
    out = stratified_subset(data, n_per_cat=2)
    abs_ids = [q["question_id"] for q in out if q["question_id"].endswith("_abs")]
    assert len(abs_ids) == 2


# ── AR2 seed-broadening ───────────────────────────────────────────────


def test_apply_ppr_overrides_broadens_seeds():
    """After apply_ppr_overrides, common nouns ≥4 chars seed the PPR.

    Test by hijacking the patched _extract_query_entities directly.
    """
    from jarvis_memory.search import ppr as ppr_mod
    from scripts.run_longmemeval import apply_ppr_overrides

    original_extract = ppr_mod._extract_query_entities

    try:
        apply_ppr_overrides()
        seeds = ppr_mod._extract_query_entities("how often do I exercise")
        # Original (proper-noun-only) returns []. Broadened returns
        # at least "exercise" — possibly "often" too if not in stoplist.
        assert "exercise" in seeds, f"AR2 didn't seed common noun; got {seeds}"
    finally:
        # Revert patches so other tests aren't affected.
        ppr_mod._extract_query_entities = original_extract
        # Also revert the PPR function patch (apply_ppr_overrides patches both).
        from jarvis_memory.search.ppr import personalized_pagerank as orig_ppr_fn
        ppr_mod.personalized_pagerank = orig_ppr_fn


def test_apply_ppr_overrides_preserves_proper_noun_seeds():
    """Broadening shouldn't drop the original proper-noun extraction."""
    from jarvis_memory.search import ppr as ppr_mod
    from scripts.run_longmemeval import apply_ppr_overrides

    original_extract = ppr_mod._extract_query_entities

    try:
        apply_ppr_overrides()
        seeds = ppr_mod._extract_query_entities("decisions in Catalyst that affected Astack")
        # Both proper nouns should survive (lowercase form per existing code).
        assert "catalyst" in seeds
        assert "astack" in seeds
    finally:
        ppr_mod._extract_query_entities = original_extract
        from jarvis_memory.search.ppr import personalized_pagerank as orig_ppr_fn
        ppr_mod.personalized_pagerank = orig_ppr_fn


def test_apply_ppr_overrides_skips_stoplist():
    """AR2 must not seed stoplist words like 'this', 'have', 'when'."""
    from jarvis_memory.search import ppr as ppr_mod
    from scripts.run_longmemeval import apply_ppr_overrides, _AR2_STOPLIST

    original_extract = ppr_mod._extract_query_entities

    try:
        apply_ppr_overrides()
        seeds = ppr_mod._extract_query_entities("when have I been doing this")
        for word in seeds:
            assert word not in _AR2_STOPLIST, f"stoplist word leaked: {word}"
    finally:
        ppr_mod._extract_query_entities = original_extract
        from jarvis_memory.search.ppr import personalized_pagerank as orig_ppr_fn
        ppr_mod.personalized_pagerank = orig_ppr_fn


def test_apply_ppr_overrides_sets_damping_to_05():
    """AR1: PPR damping defaults to 0.5 after overrides applied.

    Test by inspecting the wrapper's closure: ``ppr_with_alpha`` is a
    closure that holds the original PPR function. We inspect it by
    monkey-patching the original at the source FIRST, then applying
    overrides — so the captured closure sees our spy.
    """
    from jarvis_memory.search import ppr as ppr_mod
    from scripts.run_longmemeval import apply_ppr_overrides

    original_extract = ppr_mod._extract_query_entities
    original_ppr = ppr_mod.personalized_pagerank
    captured: dict = {}

    def spy(query, **kwargs):
        captured["damping"] = kwargs.get("damping")
        return []

    try:
        # Replace the source PPR with our spy BEFORE applying overrides.
        ppr_mod.personalized_pagerank = spy
        apply_ppr_overrides()  # captures `spy` as `_orig_ppr` in its closure
        # Now call the wrapper — it must pass damping=0.5 to spy.
        ppr_mod.personalized_pagerank("any query", driver=None)
        assert captured["damping"] == 0.5, f"AR1: got damping={captured.get('damping')}"
    finally:
        ppr_mod._extract_query_entities = original_extract
        ppr_mod.personalized_pagerank = original_ppr


# ── Stage 0: gold-session retrieval diagnostics ───────────────────────


def test_extract_session_id_strips_index_and_group_prefix():
    from scripts.run_longmemeval import _extract_session_id
    # Format: "{group_id}__{idx:03d}_{session_id}"
    assert _extract_session_id("lme_q_q1__001_abc123", "lme_q_q1") == "abc123"


def test_extract_session_id_preserves_underscores_inside_session_id():
    """Real LongMemEval session_ids contain underscores (e.g. answer_4be1b6b4_2)."""
    from scripts.run_longmemeval import _extract_session_id
    uuid = "lme_q_gpt4_2655b836__001_answer_4be1b6b4_2"
    group_id = "lme_q_gpt4_2655b836"
    assert _extract_session_id(uuid, group_id) == "answer_4be1b6b4_2"


def test_extract_session_id_returns_uuid_when_prefix_missing():
    """If the UUID doesn't carry our prefix shape, return it unchanged."""
    from scripts.run_longmemeval import _extract_session_id
    assert _extract_session_id("some-other-uuid", "lme_q_q1") == "some-other-uuid"


def test_extract_session_id_falls_back_when_no_index():
    """Pre-f9aa28c UUIDs were '{group_id}__{session_id}' (no index)."""
    from scripts.run_longmemeval import _extract_session_id
    assert _extract_session_id("lme_q_q1__abc", "lme_q_q1") == "abc"


def test_compute_diagnostics_all_gold_in_top5():
    from scripts.run_longmemeval import compute_retrieval_diagnostics
    group_id = "lme_q_q1"
    hits = [
        {"uuid": f"{group_id}__001_g1"},
        {"uuid": f"{group_id}__005_g2"},
        {"uuid": f"{group_id}__012_other"},
        {"uuid": f"{group_id}__003_g3"},
        {"uuid": f"{group_id}__008_other2"},
    ]
    d = compute_retrieval_diagnostics(hits, ["g1", "g2", "g3"], group_id)
    assert d["gold_count"] == 3
    assert d["gold_ranks"] == {"g1": 1, "g2": 2, "g3": 4}
    assert d["gold_in_top5"] == 3
    assert d["gold_in_top10"] == 3
    assert d["gold_in_pool"] == 3
    assert d["all_gold_in_top5"] is True
    assert d["any_gold_in_top5"] is True
    assert d["candidate_pool_size"] == 5


def test_compute_diagnostics_partial_retrieval():
    """One gold session at rank 7, one missing entirely."""
    from scripts.run_longmemeval import compute_retrieval_diagnostics
    group_id = "lme_q_q2"
    hits = [{"uuid": f"{group_id}__{i:03d}_n{i}"} for i in range(10)]
    hits[6]["uuid"] = f"{group_id}__006_g_found"  # rank 7
    d = compute_retrieval_diagnostics(hits, ["g_found", "g_missing"], group_id)
    assert d["gold_count"] == 2
    assert d["gold_ranks"] == {"g_found": 7, "g_missing": -1}
    assert d["gold_in_top5"] == 0
    assert d["gold_in_top10"] == 1
    assert d["gold_in_pool"] == 1
    assert d["all_gold_in_top5"] is False
    assert d["any_gold_in_top5"] is False


def test_compute_diagnostics_empty_hits():
    """No retrieval at all — all gold rank -1, no false positives."""
    from scripts.run_longmemeval import compute_retrieval_diagnostics
    d = compute_retrieval_diagnostics([], ["g1"], "lme_q_q3")
    assert d["gold_count"] == 1
    assert d["gold_ranks"] == {"g1": -1}
    assert d["gold_in_top5"] == 0
    assert d["gold_in_pool"] == 0
    assert d["all_gold_in_top5"] is False
    assert d["any_gold_in_top5"] is False
    assert d["candidate_pool_size"] == 0


def test_compute_diagnostics_first_match_wins():
    """If gold appears twice in hits (ingestion duplicate), record the better rank."""
    from scripts.run_longmemeval import compute_retrieval_diagnostics
    group_id = "lme_q_q4"
    hits = [
        {"uuid": f"{group_id}__001_other"},
        {"uuid": f"{group_id}__005_g1"},  # rank 2
        {"uuid": f"{group_id}__010_g1"},  # rank 3 — second copy of same sid
    ]
    d = compute_retrieval_diagnostics(hits, ["g1"], group_id)
    assert d["gold_ranks"]["g1"] == 2  # better rank kept


def test_compute_diagnostics_handles_id_field_fallback():
    """Hits may have ``id`` instead of ``uuid``."""
    from scripts.run_longmemeval import compute_retrieval_diagnostics
    group_id = "lme_q_q5"
    hits = [{"id": f"{group_id}__001_g1"}]
    d = compute_retrieval_diagnostics(hits, ["g1"], group_id)
    assert d["gold_ranks"]["g1"] == 1


# ── Stage 0: write_run_summary ────────────────────────────────────────


def test_write_run_summary_basic_categories(tmp_path):
    from scripts.run_longmemeval import write_run_summary
    out = tmp_path / "run.jsonl"
    out.write_text(
        json.dumps({"question_id": "q1", "predicted_category": "multi-session",
                    "answerer": "gpt41", "elapsed_sec": 12.5}) + "\n"
        + json.dumps({"question_id": "q2", "predicted_category": "multi-session",
                      "answerer": "gpt41", "elapsed_sec": 8.0}) + "\n"
        + json.dumps({"question_id": "q3", "predicted_category": "temporal-reasoning",
                      "answerer": "gpt41", "elapsed_sec": 15.0,
                      "error": "boom"}) + "\n"
    )
    summary_path = write_run_summary(out)
    assert summary_path == tmp_path / "run.summary.json"
    s = json.loads(summary_path.read_text())
    assert s["n_total"] == 3
    assert s["n_errored"] == 1
    assert s["predicted_categories"]["multi-session"] == 2
    assert s["predicted_categories"]["temporal-reasoning"] == 1
    assert s["errored_by_category"]["temporal-reasoning"] == 1
    assert s["answerer"] == ["gpt41"]
    assert s["elapsed_sec_total"] == 35.5
    # No diagnostics block when no rows have diagnostics.
    assert "diagnostics" not in s


def test_write_run_summary_includes_diagnostics_when_present(tmp_path):
    from scripts.run_longmemeval import write_run_summary
    out = tmp_path / "run.jsonl"
    out.write_text(
        json.dumps({
            "question_id": "q1",
            "predicted_category": "multi-session",
            "answerer": "gpt41",
            "elapsed_sec": 10.0,
            "diagnostics": {
                "gold_count": 3,
                "gold_in_top5": 3,
                "gold_in_top10": 3,
                "gold_in_pool": 3,
                "all_gold_in_top5": True,
                "any_gold_in_top5": True,
            },
        }) + "\n"
        + json.dumps({
            "question_id": "q2",
            "predicted_category": "multi-session",
            "answerer": "gpt41",
            "elapsed_sec": 10.0,
            "diagnostics": {
                "gold_count": 2,
                "gold_in_top5": 0,
                "gold_in_top10": 1,
                "gold_in_pool": 1,
                "all_gold_in_top5": False,
                "any_gold_in_top5": False,
            },
        }) + "\n"
    )
    summary_path = write_run_summary(out)
    s = json.loads(summary_path.read_text())
    d = s["diagnostics"]
    assert d["n_questions"] == 2
    assert d["all_gold_in_top5_pct"] == 0.5    # 1 of 2
    assert d["any_gold_in_top5_pct"] == 0.5    # 1 of 2
    assert d["any_gold_in_top10_pct"] == 1.0   # 2 of 2
    assert d["any_gold_in_pool_pct"] == 1.0    # 2 of 2
    cat = d["by_predicted_category"]["multi-session"]
    assert cat["n"] == 2
    assert cat["all_top5_pct"] == 0.5


def test_write_run_summary_diag_no_gold_goes_to_abstention_bucket(tmp_path):
    """Rows with gold_count=0 (abstention) tracked separately, not in aggregate."""
    from scripts.run_longmemeval import write_run_summary
    out = tmp_path / "run.jsonl"
    out.write_text(
        json.dumps({
            "question_id": "q1_abs", "predicted_category": "multi-session",
            "answerer": "gpt41", "elapsed_sec": 1.0,
            "diagnostics": {"gold_count": 0, "gold_in_top5": 0,
                            "gold_in_top10": 0, "gold_in_pool": 0,
                            "all_gold_in_top5": False, "any_gold_in_top5": False},
        }) + "\n"
        + json.dumps({
            "question_id": "q2", "predicted_category": "multi-session",
            "answerer": "gpt41", "elapsed_sec": 1.0,
            "diagnostics": {"gold_count": 2, "gold_in_top5": 2,
                            "gold_in_top10": 2, "gold_in_pool": 2,
                            "all_gold_in_top5": True, "any_gold_in_top5": True},
        }) + "\n"
    )
    s = json.loads(write_run_summary(out).read_text())
    d = s["diagnostics"]
    assert d["n_questions"] == 1            # only the gold-bearing q
    assert d["n_abstention"] == 1           # abstention tracked separately
    assert d["all_gold_in_top5_pct"] == 1.0  # 1/1, abstention not in denom


def test_write_run_summary_tolerates_blank_and_bad_lines(tmp_path):
    from scripts.run_longmemeval import write_run_summary
    out = tmp_path / "run.jsonl"
    out.write_text(
        json.dumps({"question_id": "q1", "predicted_category": "single-session-user",
                    "answerer": "gpt41", "elapsed_sec": 1.0}) + "\n"
        + "\n"
        + "{not json\n"
        + json.dumps({"question_id": "q2", "predicted_category": "single-session-user",
                      "answerer": "gpt41", "elapsed_sec": 1.0}) + "\n"
    )
    s = json.loads(write_run_summary(out).read_text())
    assert s["n_total"] == 2


# ── Stage 0: deterministic seeding ────────────────────────────────────


def test_run_seed_constant_is_42():
    """The run seed is the documented value used in published results."""
    from scripts.run_longmemeval import RUN_SEED
    assert RUN_SEED == 42


def test_call_llm_passes_seed_to_openai(monkeypatch):
    """OpenAI call must pass seed=RUN_SEED for run-to-run reproducibility."""
    from scripts import run_longmemeval as adapter

    captured: dict = {}

    class _Resp:
        class _Choice:
            class _Msg:
                content = "answer"
            message = _Msg()
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key")
        chat = _Chat()

    fake_module = type(adapter)("openai")
    fake_module.OpenAI = _Client
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    out = adapter.call_llm(answerer="gpt41", prompt="hello", max_tokens=10)
    assert out == "answer"
    assert captured.get("seed") == adapter.RUN_SEED
    assert captured.get("temperature") == 0
    assert captured.get("max_tokens") == 10
    assert captured.get("model") == adapter._GPT41_MODEL


def test_call_llm_does_not_pass_seed_to_anthropic(monkeypatch):
    """Anthropic Messages API has no seed param — passing it would error."""
    from scripts import run_longmemeval as adapter

    captured: dict = {}

    class _Block:
        text = "anthropic-answer"

    class _Resp:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Client:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key")
        messages = _Messages()

    fake_module = type(adapter)("anthropic")
    fake_module.Anthropic = _Client
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    out = adapter.call_llm(answerer="opus", prompt="hello", max_tokens=10)
    assert out == "anthropic-answer"
    assert "seed" not in captured, "Anthropic Messages.create has no seed kwarg"
    assert captured.get("temperature") == 0


def test_main_reexecs_when_pythonhashseed_unset(monkeypatch):
    """main() must re-exec with PYTHONHASHSEED=42 if not already set."""
    from scripts import run_longmemeval as adapter

    captured: dict = {}

    def fake_execvp(path, argv):
        captured["path"] = path
        captured["argv"] = argv
        # Don't actually exec — raise to short-circuit main().
        raise SystemExit(99)

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    # Pretend we're not in pytest so the in-test guard doesn't fire.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("os.execvp", fake_execvp)

    with pytest.raises(SystemExit) as exc:
        adapter.main()
    assert exc.value.code == 99
    assert "python" in captured["path"]
    assert captured["argv"][0] == captured["path"]
    # PYTHONHASHSEED env var must have been set before exec.
    import os
    assert os.environ.get("PYTHONHASHSEED") == "42"


def test_main_skips_reexec_when_pytest_marker_present(monkeypatch):
    """When PYTEST_CURRENT_TEST is set, main() must NOT re-exec."""
    from scripts import run_longmemeval as adapter

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake_test_marker")

    called: dict = {}

    def fake_execvp(path, argv):
        called["yes"] = True
        raise RuntimeError("execvp must NOT be called under pytest")

    monkeypatch.setattr("os.execvp", fake_execvp)
    # Need a clean argv so argparse doesn't error on test runner args.
    monkeypatch.setattr("sys.argv", ["run_longmemeval.py", "--help"])

    with pytest.raises(SystemExit):
        adapter.main()  # argparse --help triggers SystemExit(0)
    assert "yes" not in called


def test_run_one_question_includes_diagnostics_in_error_row(monkeypatch):
    """If generation crashes after retrieval, the error row still carries diagnostics."""
    from scripts import run_longmemeval as adapter

    fake_hits = [
        {"uuid": "lme_q_qX__001_g1", "content": "x", "referenced_date": "2024-01-01T00:00:00"},
        {"uuid": "lme_q_qX__002_other", "content": "y", "referenced_date": "2024-01-02T00:00:00"},
    ]

    monkeypatch.setattr(adapter, "ingest_question_haystack",
                        lambda **kw: 2)
    monkeypatch.setattr(adapter, "retrieve_with_omega_recipe",
                        lambda **kw: list(fake_hits))

    def boom(**kwargs):
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(adapter, "call_llm", boom)

    q = {"question_id": "qX", "question": "did I do the thing", "question_date": ""}
    row = adapter.run_one_question(
        q=q,
        answerer="gpt41",
        driver=None,
        embedding_store=None,
        chroma_collection=None,
        oracle_answer_session_ids=["g1"],
    )
    assert row["error"], "error row must have error trace"
    assert row["hypothesis"] == ""
    assert "diagnostics" in row, "retrieval signal must survive a generation crash"
    assert row["diagnostics"]["gold_ranks"]["g1"] == 1
    assert row["diagnostics"]["any_gold_in_top5"] is True
    assert row["seed_honored"] is True  # gpt41 path


# ── Stage 1: --use-oracle-categories flag ─────────────────────────────


def _stub_pipeline(monkeypatch):
    """Stub ingest/retrieve/LLM for fast in-process run_one_question tests."""
    from scripts import run_longmemeval as adapter
    monkeypatch.setattr(adapter, "ingest_question_haystack", lambda **kw: 0)
    monkeypatch.setattr(adapter, "retrieve_with_omega_recipe", lambda **kw: [])
    monkeypatch.setattr(adapter, "call_llm", lambda **kw: "stubbed")


def test_run_one_question_uses_oracle_category_when_provided(monkeypatch):
    """Stage 1: when oracle_category is passed, it overrides the classifier."""
    from scripts import run_longmemeval as adapter

    _stub_pipeline(monkeypatch)
    # Ask a question the heuristic classifier would label as
    # ``single-session-user`` (no temporal/preference/etc cues), but
    # supply an oracle override of ``multi-session``.
    q = {"question_id": "qX", "question": "did i do the thing",
         "question_date": "", "haystack_sessions": [], "haystack_session_ids": [],
         "haystack_dates": []}

    row = adapter.run_one_question(
        q=q,
        answerer="gpt41",
        driver=None,
        embedding_store=None,
        chroma_collection=None,
        oracle_category="multi-session",
    )
    assert row["predicted_category"] == "multi-session"
    assert row["category_source"] == "oracle"
    # Shadow label preserves the heuristic's original prediction.
    assert row["shadow_classifier_label"] != ""
    # Sanity: shadow != oracle (the test premise relies on a misclassification)
    assert row["shadow_classifier_label"] != "multi-session"


def test_run_one_question_uses_classifier_when_no_oracle_category(monkeypatch):
    """Sanity: with no oracle override, behavior matches Stage 0."""
    from scripts import run_longmemeval as adapter

    _stub_pipeline(monkeypatch)
    q = {"question_id": "qY", "question": "did i do the thing",
         "question_date": "", "haystack_sessions": [], "haystack_session_ids": [],
         "haystack_dates": []}

    row = adapter.run_one_question(
        q=q,
        answerer="gpt41",
        driver=None,
        embedding_store=None,
        chroma_collection=None,
    )
    assert row["category_source"] == "classifier"
    # Classifier and predicted should match when no oracle override.
    assert row["predicted_category"] == row["shadow_classifier_label"]


# ── Stage 2: list-extraction post-processing ─────────────────────────


def test_total_line_appended_when_missing_from_ms_count():
    """LLM forgot 'Total: N' — we append it after counting list items."""
    from scripts.run_longmemeval import maybe_append_total_line
    hyp = (
        "You bought:\n"
        "- apples [Note 3]\n"
        "- oranges [Note 5]\n"
        "- pears [Note 7]\n"
    )
    out = maybe_append_total_line(hyp, "multi-session", counting=True)
    assert "Total: 3" in out
    assert out.startswith(hyp.rstrip())  # original list intact


def test_total_line_skipped_when_already_present():
    """Don't double-append if LLM already wrote a Total line."""
    from scripts.run_longmemeval import maybe_append_total_line
    hyp = "- a\n- b\n\nTotal: 2"
    out = maybe_append_total_line(hyp, "multi-session", counting=True)
    assert out == hyp  # unchanged


def test_total_line_recognizes_count_or_answer_synonyms():
    """'Count: 3', 'Answer: 5', 'total: 4' all count as already-present."""
    from scripts.run_longmemeval import maybe_append_total_line
    for synonym in ("Count: 3", "Answer: 3", "total: 3", "TOTAL = 3"):
        hyp = f"- a\n- b\n- c\n\n{synonym}"
        out = maybe_append_total_line(hyp, "multi-session", counting=True)
        assert out == hyp, f"unexpected append for synonym: {synonym!r}"


def test_total_line_skipped_for_non_ms_categories():
    """Only fires for multi-session category."""
    from scripts.run_longmemeval import maybe_append_total_line
    hyp = "- a\n- b\n- c"
    for cat in ("temporal-reasoning", "single-session-user", "knowledge-update"):
        out = maybe_append_total_line(hyp, cat, counting=True)
        assert out == hyp, f"unexpectedly appended for {cat}"


def test_total_line_skipped_for_non_counting_questions():
    """MS but not a counting question — leave alone."""
    from scripts.run_longmemeval import maybe_append_total_line
    out = maybe_append_total_line(
        "- a\n- b\n- c", "multi-session", counting=False)
    assert out == "- a\n- b\n- c"


def test_total_line_skipped_when_too_few_list_items():
    """A single bullet might be incidental — don't fire on len<2."""
    from scripts.run_longmemeval import maybe_append_total_line
    out = maybe_append_total_line(
        "- only one item", "multi-session", counting=True)
    assert "Total" not in out


def test_total_line_handles_numbered_lists():
    """1. foo / 1) foo  — both recognized."""
    from scripts.run_longmemeval import maybe_append_total_line
    hyp = "1. apples\n2. oranges\n3) pears"
    out = maybe_append_total_line(hyp, "multi-session", counting=True)
    assert "Total: 3" in out


def test_total_line_handles_empty_hypothesis():
    from scripts.run_longmemeval import maybe_append_total_line
    assert maybe_append_total_line("", "multi-session", counting=True) == ""
    assert maybe_append_total_line("   \n", "multi-session", counting=True) == "   \n"


# ── Stage 2: prompt + classifier config changes ──────────────────────


def test_counting_k_floor_bumped_to_60():
    """Stage 2: counting K floor bumped 45 → 60 for wider MS recall."""
    from scripts.longmemeval.classifier import COUNTING_K_FLOOR
    assert COUNTING_K_FLOOR == 60


def test_multi_session_min_rel_lowered_to_005():
    """Stage 2: MS min_rel dropped 0.08 → 0.05 to keep more borderline candidates."""
    from scripts.longmemeval.classifier import FILTER_CONFIG
    assert FILTER_CONFIG["multi-session"]["min_rel"] == 0.05


def test_multisession_prompt_has_stage2_enumeration_discipline():
    """The MS prompt includes the new Stage 2 enumeration rules and final-line format."""
    from scripts.longmemeval.prompts import RAG_PROMPT_MULTISESSION
    # Atlas/AgentMemory rules
    assert "ENUMERATION DISCIPLINE" in RAG_PROMPT_MULTISESSION
    assert "USER STATEMENT BEATS ASSISTANT SKEPTICISM" in RAG_PROMPT_MULTISESSION
    assert "Preserve quantities, units, and dates EXACTLY" in RAG_PROMPT_MULTISESSION
    # Final-line format reinforcement
    assert 'Your final line MUST be exactly: "Total: N"' in RAG_PROMPT_MULTISESSION
    # Old strict-match rule was softened — should NOT include the original phrasing
    assert "REMOVE items that don't strictly match" not in RAG_PROMPT_MULTISESSION


# ── Stage 1: abstention guard ─────────────────────────────────────────


def test_extract_question_proper_nouns_dedupes_and_filters():
    from scripts.run_longmemeval import _extract_question_proper_nouns
    out = _extract_question_proper_nouns(
        "How did the Astack deploy go? Did Astack also fix the Catalyst bug?"
    )
    # "How" + "Did" filtered; Astack deduped.
    assert out == ["Astack", "Catalyst"]


def test_extract_question_proper_nouns_skips_modals():
    from scripts.run_longmemeval import _extract_question_proper_nouns
    assert _extract_question_proper_nouns("Where did I park?") == []


def test_extract_question_proper_nouns_handles_no_capitals():
    from scripts.run_longmemeval import _extract_question_proper_nouns
    assert _extract_question_proper_nouns("how often do i exercise") == []


def test_abstention_guard_skips_when_score_above_threshold():
    """High-confidence retrieval — guard does not fire."""
    from scripts.run_longmemeval import maybe_build_abstention_prefix
    out = maybe_build_abstention_prefix(
        question="What did Bob say about Astack?",
        hits=[{"content": "no mention here"}],
        top_score=0.5,
    )
    assert out is None


def test_abstention_guard_skips_when_no_proper_nouns():
    """Low-confidence retrieval but no entity to flag — guard does not fire."""
    from scripts.run_longmemeval import maybe_build_abstention_prefix
    out = maybe_build_abstention_prefix(
        question="how often do i exercise",
        hits=[{"content": "x"}],
        top_score=0.05,
    )
    assert out is None


def test_abstention_guard_skips_when_entity_present_in_hits():
    """Low confidence + proper noun, but the entity IS in retrieved content."""
    from scripts.run_longmemeval import maybe_build_abstention_prefix
    out = maybe_build_abstention_prefix(
        question="What did Astack do?",
        hits=[{"content": "astack shipped a new feature"}],
        top_score=0.05,
    )
    assert out is None  # case-insensitive match found


def test_abstention_guard_fires_when_entity_missing():
    """The targeted failure mode: weak retrieval + entity absent → abstain."""
    from scripts.run_longmemeval import maybe_build_abstention_prefix
    out = maybe_build_abstention_prefix(
        question="What is the capital of Astack-istan?",
        hits=[{"content": "completely unrelated"}],
        top_score=0.10,
    )
    assert out is not None
    assert "Astack-istan" in out or "'Astack" in out  # entity surfaces in the prefix
    assert "ABSTENTION" in out or "abstention" in out.lower()


def test_abstention_guard_handles_empty_hits():
    """No hits at all = abstain on any proper noun."""
    from scripts.run_longmemeval import maybe_build_abstention_prefix
    out = maybe_build_abstention_prefix(
        question="What did Catalyst do?",
        hits=[],
        top_score=0.0,
    )
    assert out is not None
    assert "Catalyst" in out


def test_abstention_guard_threshold_is_inclusive_floor():
    """At exactly the threshold, do NOT fire — score is high enough."""
    from scripts.run_longmemeval import maybe_build_abstention_prefix, _ABSTENTION_THRESHOLD
    out = maybe_build_abstention_prefix(
        question="Astack?",
        hits=[{"content": "no astack"}],
        top_score=_ABSTENTION_THRESHOLD,
    )
    assert out is None  # >= threshold means don't fire


def test_abstention_guard_threshold_calibrated_to_omega():
    """Stage 1 fix: threshold matches OMEGA's ABSTENTION_FILTER.min_rel (0.20),
    not the buggy 0.30 that fired on the median of every category."""
    from scripts.run_longmemeval import _ABSTENTION_THRESHOLD
    from scripts.longmemeval.classifier import ABSTENTION_FILTER
    assert _ABSTENTION_THRESHOLD == 0.20
    assert _ABSTENTION_THRESHOLD == ABSTENTION_FILTER["min_rel"]


def test_abstention_fired_field_in_run_one_question_row(monkeypatch):
    """End-to-end: when guard fires, abstention_fired=True in the row."""
    from scripts import run_longmemeval as adapter

    # Hits content has nothing matching "FooBarCorp" — guard should fire.
    fake_hits = [
        {"uuid": "lme_q_qX__001_x", "content": "completely unrelated stuff",
         "score": 0.05, "similarity": 0.05,
         "referenced_date": "2024-01-01T00:00:00"},
    ]
    monkeypatch.setattr(adapter, "ingest_question_haystack", lambda **kw: 1)
    monkeypatch.setattr(adapter, "retrieve_with_omega_recipe",
                        lambda **kw: list(fake_hits))
    monkeypatch.setattr(adapter, "call_llm", lambda **kw: "answer")

    q = {"question_id": "qX", "question": "What did FooBarCorp announce?",
         "question_date": "", "haystack_sessions": [],
         "haystack_session_ids": [], "haystack_dates": []}
    row = adapter.run_one_question(
        q=q, answerer="gpt41", driver=None, embedding_store=None,
        chroma_collection=None, oracle_category="single-session-user",
    )
    assert row["abstention_fired"] is True


def test_abstention_fired_false_when_score_above_threshold(monkeypatch):
    """End-to-end inverse: high confidence → no abstention prefix."""
    from scripts import run_longmemeval as adapter

    fake_hits = [
        {"uuid": "lme_q_qX__001_x", "content": "anything",
         "score": 0.6, "similarity": 0.6,
         "referenced_date": "2024-01-01T00:00:00"},
    ]
    monkeypatch.setattr(adapter, "ingest_question_haystack", lambda **kw: 1)
    monkeypatch.setattr(adapter, "retrieve_with_omega_recipe",
                        lambda **kw: list(fake_hits))
    monkeypatch.setattr(adapter, "call_llm", lambda **kw: "answer")

    q = {"question_id": "qX", "question": "What did FooBarCorp announce?",
         "question_date": "", "haystack_sessions": [],
         "haystack_session_ids": [], "haystack_dates": []}
    row = adapter.run_one_question(
        q=q, answerer="gpt41", driver=None, embedding_store=None,
        chroma_collection=None, oracle_category="single-session-user",
    )
    assert row["abstention_fired"] is False


# ── Stage 1: per-category context budget ──────────────────────────────


def test_trim_to_context_budget_drops_lowest_score_first():
    """Top-scored hits survive; lowest-score hits get dropped first."""
    from scripts.run_longmemeval import trim_to_context_budget
    hits = [
        {"uuid": "a", "content": "x" * 5000, "score": 0.9, "referenced_date": "2024-01-01T00:00:00"},
        {"uuid": "b", "content": "y" * 5000, "score": 0.5, "referenced_date": "2024-02-01T00:00:00"},
        {"uuid": "c", "content": "z" * 5000, "score": 0.8, "referenced_date": "2024-03-01T00:00:00"},
        {"uuid": "d", "content": "w" * 5000, "score": 0.3, "referenced_date": "2024-04-01T00:00:00"},
    ]
    # Budget = 12000 chars. min_hits=3 floor. Sorted by score: a(.9), c(.8), b(.5), d(.3).
    # a+c = 10k, +b = 15k > 12k but kept (still need to hit min_hits=3); +d = 20k > 12k AND len>=3, stop.
    # So kept = [a, c, b], date-sorted = [a (Jan), b (Feb), c (Mar)].
    out = trim_to_context_budget(hits, "single-session-user", budget_chars=12000)
    uuids = [h["uuid"] for h in out]
    assert uuids == ["a", "b", "c"]


def test_trim_to_context_budget_respects_min_hits_floor():
    """Even when first hit alone exceeds budget, keep min_hits anyway."""
    from scripts.run_longmemeval import trim_to_context_budget
    hits = [
        {"uuid": "big1", "content": "x" * 50000, "score": 0.9, "referenced_date": "2024-01-01T00:00:00"},
        {"uuid": "big2", "content": "y" * 50000, "score": 0.5, "referenced_date": "2024-02-01T00:00:00"},
        {"uuid": "big3", "content": "z" * 50000, "score": 0.7, "referenced_date": "2024-03-01T00:00:00"},
        {"uuid": "big4", "content": "w" * 50000, "score": 0.3, "referenced_date": "2024-04-01T00:00:00"},
    ]
    out = trim_to_context_budget(hits, "single-session-user", budget_chars=1000, min_hits=3)
    assert len(out) == 3, f"min_hits floor of 3 not respected: got {len(out)}"
    # Top 3 by score: big1(.9), big3(.7), big2(.5). Date-sorted: Jan, Feb, Mar.
    assert [h["uuid"] for h in out] == ["big1", "big2", "big3"]


def test_trim_to_context_budget_empty_input_returns_empty():
    from scripts.run_longmemeval import trim_to_context_budget
    assert trim_to_context_budget([], "multi-session") == []


def test_trim_to_context_budget_skips_uncapped_category():
    """multi-session/temporal/KU are NOT in CONTEXT_BUDGET_CHARS — return all hits."""
    from scripts.run_longmemeval import trim_to_context_budget
    from scripts.longmemeval.classifier import CONTEXT_BUDGET_CHARS

    hits = [
        {"uuid": str(i), "content": "x" * 8000, "score": 1.0 - i * 0.01,
         "referenced_date": f"2024-01-{i+1:02d}T00:00:00"}
        for i in range(15)
    ]
    # MS not in the dict (Atlas's MS=30K trim broke a real temporal
    # question in smoke; restored to no-trim).
    assert "multi-session" not in CONTEXT_BUDGET_CHARS
    assert "knowledge-update" not in CONTEXT_BUDGET_CHARS
    assert "temporal-reasoning" not in CONTEXT_BUDGET_CHARS

    out_ms = trim_to_context_budget(hits, "multi-session")
    assert len(out_ms) == 15  # no trim
    out_tr = trim_to_context_budget(hits, "temporal-reasoning")
    assert len(out_tr) == 15  # no trim

    # SS is in the dict and DOES trim.
    out_ss = trim_to_context_budget(hits, "single-session-user")
    assert len(out_ss) < 15


def test_trim_to_context_budget_returns_date_sorted():
    """Output order = date ascending, regardless of input order."""
    from scripts.run_longmemeval import trim_to_context_budget
    hits = [
        {"uuid": "c", "content": "z", "score": 0.5, "referenced_date": "2024-03-01T00:00:00"},
        {"uuid": "a", "content": "x", "score": 0.9, "referenced_date": "2024-01-01T00:00:00"},
        {"uuid": "b", "content": "y", "score": 0.7, "referenced_date": "2024-02-01T00:00:00"},
    ]
    out = trim_to_context_budget(hits, "multi-session")
    assert [h["uuid"] for h in out] == ["a", "b", "c"]


def test_trim_to_context_budget_actually_trims_at_realistic_session_sizes():
    """50K SS budget allows ~5 sessions at typical ~10K-char real sizes.

    Reviewer flagged: smaller test fixtures hid a bug where the
    min_hits floor dominated every SS run, making the budget a no-op.
    Use realistic 10K-char hits to exercise the budget path itself.
    """
    from scripts.run_longmemeval import trim_to_context_budget
    hits = [
        {"uuid": str(i), "content": "x" * 10500, "score": 1.0 - i * 0.05,
         "referenced_date": f"2024-01-{i+1:02d}T00:00:00"}
        for i in range(12)
    ]
    out = trim_to_context_budget(hits, "single-session-user")
    # 50K / 10.5K ≈ 4.7 → 5 hits fit, 6th breaks budget. Loop appends
    # while either (kept < min_hits) OR (cumulative + next ≤ budget).
    # 1: 10.5K (kept=1, < min) → keep
    # 2: 21K  (kept=2, < min) → keep
    # 3: 31.5K (kept=3, ≥ min, +10.5=42K ≤ 50K) → keep
    # 4: 42K (kept=4, +10.5=52.5K > 50K) → break before kept=4 actually
    #    Wait: the check is "if kept and len(kept) >= min_hits and total + c > budget"
    #    Before iter 4: kept has 3, total=31.5K. c=10.5K. 31.5+10.5=42K, ≤50K, no break, kept=4.
    # 5: kept=4, total=42K. c=10.5K. 42+10.5=52.5K > 50K. Break BEFORE adding.
    # Wait — break is checked BEFORE append. So at iter 4 we have kept=3, total=31.5K.
    #    The break check fires when kept ≥ min_hits AND total + c > budget.
    #    iter 4: kept=3 ≥ 3, 31.5+10.5=42 ≤ 50, no break, append → kept=4 total=42.
    #    iter 5: kept=4 ≥ 3, 42+10.5=52.5 > 50, BREAK.
    # → 4 hits.
    assert len(out) == 4, f"expected 4 hits at 10.5k each, got {len(out)}"


def test_run_one_question_error_row_carries_stage1_fields(monkeypatch):
    """Reviewer-flagged regression: error row had been dropping
    n_hits_pre_trim, top_score, abstention_fired, etc. Restored Stage 1 fix."""
    from scripts import run_longmemeval as adapter

    fake_hits = [
        {"uuid": "lme_q_qX__001_x", "content": "y" * 100,
         "score": 0.5, "similarity": 0.5,
         "referenced_date": "2024-01-01T00:00:00"},
        {"uuid": "lme_q_qX__002_y", "content": "z" * 100,
         "score": 0.4, "similarity": 0.4,
         "referenced_date": "2024-01-02T00:00:00"},
    ]
    monkeypatch.setattr(adapter, "ingest_question_haystack", lambda **kw: 2)
    monkeypatch.setattr(adapter, "retrieve_with_omega_recipe",
                        lambda **kw: list(fake_hits))

    def boom(**kw):
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(adapter, "call_llm", boom)

    q = {"question_id": "qX", "question": "what did i do",
         "question_date": "", "haystack_sessions": [],
         "haystack_session_ids": [], "haystack_dates": []}
    row = adapter.run_one_question(
        q=q, answerer="gpt41", driver=None, embedding_store=None,
        chroma_collection=None, oracle_category="multi-session",
    )
    assert row["error"]
    # All Stage 1 + earlier observability fields must be present.
    for f in ("n_sessions_ingested", "n_hits_used", "n_hits_pre_trim",
              "top_score", "abstention_fired", "max_tokens",
              "category_source", "shadow_classifier_label", "classifier_rule",
              "counting", "seed_honored"):
        assert f in row, f"error row missing field {f!r}"
    assert row["n_hits_pre_trim"] == 2  # retrieval succeeded before crash
    assert row["n_sessions_ingested"] == 2


def test_write_run_summary_aggregates_stage1_fields(tmp_path):
    """Summary now reports abstention_fired, category_source, trim averages."""
    from scripts.run_longmemeval import write_run_summary
    out = tmp_path / "run.jsonl"
    out.write_text(
        json.dumps({"question_id": "q1", "predicted_category": "multi-session",
                    "answerer": "gpt41", "elapsed_sec": 1.0,
                    "abstention_fired": True, "category_source": "oracle",
                    "n_hits_pre_trim": 20, "n_hits_used": 20}) + "\n"
        + json.dumps({"question_id": "q2", "predicted_category": "single-session-user",
                      "answerer": "gpt41", "elapsed_sec": 1.0,
                      "abstention_fired": False, "category_source": "oracle",
                      "n_hits_pre_trim": 12, "n_hits_used": 4}) + "\n"
    )
    s = json.loads(write_run_summary(out).read_text())
    assert s["abstention_fired_total"] == 1
    assert s["abstention_fired_pct"] == 0.5
    assert s["category_source"] == {"oracle": 2}
    assert s["avg_hits_pre_trim"] == 16.0
    assert s["avg_hits_used"] == 12.0


def test_trim_to_context_budget_handles_missing_score_field():
    """A hit with no score sorts to bottom; min_hits floor still applies."""
    from scripts.run_longmemeval import trim_to_context_budget
    hits = [
        {"uuid": "scored", "content": "x" * 100, "score": 0.5, "referenced_date": "2024-01-01"},
        {"uuid": "unscored", "content": "y" * 100, "referenced_date": "2024-02-01"},
    ]
    out = trim_to_context_budget(hits, "multi-session", budget_chars=1000)
    assert len(out) == 2  # both fit


def test_trim_appears_in_run_one_question_row(monkeypatch):
    """run_one_question records n_hits_pre_trim and trims at SS budget."""
    from scripts import run_longmemeval as adapter

    # Ten 10K-char hits → 100K total. SS budget=50K, min_hits=3.
    # After iter 1..4 cumulative is 10K, 20K, 30K, 40K (≤50K, all kept).
    # iter 5: total=40K, c=10K, 40K+10K=50K > 50K is False — kept.
    # iter 6: total=50K, c=10K, 50K+10K=60K > 50K is True, kept≥3 — break.
    # → 5 hits.
    fake_hits = [
        {"uuid": f"lme_q_qX__{i:03d}_x", "content": "x" * 10000,
         "score": 1.0 - i * 0.01,
         "referenced_date": f"2024-01-{i+1:02d}T00:00:00"}
        for i in range(10)
    ]

    monkeypatch.setattr(adapter, "ingest_question_haystack", lambda **kw: 10)
    monkeypatch.setattr(adapter, "retrieve_with_omega_recipe",
                        lambda **kw: list(fake_hits))
    monkeypatch.setattr(adapter, "call_llm", lambda **kw: "answer")

    q = {"question_id": "qX", "question": "did i do the thing",
         "question_date": "", "haystack_sessions": [], "haystack_session_ids": [],
         "haystack_dates": []}
    row = adapter.run_one_question(
        q=q, answerer="gpt41", driver=None, embedding_store=None,
        chroma_collection=None,
        oracle_category="single-session-user",
    )
    assert row["n_hits_pre_trim"] == 10
    assert row["n_hits_used"] == 5


def test_run_one_question_oracle_category_survives_generation_crash(monkeypatch):
    """If generation crashes, error row still records oracle category source."""
    from scripts import run_longmemeval as adapter

    monkeypatch.setattr(adapter, "ingest_question_haystack", lambda **kw: 0)
    monkeypatch.setattr(adapter, "retrieve_with_omega_recipe", lambda **kw: [])

    def boom(**kw):
        raise RuntimeError("oops")

    monkeypatch.setattr(adapter, "call_llm", boom)

    q = {"question_id": "qZ", "question": "did i do the thing",
         "question_date": "", "haystack_sessions": [], "haystack_session_ids": [],
         "haystack_dates": []}

    row = adapter.run_one_question(
        q=q,
        answerer="gpt41",
        driver=None,
        embedding_store=None,
        chroma_collection=None,
        oracle_category="temporal-reasoning",
    )
    assert row["error"]
    assert row["predicted_category"] == "temporal-reasoning"
    assert row["category_source"] == "oracle"
