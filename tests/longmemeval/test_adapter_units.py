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
