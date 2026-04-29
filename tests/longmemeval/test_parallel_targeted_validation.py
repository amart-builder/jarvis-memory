from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.run_parallel_targeted_validation import (
    build_worker_specs,
    canonical_qids,
    compare_parity,
    full_dataset_qids,
    load_env_file,
    merge_worker_outputs,
    required_env_vars,
    shard_qids,
    validate_required_env,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_shard_qids_is_deterministic_round_robin():
    assert shard_qids(["q1", "q2", "q3", "q4", "q5"], 2) == [
        ["q1", "q3", "q5"],
        ["q2", "q4"],
    ]


def test_load_env_file_populates_missing_values_without_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=file-openai\n"
        "export ANTHROPIC_API_KEY='file-anthropic'\n"
        "EXISTING=file-existing\n"
        "MALFORMED\n"
        "1BAD=ignored\n"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "parent-existing")

    loaded = load_env_file(env_file)

    assert "OPENAI_API_KEY" in loaded
    assert "ANTHROPIC_API_KEY" in loaded
    assert "1BAD" not in loaded
    assert loaded.count("EXISTING") == 1
    assert os.environ["OPENAI_API_KEY"] == "file-openai"
    assert os.environ["ANTHROPIC_API_KEY"] == "file-anthropic"
    assert os.environ["EXISTING"] == "parent-existing"


def test_required_env_vars_respects_skip_flags():
    assert required_env_vars(
        answerer="gpt41",
        run_adapter=False,
        run_judge=False,
        judge_model="gpt-4o",
    ) == set()


def test_validate_required_env_fails_fast_for_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        validate_required_env(
            answerer="gpt41",
            run_adapter=True,
            run_judge=True,
            judge_model="gpt-4o",
        )


def test_canonical_qids_preserves_dataset_order():
    dataset = [
        {"question_id": "q3"},
        {"question_id": "q1"},
        {"question_id": "q2"},
    ]
    assert canonical_qids(dataset, {"q1", "q2"}) == ["q1", "q2"]


def test_full_dataset_qids_preserves_dataset_order():
    dataset = [
        {"question_id": "q3"},
        {"question_id": "q1"},
        {"question_id": "q2"},
    ]
    assert full_dataset_qids(dataset) == ["q3", "q1", "q2"]


def test_build_worker_specs_writes_isolated_worker_datasets(tmp_path):
    dataset = [{"question_id": f"q{i}", "question": str(i)} for i in range(4)]
    out = tmp_path / "targeted.jsonl"
    work_dir, specs = build_worker_specs(
        output=out,
        dataset=dataset,
        qids=["q0", "q1", "q2", "q3"],
        workers=2,
        run_id="Parity Run!",
    )

    assert work_dir.name == "parity_run"
    assert len(specs) == 2
    assert specs[0].qids == ["q0", "q2"]
    assert specs[1].qids == ["q1", "q3"]
    assert specs[0].neo4j_label != specs[1].neo4j_label
    assert specs[0].chroma_path != specs[1].chroma_path
    assert specs[0].chroma_collection != specs[1].chroma_collection

    worker0 = json.loads(specs[0].dataset_path.read_text())
    assert [row["question_id"] for row in worker0] == ["q0", "q2"]


def test_merge_worker_outputs_rejects_missing_duplicate_or_extra_qids(tmp_path):
    dataset = [{"question_id": "q1"}, {"question_id": "q2"}]
    _, specs = build_worker_specs(
        output=tmp_path / "out.jsonl",
        dataset=dataset,
        qids=["q1", "q2"],
        workers=2,
        run_id="merge",
    )
    _write_jsonl(specs[0].output_path, [{"question_id": "q1"}])
    _write_jsonl(specs[1].output_path, [{"question_id": "q1"}, {"question_id": "qx"}])

    with pytest.raises(ValueError, match="missing"):
        merge_worker_outputs(specs, ["q1", "q2"], tmp_path / "merged.jsonl")


def test_merge_worker_outputs_writes_canonical_order(tmp_path):
    dataset = [{"question_id": "q1"}, {"question_id": "q2"}, {"question_id": "q3"}]
    merged = tmp_path / "merged.jsonl"
    _, specs = build_worker_specs(
        output=tmp_path / "out.jsonl",
        dataset=dataset,
        qids=["q1", "q2", "q3"],
        workers=2,
        run_id="merge-ok",
    )
    _write_jsonl(specs[0].output_path, [{"question_id": "q1"}, {"question_id": "q3"}])
    _write_jsonl(specs[1].output_path, [{"question_id": "q2"}])

    result = merge_worker_outputs(specs, ["q1", "q2", "q3"], merged)
    assert result["n_rows"] == 3
    rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert [row["question_id"] for row in rows] == ["q1", "q2", "q3"]


def test_compare_parity_normalizes_worker_uuid_prefixes(tmp_path):
    serial = tmp_path / "serial.jsonl"
    parallel = tmp_path / "parallel.jsonl"
    base = {
        "question_id": "q1",
        "predicted_category": "multi-session",
        "n_hits_used": 2,
        "n_hits_pre_trim": 4,
        "diagnostics": {
            "prompt_hash": "abc123",
            "final_hit_uuids": [
                "lme_q_q1__000_session-a",
                "lme_q_q1__001_session-b",
            ],
            "pipeline_stage_ranks": {"filtered": {"session-a": 1}},
            "pipeline_stage_sizes": {"filtered": 2},
        },
    }
    worker = json.loads(json.dumps(base))
    worker["diagnostics"]["final_hit_uuids"] = [
        "lme_run_w0_q_q1__000_session-a",
        "lme_run_w0_q_q1__001_session-b",
    ]
    _write_jsonl(serial, [base])
    _write_jsonl(parallel, [worker])

    report = compare_parity(serial, parallel)
    assert report["ok"] is True


def test_compare_parity_reports_prompt_hash_mismatch(tmp_path):
    serial = tmp_path / "serial.jsonl"
    parallel = tmp_path / "parallel.jsonl"
    row = {
        "question_id": "q1",
        "predicted_category": "multi-session",
        "n_hits_used": 1,
        "n_hits_pre_trim": 1,
        "diagnostics": {
            "prompt_hash": "one",
            "final_hit_uuids": ["lme_q_q1__000_session-a"],
        },
    }
    other = json.loads(json.dumps(row))
    other["diagnostics"]["prompt_hash"] = "two"
    _write_jsonl(serial, [row])
    _write_jsonl(parallel, [other])

    report = compare_parity(serial, parallel)
    assert report["ok"] is False
    assert report["mismatches"][0]["question_id"] == "q1"
