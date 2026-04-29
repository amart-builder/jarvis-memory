#!/usr/bin/env python3
"""Parallel targeted-validation harness for LongMemEval.

This wraps ``scripts/run_longmemeval.py`` with process-level sharding while
preserving score safety:

* each worker gets a distinct Neo4j label;
* each worker gets a distinct Chroma persistent path;
* each worker writes its own JSONL/log files;
* the coordinator validates no missing/duplicate qids before merging;
* optional parity checking compares prompt hashes and ordered retrieval hits
  against a serial run.

Use this after the serial adapter has a stable implementation. It is a
wall-clock optimization, not a scoring change.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_longmemeval import write_run_summary  # noqa: E402
from scripts.run_targeted_validation import (  # noqa: E402
    _DEFAULT_BASELINE_EVAL,
    _DEFAULT_BASELINE_JSONL,
    _DEFAULT_DATASET,
    _DEFAULT_JUDGE_MODEL,
    _DEFAULT_JUDGE_SCRIPT,
    _DEFAULT_ORACLE,
    _DEFAULT_REGRESSION_SAMPLE,
    _DEFAULT_SEED,
    _PARALLEL_JUDGE_SCRIPT,
    parse_baseline_eval,
    parse_judge_output,
    print_delta_report,
    run_judge,
)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    qids: list[str]
    dataset_path: Path
    output_path: Path
    log_path: Path
    chroma_path: Path
    neo4j_label: str
    chroma_collection: str
    group_prefix: str


def load_env_file(env_file: Path | None = None) -> list[str]:
    """Load repo-local .env values so subprocess launches are self-contained."""
    path = env_file or (REPO_ROOT / ".env")
    if not path.exists():
        return []

    loaded: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
        loaded.append(key)
    return loaded


def required_env_vars(*,
                      answerer: str,
                      run_adapter: bool,
                      run_judge: bool,
                      judge_model: str) -> set[str]:
    required: set[str] = set()
    answerer_key = answerer.strip().lower()
    if run_adapter:
        if answerer_key in {"gpt4o", "gpt41"}:
            required.add("OPENAI_API_KEY")
        elif answerer_key in {"opus", "claude", "claude-opus"}:
            required.add("ANTHROPIC_API_KEY")

    judge_key = judge_model.strip().lower()
    if run_judge and (judge_key.startswith("gpt") or judge_key.startswith("o")):
        required.add("OPENAI_API_KEY")
    return required


def validate_required_env(*,
                          answerer: str,
                          run_adapter: bool,
                          run_judge: bool,
                          judge_model: str) -> None:
    missing = sorted(
        key for key in required_env_vars(
            answerer=answerer,
            run_adapter=run_adapter,
            run_judge=run_judge,
            judge_model=judge_model,
        )
        if not os.environ.get(key)
    )
    if missing:
        raise RuntimeError(
            "missing required environment variables after loading repo .env: "
            + ", ".join(missing)
        )


def _json_load(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _slug(text: str, *, max_len: int = 40, sep: str = "_") -> str:
    out = re.sub(r"[^A-Za-z0-9]+", sep, text).strip(sep).lower()
    if not out:
        out = "run"
    return out[:max_len].strip(sep) or "run"


def _neo4j_label(base: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_]", "_", base)
    if not re.match(r"[A-Za-z]", label):
        label = f"L{label}"
    return label


def canonical_qids(dataset: list[dict[str, Any]], keep: set[str]) -> list[str]:
    """Return target qids in source-dataset order."""
    return [q["question_id"] for q in dataset if q.get("question_id") in keep]


def full_dataset_qids(dataset: list[dict[str, Any]]) -> list[str]:
    """Return every qid in source-dataset order for headline runs."""
    return [str(q["question_id"]) for q in dataset if q.get("question_id")]


def shard_qids(qids: list[str], workers: int) -> list[list[str]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    shards = [[] for _ in range(workers)]
    for idx, qid in enumerate(qids):
        shards[idx % workers].append(qid)
    return shards


def write_subset(dataset: list[dict[str, Any]], qids: list[str], path: Path) -> None:
    qid_set = set(qids)
    subset = [q for q in dataset if q.get("question_id") in qid_set]
    ordered = {q.get("question_id"): q for q in subset}
    missing = [qid for qid in qids if qid not in ordered]
    if missing:
        raise ValueError(f"worker subset missing qids from dataset: {missing[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([ordered[qid] for qid in qids]))


def build_worker_specs(*,
                       output: Path,
                       dataset: list[dict[str, Any]],
                       qids: list[str],
                       workers: int,
                       run_id: str | None = None) -> tuple[Path, list[WorkerSpec]]:
    run_slug = _slug(run_id or f"{output.stem}_{os.getpid()}", max_len=32)
    label_slug = _slug(run_slug, max_len=24)
    coll_slug = _slug(run_slug, max_len=24, sep="-")
    work_dir = output.parent / f"{output.stem}.parallel" / run_slug
    shards = shard_qids(qids, workers)

    specs: list[WorkerSpec] = []
    for worker_id, worker_qids in enumerate(shards):
        worker_dir = work_dir / f"worker_{worker_id}"
        neo4j_label = _neo4j_label(f"LMETestEpisode_{label_slug}_w{worker_id}")
        spec = WorkerSpec(
            worker_id=worker_id,
            qids=worker_qids,
            dataset_path=worker_dir / "dataset.json",
            output_path=worker_dir / "output.jsonl",
            log_path=worker_dir / "adapter.log",
            chroma_path=worker_dir / "chroma",
            neo4j_label=neo4j_label,
            chroma_collection=f"jarvis-lme-{coll_slug}-w{worker_id}",
            group_prefix=f"lme_{label_slug}_w{worker_id}_q",
        )
        write_subset(dataset, worker_qids, spec.dataset_path)
        specs.append(spec)
    return work_dir, specs


def run_workers(*,
                specs: list[WorkerSpec],
                answerer: str,
                extra_flags: list[str]) -> int:
    procs: list[tuple[WorkerSpec, subprocess.Popen]] = []
    for spec in specs:
        if not spec.qids:
            continue
        env = os.environ.copy()
        env.update({
            "PYTHONHASHSEED": "42",
            "JARVIS_LME_NEO4J_LABEL": spec.neo4j_label,
            "JARVIS_LME_CHROMA_PATH": str(spec.chroma_path),
            "JARVIS_LME_CHROMA_COLLECTION": spec.chroma_collection,
            "JARVIS_LME_GROUP_PREFIX": spec.group_prefix,
        })
        cmd = [
            sys.executable,
            str(Path("scripts") / "run_longmemeval.py"),
            "--output", str(spec.output_path),
            "--dataset", str(spec.dataset_path),
            "--answerer", answerer,
            *extra_flags,
        ]
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = spec.log_path.open("w")
        print(
            f"worker {spec.worker_id}: {len(spec.qids)} qids, "
            f"label={spec.neo4j_label}, chroma={spec.chroma_path}"
        )
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.close()
        procs.append((spec, proc))

    failed = False
    for spec, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed = True
            print(f"ERROR: worker {spec.worker_id} exited {rc}; log={spec.log_path}", file=sys.stderr)
            tail = spec.log_path.read_text(errors="replace").splitlines()[-40:]
            if tail:
                print("\n".join(tail), file=sys.stderr)
    return 1 if failed else 0


def merge_worker_outputs(specs: list[WorkerSpec], qids: list[str], output: Path) -> dict[str, Any]:
    expected = set(qids)
    rows_by_qid: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    extras: list[str] = []

    for spec in specs:
        rows = _read_jsonl(spec.output_path)
        for row in rows:
            qid = row.get("question_id")
            if qid not in expected:
                extras.append(str(qid))
                continue
            if qid in rows_by_qid:
                duplicates.append(str(qid))
            rows_by_qid[str(qid)] = row

    missing = [qid for qid in qids if qid not in rows_by_qid]
    if missing or duplicates or extras:
        raise ValueError(
            "parallel merge failed: "
            f"missing={missing[:10]} duplicates={duplicates[:10]} extras={extras[:10]}"
        )

    merged = [rows_by_qid[qid] for qid in qids]
    _write_jsonl(output, merged)
    return {
        "n_rows": len(merged),
        "qids": qids,
        "worker_outputs": [str(s.output_path) for s in specs],
    }


def _normalize_lme_uuid(uid: str) -> str:
    """Strip eval group/id prefix so serial and worker UUIDs compare equal."""
    if "__" not in uid:
        return uid
    tail = uid.split("__", 1)[1]
    return re.sub(r"^\d{3}_", "", tail)


def _row_parity_signature(row: dict[str, Any], *, strict_hypothesis: bool) -> dict[str, Any]:
    diagnostics = row.get("diagnostics") or {}
    final_hit_uuids = diagnostics.get("final_hit_uuids")
    prompt_hash = diagnostics.get("prompt_hash")
    if not prompt_hash or not isinstance(final_hit_uuids, list):
        raise ValueError(
            f"row {row.get('question_id')} lacks diagnostics.prompt_hash/final_hit_uuids; "
            "run both serial and parallel with --diagnostics"
        )

    sig = {
        "question_id": row.get("question_id"),
        "predicted_category": row.get("predicted_category"),
        "n_hits_used": row.get("n_hits_used"),
        "n_hits_pre_trim": row.get("n_hits_pre_trim"),
        "prompt_hash": prompt_hash,
        "final_hit_session_ids": [_normalize_lme_uuid(str(uid)) for uid in final_hit_uuids],
        "pipeline_stage_ranks": diagnostics.get("pipeline_stage_ranks"),
        "pipeline_stage_sizes": diagnostics.get("pipeline_stage_sizes"),
    }
    if strict_hypothesis:
        sig["hypothesis"] = row.get("hypothesis")
    return sig


def compare_parity(serial_path: Path,
                   parallel_path: Path,
                   *,
                   strict_hypothesis: bool = False) -> dict[str, Any]:
    serial_rows = _read_jsonl(serial_path)
    parallel_rows = _read_jsonl(parallel_path)
    serial_qids = [str(r.get("question_id")) for r in serial_rows]
    parallel_qids = [str(r.get("question_id")) for r in parallel_rows]

    report: dict[str, Any] = {
        "serial_path": str(serial_path),
        "parallel_path": str(parallel_path),
        "strict_hypothesis": strict_hypothesis,
        "ok": True,
        "mismatches": [],
    }
    if serial_qids != parallel_qids:
        report["ok"] = False
        report["mismatches"].append({
            "kind": "qid_order",
            "serial_qids": serial_qids,
            "parallel_qids": parallel_qids,
        })
        return report

    for serial, parallel in zip(serial_rows, parallel_rows):
        qid = serial.get("question_id")
        serial_sig = _row_parity_signature(serial, strict_hypothesis=strict_hypothesis)
        parallel_sig = _row_parity_signature(parallel, strict_hypothesis=strict_hypothesis)
        if serial_sig != parallel_sig:
            report["ok"] = False
            report["mismatches"].append({
                "kind": "row",
                "question_id": qid,
                "serial": serial_sig,
                "parallel": parallel_sig,
            })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline-jsonl", type=Path, default=_DEFAULT_BASELINE_JSONL)
    parser.add_argument("--baseline-eval", type=Path, default=_DEFAULT_BASELINE_EVAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--oracle", type=Path, default=_DEFAULT_ORACLE)
    parser.add_argument("--judge-script", type=Path, default=_DEFAULT_JUDGE_SCRIPT)
    parser.add_argument("--judge-model", type=str, default=_DEFAULT_JUDGE_MODEL)
    parser.add_argument("--answerer", type=str, default="gpt41")
    parser.add_argument("--regression-sample", type=int, default=_DEFAULT_REGRESSION_SAMPLE)
    parser.add_argument("--categories", type=str, default="")
    parser.add_argument("--full-dataset", action="store_true",
                        help="Run every question in --dataset instead of the targeted wrongs+sample gate.")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--adapter-workers", type=int, default=4)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--use-oracle-categories", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--parity-against", type=Path, default=None,
                        help="Serial JSONL to compare against after merge. Requires --diagnostics on both runs.")
    parser.add_argument("--strict-hypothesis", action="store_true",
                        help="Also require hypothesis text equality in --parity-against mode.")
    parser.add_argument("--cleanup-workdir", action="store_true",
                        help="Delete per-worker datasets/logs/Chroma after merge and optional parity check.")
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument("--parallel-judge", dest="parallel_judge", action="store_true", default=True)
    judge_group.add_argument("--sequential-judge", dest="parallel_judge", action="store_false")
    args = parser.parse_args()

    loaded_env = load_env_file()
    if loaded_env:
        print(f"Loaded repo .env from {REPO_ROOT / '.env'}")

    if not args.dry_run:
        try:
            validate_required_env(
                answerer=args.answerer,
                run_adapter=not args.skip_run,
                run_judge=not args.skip_judge,
                judge_model=args.judge_model,
            )
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    if not args.baseline_eval.exists():
        print(f"ERROR: baseline eval not found: {args.baseline_eval}", file=sys.stderr)
        return 2
    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    cats_filter: set[str] | None = None
    if args.categories.strip():
        cats_filter = {c.strip() for c in args.categories.split(",") if c.strip()}
        print(f"Filtering to categories: {sorted(cats_filter)}")

    dataset = _json_load(args.dataset)
    wrongs, rights = parse_baseline_eval(args.baseline_eval, cats_filter)

    if args.full_dataset:
        sampled: set[str] = set()
        qids = full_dataset_qids(dataset)
        print(f"Baseline: {len(wrongs)} wrongs, {len(rights)} rights in scope")
        print(f"Full dataset: {len(qids)} questions")
    else:
        rng = random.Random(args.seed)
        rights_qids = sorted(rights.keys())
        n_sample = min(args.regression_sample, len(rights_qids))
        sampled = set(rng.sample(rights_qids, n_sample))
        target_qid_set = set(wrongs.keys()) | sampled

        qids = canonical_qids(dataset, target_qid_set)
        if len(qids) != len(target_qid_set):
            missing = sorted(target_qid_set - set(qids))
            print(
                f"WARN: target had {len(target_qid_set)} qids but "
                f"{len(missing)} missing from dataset",
                file=sys.stderr,
            )

        print(f"Baseline: {len(wrongs)} wrongs, {len(rights)} rights in scope")
        print(f"Regression sample: {n_sample} of {len(rights_qids)} rights")
        print(f"Target subset: {len(qids)} questions")

    extra: list[str] = []
    if args.use_oracle_categories:
        extra.append("--use-oracle-categories")
    if args.diagnostics:
        extra.append("--diagnostics")

    work_dir, specs = build_worker_specs(
        output=args.output,
        dataset=dataset,
        qids=qids,
        workers=args.adapter_workers,
        run_id=args.run_id,
    )
    manifest_path = work_dir / "manifest.json"
    manifest = {
        "output": str(args.output),
        "work_dir": str(work_dir),
        "adapter_workers": args.adapter_workers,
        "worker_isolation": [spec.__dict__ for spec in specs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Work dir: {work_dir}")
    print(f"Manifest: {manifest_path}")

    if args.dry_run:
        print("Dry run only; no workers launched.")
        return 0

    if not args.skip_run:
        rc = run_workers(specs=specs, answerer=args.answerer, extra_flags=extra)
        if rc != 0:
            return rc
        merge_worker_outputs(specs, qids, args.output)
        try:
            summary_path = write_run_summary(args.output)
            print(f"Summary: {summary_path}")
        except Exception as e:  # noqa: BLE001
            print(f"WARN: failed to write run summary: {e}", file=sys.stderr)
    else:
        if not args.output.exists():
            print(f"ERROR: --skip-run set but output does not exist: {args.output}", file=sys.stderr)
            return 2

    if args.parity_against is not None:
        report = compare_parity(
            args.parity_against,
            args.output,
            strict_hypothesis=args.strict_hypothesis,
        )
        report_path = args.output.parent / f"{args.output.stem}.parity.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if not report["ok"]:
            print(f"ERROR: parity check failed; see {report_path}", file=sys.stderr)
            return 1
        print(f"Parity check passed: {report_path}")

    if args.skip_judge:
        if args.cleanup_workdir:
            shutil.rmtree(work_dir)
            print(f"Cleaned worker dir: {work_dir}")
        print(f"Skipped judge. Output: {args.output}")
        return 0

    judge_out = args.output.parent / (args.output.name + ".eval-results-" + args.judge_model)
    judge_script = _PARALLEL_JUDGE_SCRIPT if args.parallel_judge else args.judge_script
    rc = run_judge(
        judge_script=judge_script,
        judge_model=args.judge_model,
        hyp_jsonl=args.output,
        oracle_path=args.oracle,
        parallel=args.parallel_judge,
        workers=args.judge_workers,
    )
    if rc != 0:
        return rc

    if not judge_out.exists():
        print(f"ERROR: expected judge output not found: {judge_out}", file=sys.stderr)
        return 1

    labels = parse_judge_output(judge_out)
    if args.full_dataset:
        n_right = sum(1 for qid in qids if labels.get(qid, False))
        n_total = len(qids)
        pct = (100 * n_right / n_total) if n_total else 0.0
        missing = [qid for qid in qids if qid not in labels]
        if missing:
            print(f"WARN: judge output missing {len(missing)} qids", file=sys.stderr)
        print()
        print("=" * 72)
        print("FULL DATASET RESULT")
        print("=" * 72)
        print(f"RIGHT {n_right} / {n_total} = {pct:.2f}%")
        print("=" * 72)
    else:
        print_delta_report(
            wrongs=wrongs,
            rights=rights,
            sampled_rights=sampled,
            new_labels=labels,
        )

    if args.cleanup_workdir:
        shutil.rmtree(work_dir)
        print(f"Cleaned worker dir: {work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
