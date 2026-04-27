#!/usr/bin/env python3
"""Targeted validation harness for LongMemEval stages.

Cheap stand-in for a full 500-question run between stages. Re-runs:
  1. All baseline wrongs in the categories the stage touches (the
     failures we're trying to fix).
  2. A small random sample of previously-correct questions in the
     same categories (regression check — catches the case where a
     stage "fixes" 25 wrongs but breaks 10 rights, which a wrongs-
     only run would miss).

Then runs the judge on the targeted output and prints a delta vs
baseline.

Per-stage cost: ~$3-5 in API calls (vs $33 for full 500q). Saves ~$28
and ~2 hrs of wall time per stage when iterating on prompts/retrieval.

Usage:
    python scripts/run_targeted_validation.py \\
        --baseline-jsonl runs/lme_gpt41_v1.1.jsonl \\
        --baseline-eval runs/lme_gpt41_v1.1.jsonl.eval-results-gpt-4o \\
        --output runs/lme_gpt41_stage1_targeted.jsonl \\
        --answerer gpt41 \\
        --use-oracle-categories --diagnostics

The harness invokes the existing adapter via subprocess so it
inherits all the Stage 0/1 flags and the deterministic re-exec path
without re-implementing them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path


_DEFAULT_BASELINE_JSONL = Path("runs/lme_gpt41_v1.1.jsonl")
_DEFAULT_BASELINE_EVAL = Path("runs/lme_gpt41_v1.1.jsonl.eval-results-gpt-4o")
_DEFAULT_DATASET = Path("data/longmemeval/longmemeval_s_cleaned.json")
_DEFAULT_ORACLE = Path("data/longmemeval/longmemeval_oracle.json")
_DEFAULT_JUDGE_SCRIPT = Path("/tmp/lme-official/src/evaluation/evaluate_qa.py")
_DEFAULT_JUDGE_MODEL = "gpt-4o"
_DEFAULT_REGRESSION_SAMPLE = 30
_DEFAULT_SEED = 42


def parse_baseline_eval(eval_path: Path,
                        category_filter: set[str] | None = None,
                        ) -> tuple[dict[str, str], dict[str, str]]:
    """Read the judge's eval-results JSONL.

    Returns ``(wrongs_by_qid, rights_by_qid)`` where each maps
    ``question_id`` → ``predicted_category``. ``category_filter``,
    when provided, restricts both maps to questions whose baseline
    predicted_category is in the set.
    """
    wrongs: dict[str, str] = {}
    rights: dict[str, str] = {}
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            cat = row.get("predicted_category", "unknown")
            label = row.get("autoeval_label", {}).get("label", False)
            if not qid:
                continue
            if category_filter and cat not in category_filter:
                continue
            if label:
                rights[qid] = cat
            else:
                wrongs[qid] = cat
    return wrongs, rights


def write_subset_dataset(dataset_path: Path,
                         keep_qids: set[str],
                         out_path: Path) -> int:
    """Write a copy of the dataset filtered to ``keep_qids``."""
    with dataset_path.open() as f:
        full = json.load(f)
    subset = [q for q in full if q.get("question_id") in keep_qids]
    out_path.write_text(json.dumps(subset))
    return len(subset)


def parse_judge_output(eval_path: Path) -> dict[str, bool]:
    """Map question_id → judge label (True/False)."""
    out: dict[str, bool] = {}
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if qid:
                out[qid] = bool(row.get("autoeval_label", {}).get("label", False))
    return out


def run_adapter(*,
                output_path: Path,
                dataset_path: Path,
                answerer: str,
                extra_flags: list[str]) -> int:
    """Invoke the existing adapter via subprocess."""
    cmd = [
        sys.executable,
        str(Path("scripts") / "run_longmemeval.py"),
        "--output", str(output_path),
        "--dataset", str(dataset_path),
        "--answerer", answerer,
        *extra_flags,
    ]
    print(f"\n→ Running adapter: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def run_judge(*,
              judge_script: Path,
              judge_model: str,
              hyp_jsonl: Path,
              oracle_path: Path) -> int:
    """Invoke the official LongMemEval judge."""
    cmd = [
        sys.executable,
        str(judge_script),
        judge_model,
        str(hyp_jsonl),
        str(oracle_path),
    ]
    print(f"\n→ Running judge: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def print_delta_report(*,
                       wrongs: dict[str, str],
                       rights: dict[str, str],
                       sampled_rights: set[str],
                       new_labels: dict[str, bool]) -> dict:
    """Compute and print the fix/still-wrong/regress/stable buckets."""
    fixed, still_wrong = [], []
    regressed, stable = [], []

    for qid in wrongs:
        if qid not in new_labels:
            continue  # Not in the targeted run — shouldn't happen
        if new_labels[qid]:
            fixed.append(qid)
        else:
            still_wrong.append(qid)

    for qid in sampled_rights:
        if qid not in new_labels:
            continue
        if new_labels[qid]:
            stable.append(qid)
        else:
            regressed.append(qid)

    n_w = len(wrongs)
    n_r = len(sampled_rights)

    def _pct(num: int, denom: int) -> str:
        return f"{100*num/denom:.0f}%" if denom > 0 else "n/a"

    print()
    print("=" * 72)
    print("TARGETED VALIDATION DELTA")
    print("=" * 72)
    print(f"FIXED         {len(fixed):>3} of {n_w:>3} baseline wrongs    ({_pct(len(fixed), n_w)})")
    print(f"STILL WRONG   {len(still_wrong):>3} of {n_w:>3} baseline wrongs    ({_pct(len(still_wrong), n_w)})")
    print(f"REGRESSED     {len(regressed):>3} of {n_r:>3} regression sample  ({_pct(len(regressed), n_r)})")
    print(f"STABLE        {len(stable):>3} of {n_r:>3} regression sample  ({_pct(len(stable), n_r)})")

    # Per-category breakdown of fixes
    fixed_by_cat: Counter[str] = Counter()
    still_by_cat: Counter[str] = Counter()
    for qid in fixed:
        fixed_by_cat[wrongs[qid]] += 1
    for qid in still_wrong:
        still_by_cat[wrongs[qid]] += 1

    print()
    print("By baseline predicted_category (wrongs only):")
    print(f"  {'category':<28} {'fixed':>6} {'still':>6} {'total':>6}")
    cats = sorted(set(list(fixed_by_cat) + list(still_by_cat)))
    for cat in cats:
        f, s = fixed_by_cat[cat], still_by_cat[cat]
        print(f"  {cat:<28} {f:>6} {s:>6} {f+s:>6}")

    if regressed:
        print()
        print("⚠ Regressions (these were RIGHT in baseline, are WRONG now):")
        for qid in regressed:
            print(f"  - {qid} (was {rights[qid]})")

    print("=" * 72)
    return {
        "fixed": fixed,
        "still_wrong": still_wrong,
        "regressed": regressed,
        "stable": stable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-jsonl", type=Path, default=_DEFAULT_BASELINE_JSONL,
                        help="Baseline run output (where the wrongs come from).")
    parser.add_argument("--baseline-eval", type=Path, default=_DEFAULT_BASELINE_EVAL,
                        help="Baseline judge eval-results JSONL.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL for the targeted re-run.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET,
                        help="Source dataset (s_cleaned.json) — gets subsetted to target qids.")
    parser.add_argument("--oracle", type=Path, default=_DEFAULT_ORACLE,
                        help="Oracle JSON for judge ground truth.")
    parser.add_argument("--judge-script", type=Path, default=_DEFAULT_JUDGE_SCRIPT,
                        help="Path to the LongMemEval evaluate_qa.py script.")
    parser.add_argument("--judge-model", type=str, default=_DEFAULT_JUDGE_MODEL,
                        help="Judge model id (default: gpt-4o; the published baseline).")
    parser.add_argument("--answerer", type=str, default="gpt41",
                        help="Answerer for the re-run: opus|gpt4o|gpt41 (default gpt41).")
    parser.add_argument("--regression-sample", type=int, default=_DEFAULT_REGRESSION_SAMPLE,
                        help="Number of previously-correct questions to spot-check for regressions.")
    parser.add_argument("--categories", type=str, default="",
                        help="Comma-separated list of baseline predicted_categories to focus on. "
                             "Empty = all 6 categories.")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED,
                        help="Seed for the regression-sample selection (default 42 — matches RUN_SEED).")
    parser.add_argument("--use-oracle-categories", action="store_true",
                        help="Pass through to the adapter.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Pass through to the adapter.")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip the adapter+judge calls and only compute the delta from "
                             "an existing output. Useful when a run completed but printing crashed.")
    args = parser.parse_args()

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

    wrongs, rights = parse_baseline_eval(args.baseline_eval, cats_filter)
    print(f"Baseline: {len(wrongs)} wrongs, {len(rights)} rights "
          f"({len(wrongs) + len(rights)} total in scope)")

    # Sample previously-correct questions for the regression check.
    rng = random.Random(args.seed)
    rights_qids = sorted(rights.keys())  # sorted for determinism with seed
    n_sample = min(args.regression_sample, len(rights_qids))
    sampled = set(rng.sample(rights_qids, n_sample))
    print(f"Regression sample: {n_sample} of {len(rights_qids)} rights")

    target_qids = set(wrongs.keys()) | sampled
    print(f"Target subset: {len(target_qids)} questions to re-run")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    judge_out = args.output.parent / (args.output.name + ".eval-results-" + args.judge_model)

    if not args.skip_run:
        # Subset the dataset to the target qids, drop in /tmp.
        subset_path = Path("/tmp") / f"lme_targeted_subset_{os.getpid()}.json"
        try:
            n_in_subset = write_subset_dataset(args.dataset, target_qids, subset_path)
            if n_in_subset != len(target_qids):
                print(f"WARN: target had {len(target_qids)} qids but only {n_in_subset} found in dataset",
                      file=sys.stderr)

            extra: list[str] = []
            if args.use_oracle_categories:
                extra.append("--use-oracle-categories")
            if args.diagnostics:
                extra.append("--diagnostics")

            rc = run_adapter(
                output_path=args.output,
                dataset_path=subset_path,
                answerer=args.answerer,
                extra_flags=extra,
            )
            if rc != 0:
                print(f"ERROR: adapter exited with code {rc}", file=sys.stderr)
                return rc
        finally:
            try:
                subset_path.unlink()
            except FileNotFoundError:
                pass

        # Judge the new output. The judge writes results next to the
        # input file as <input>.eval-results-<model>.
        rc = run_judge(
            judge_script=args.judge_script,
            judge_model=args.judge_model,
            hyp_jsonl=args.output,
            oracle_path=args.oracle,
        )
        if rc != 0:
            print(f"ERROR: judge exited with code {rc}", file=sys.stderr)
            return rc

    if not judge_out.exists():
        print(f"ERROR: expected judge output not found: {judge_out}", file=sys.stderr)
        return 1

    new_labels = parse_judge_output(judge_out)
    print_delta_report(
        wrongs=wrongs,
        rights=rights,
        sampled_rights=sampled,
        new_labels=new_labels,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
