#!/usr/bin/env python3
"""Phase 8 gate report — apply Codex's gate to a targeted-validation run.

Reads the v1.1 baseline + a targeted re-run's judged output, computes the
fix/regression breakdown and BOTH projection methods (naive total +
rate projection), and emits a PASS/FAIL verdict per Codex's gate:

    PASS = naive_projection >= 96.0%
        AND regressed <= 1
        AND no obvious regression cluster (>3 newly-wrong in same category)

Usage:
    python scripts/longmemeval/phase8_gate_report.py \\
        --baseline-eval runs/lme_gpt41_v1.1.jsonl.eval-results-gpt-4o \\
        --targeted-eval runs/lme_phase8_104q.jsonl.eval-results-gpt-4o
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

GATE_NAIVE_PCT = 96.0
GATE_MAX_REGRESSIONS = 1
GATE_CLUSTER_THRESHOLD = 3  # >3 newly-wrong in same category = cluster fail


def load_eval(path: Path) -> dict[str, dict]:
    """{question_id: row}."""
    out = {}
    for line in open(path):
        row = json.loads(line)
        out[row["question_id"]] = row
    return out


def label(row: dict) -> bool:
    return bool(row.get("autoeval_label", {}).get("label", False))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-eval", type=Path, required=True,
                   help="Baseline (v1.1) judged JSONL — 500 questions.")
    p.add_argument("--targeted-eval", type=Path, required=True,
                   help="Phase 8 targeted run judged JSONL — ~104 questions.")
    args = p.parse_args()

    baseline = load_eval(args.baseline_eval)
    targeted = load_eval(args.targeted_eval)

    if len(baseline) != 500:
        print(f"WARN: baseline has {len(baseline)} rows, expected 500", file=sys.stderr)

    baseline_right = sum(1 for r in baseline.values() if label(r))
    baseline_wrong = len(baseline) - baseline_right
    baseline_pct = 100 * baseline_right / len(baseline)

    fixed, still_wrong, regressed, stable, missing = [], [], [], [], []
    fixed_by_cat: Counter[str] = Counter()
    still_by_cat: Counter[str] = Counter()
    reg_by_cat: Counter[str] = Counter()

    for qid, brow in baseline.items():
        was_right = label(brow)
        cat = brow.get("predicted_category", "?")
        if qid not in targeted:
            missing.append(qid)
            continue
        is_right = label(targeted[qid])
        if was_right and not is_right:
            regressed.append(qid)
            reg_by_cat[cat] += 1
        elif not was_right and is_right:
            fixed.append(qid)
            fixed_by_cat[cat] += 1
        elif not was_right and not is_right:
            still_wrong.append(qid)
            still_by_cat[cat] += 1
        else:
            stable.append(qid)

    n_targeted = len(targeted)
    n_fixed = len(fixed)
    n_regressed = len(regressed)
    n_sampled_rights = sum(1 for qid in targeted
                           if qid in baseline and label(baseline[qid]))

    # Naive projection: assume untargeted-unsampled questions stay at baseline
    # outcome. Then 500-accuracy = baseline_right + fixed - regressed.
    naive_right = baseline_right + n_fixed - n_regressed
    naive_pct = 100 * naive_right / 500

    # Rate projection: extrapolate regression rate from sampled-rights to all
    # baseline rights. Generous to fixes (count them at face value), pessimistic
    # to regressions (assume same rate applies to the unsampled rights).
    if n_sampled_rights > 0:
        reg_rate = n_regressed / n_sampled_rights
        projected_right = baseline_right * (1 - reg_rate) + n_fixed
    else:
        reg_rate = 0.0
        projected_right = baseline_right + n_fixed
    rate_pct = 100 * projected_right / 500

    # Cluster check
    cluster_fail = False
    cluster_cats = []
    newly_wrong_by_cat: Counter[str] = Counter()
    for qid in regressed:
        cat = baseline[qid].get("predicted_category", "?")
        newly_wrong_by_cat[cat] += 1
    for cat, n in newly_wrong_by_cat.items():
        if n > GATE_CLUSTER_THRESHOLD:
            cluster_fail = True
            cluster_cats.append((cat, n))

    # Gate
    naive_pass = naive_pct >= GATE_NAIVE_PCT
    reg_pass = n_regressed <= GATE_MAX_REGRESSIONS
    overall_pass = naive_pass and reg_pass and not cluster_fail

    print()
    print("=" * 72)
    print("PHASE 8 GATE REPORT")
    print("=" * 72)
    print(f"Baseline (v1.1):           {baseline_right}/{len(baseline)} = {baseline_pct:.2f}%")
    print(f"Targeted re-run:           {n_targeted} questions judged")
    print(f"  Baseline-wrongs in run:  {n_targeted - n_sampled_rights}")
    print(f"  Sampled-rights in run:   {n_sampled_rights}")
    if missing:
        print(f"  Missing from targeted:   {len(missing)}")
    print()
    print(f"FIXED         {n_fixed:>4}   (baseline-wrong → right)")
    print(f"STILL WRONG   {len(still_wrong):>4}")
    print(f"REGRESSED     {n_regressed:>4}   (baseline-right → wrong)")
    print(f"STABLE        {len(stable):>4}")
    print()
    print("Fixes by baseline predicted_category:")
    for cat in sorted(set(list(fixed_by_cat) + list(still_by_cat))):
        f, s = fixed_by_cat[cat], still_by_cat[cat]
        rate = (100 * f / (f + s)) if (f + s) else 0.0
        print(f"  {cat:<28}  fixed={f:>3}  still={s:>3}  fix_rate={rate:>5.1f}%")
    if regressed:
        print()
        print("Regressions by category:")
        for cat in sorted(reg_by_cat):
            print(f"  {cat:<28}  reg={reg_by_cat[cat]}")
        print("Regressed qids:")
        for qid in regressed:
            cat = baseline[qid].get("predicted_category", "?")
            print(f"  - {qid} ({cat})")
    print()
    print("PROJECTIONS (both — Codex gate requires reporting both):")
    print(f"  Naive total      = {baseline_right} + {n_fixed} - {n_regressed} = {naive_right}/500 = {naive_pct:.2f}%")
    print(f"  Rate projection  = {baseline_right} × (1 - {reg_rate:.4f}) + {n_fixed} = {projected_right:.1f}/500 = {rate_pct:.2f}%")
    print()
    print("CODEX GATE:")
    print(f"  ≥ {GATE_NAIVE_PCT}% naive    : {'PASS' if naive_pass else 'FAIL'}  (naive={naive_pct:.2f}%)")
    print(f"  ≤ {GATE_MAX_REGRESSIONS} regression : {'PASS' if reg_pass else 'FAIL'}  (regressed={n_regressed})")
    if cluster_fail:
        cats = ", ".join(f"{c}={n}" for c, n in cluster_cats)
        print(f"  No cluster        : FAIL  (>{GATE_CLUSTER_THRESHOLD} newly-wrong in: {cats})")
    else:
        print(f"  No cluster        : PASS")
    print()
    print(f"OVERALL: {'PASS — proceed to Phase 10 (500q headline)' if overall_pass else 'FAIL — abandon Phase 8 branch per Codex'}")
    print("=" * 72)

    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())
