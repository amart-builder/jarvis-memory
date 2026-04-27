#!/usr/bin/env python3
"""Drop-in parallel replacement for the official LongMemEval judge.

Mirrors the CLI and output format of ``/tmp/lme-official/src/evaluation/evaluate_qa.py``
(``python run_parallel_judge.py <model> <hyp_jsonl> <ref_json>``) so it
plugs into the existing targeted-validation harness without changes.

Why this exists: the official judge processes questions sequentially —
500 questions × ~3-5 sec/judge call = 25-40 min. Each call is a stateless
OpenAI chat completion; per-question state is independent. Trivial to
parallelize with ``ThreadPoolExecutor``. With 8 workers, judge time
drops to ~5 min.

Safety analysis (per Alex's "only parallelize where provably safe" rule):
  * No shared model singletons (unlike the adapter's cross-encoder).
  * No shared DB / collection state.
  * The OpenAI ``client`` instance is documented as thread-safe.
  * Each worker writes to a per-task dict; the main thread re-emits in
    input order. No write contention.
  * Failure of one judge call doesn't cascade — exception is captured
    per-row and the row gets ``autoeval_label.label = False`` plus an
    ``autoeval_error`` field for diagnostics.

Determinism: same as the sequential judge. ``temperature=0``, no batch
ordering effects on per-question grading. With OpenAI's "best effort"
seed semantics, individual labels could in theory differ by 1-2% from
the sequential judge — same noise floor as gpt-4.1 generation.

Usage:
    python scripts/run_parallel_judge.py gpt-4o \\
        runs/lme_gpt41_v1.1.jsonl \\
        data/longmemeval/longmemeval_oracle.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import backoff
import openai
from openai import OpenAI


# Mirror the official judge's model_zoo so behavior is identical.
_MODEL_ZOO: dict[str, tuple[str, str]] = {
    "llama-3.1-70b-instruct": ("meta-llama/Meta-Llama-3.1-70B-Instruct", "local"),
    "gpt-4o-mini": ("gpt-4o-mini-2024-07-18", "openai"),
    "gpt-4o": ("gpt-4o-2024-08-06", "openai"),
}

# Per-category prompt templates, copied verbatim from the official
# evaluate_qa.py (lines 24-43). We don't import the official module to
# avoid a runtime dependency on a /tmp checkout.
_PROMPT_TEMPLATES: dict[str, str] = {
    # All non-abstention SS-* and multi-session questions share one
    # template (official lines 27-29).
    "single-session-user": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response is equivalent to the correct answer or contains "
        "all the intermediate steps to get the correct answer, you should also answer "
        "yes. If the response only contains a subset of the information required by "
        "the answer, answer no. \n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel "
        "Response: {r}\n\nIs the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response is equivalent to the correct answer or contains "
        "all the intermediate steps to get the correct answer, you should also answer "
        "yes. If the response only contains a subset of the information required by "
        "the answer, answer no. In addition, do not penalize off-by-one errors for "
        "the number of days. If the question asks for the number of "
        "days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
        "predicting 19 days when the answer is 18), the model's response is still "
        "correct. \n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response contains some previous information along with an "
        "updated answer, the response should be considered as correct as long as the "
        "updated answer is the required answer.\n\nQuestion: {q}\n\nCorrect Answer: "
        "{a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or "
        "no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and "
        "a response from a model. Please answer yes if the response satisfies the "
        "desired response. Otherwise, answer no. The model does not need to reflect "
        "all the points in the rubric. The response is correct as long as it recalls "
        "and utilizes the user's personal information correctly.\n\nQuestion: {q}"
        "\n\nRubric: {a}\n\nModel Response: {r}\n\nIs the model response correct? "
        "Answer yes or no only."
    ),
}
# multi-session and single-session-assistant share the SS-user template.
_PROMPT_TEMPLATES["single-session-assistant"] = _PROMPT_TEMPLATES["single-session-user"]
_PROMPT_TEMPLATES["multi-session"] = _PROMPT_TEMPLATES["single-session-user"]

# Abstention questions get a different prompt asking whether the model
# correctly REFUSED to answer (official lines 41-42).
_ABSTENTION_PROMPT: str = (
    "I will give you an unanswerable question, an explanation, and a response from "
    "a model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not.\n\nQuestion: {q}"
    "\n\nExplanation: {a}\n\nModel Response: {r}\n\nDoes the model correctly "
    "identify the question as unanswerable? Answer yes or no only."
)


def build_anscheck_prompt(qtype: str, question: str, answer: str,
                          response: str, *, abstention: bool) -> str:
    """Same prompt selection as the official judge — exposed as a pure
    function so unit tests can pin the wording.
    """
    if abstention:
        return _ABSTENTION_PROMPT.format(q=question, a=answer, r=response)
    template = _PROMPT_TEMPLATES.get(qtype)
    if template is None:
        raise NotImplementedError(f"unknown question_type: {qtype!r}")
    return template.format(q=question, a=answer, r=response)


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIError),
    max_tries=8,
)
def _judge_one(client: OpenAI, model: str, prompt: str) -> str:
    """One judge call. Backoff matches the official judge's retry policy."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        temperature=0,
        max_tokens=10,
    )
    return (resp.choices[0].message.content or "").strip()


def grade_entry(client: OpenAI, model: str, entry: dict[str, Any],
                qid_to_qtype: dict[str, str], qid_to_qdata: dict[str, dict],
                ) -> dict[str, Any]:
    """Grade one hypothesis. Returns the entry mutated with autoeval_label.

    On error (network, malformed row, etc.) the entry gets
    ``autoeval_label.label=False`` and ``autoeval_error: <str>`` so the
    pipeline doesn't fail — same fail-open posture as the adapter.
    """
    qid = entry.get("question_id")
    if qid is None or qid not in qid_to_qtype:
        entry["autoeval_label"] = {"model": model, "label": False}
        entry["autoeval_error"] = "missing_or_unknown_qid"
        return entry

    qtype = qid_to_qtype[qid]
    qdata = qid_to_qdata[qid]

    try:
        prompt = build_anscheck_prompt(
            qtype=qtype,
            question=qdata["question"],
            answer=qdata["answer"],
            response=entry.get("hypothesis", ""),
            abstention="_abs" in qid,
        )
        eval_response = _judge_one(client, model, prompt)
        label = "yes" in eval_response.lower()
        entry["autoeval_label"] = {"model": model, "label": label}
    except Exception as e:  # noqa: BLE001 — fail-open per row
        entry["autoeval_label"] = {"model": model, "label": False}
        entry["autoeval_error"] = f"{type(e).__name__}: {e}"
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel drop-in for the official LongMemEval judge.",
    )
    parser.add_argument("metric_model", choices=sorted(_MODEL_ZOO.keys()),
                        help="Judge model id (matches official judge model_zoo).")
    parser.add_argument("hyp_file", type=Path, help="Hypothesis JSONL.")
    parser.add_argument("ref_file", type=Path, help="Oracle JSON with ground truth.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent OpenAI calls (default 8). Lower if hitting rate limits.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-question prints (the official judge prints every grade).")
    args = parser.parse_args()

    if not args.hyp_file.exists():
        print(f"ERROR: hyp file not found: {args.hyp_file}", file=sys.stderr)
        return 2
    if not args.ref_file.exists():
        print(f"ERROR: ref file not found: {args.ref_file}", file=sys.stderr)
        return 2

    metric_model, metric_source = _MODEL_ZOO[args.metric_model]
    if metric_source == "openai":
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORGANIZATION"),
        )
    else:
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

    # Load hypotheses (JSONL or JSON array — official supports both).
    try:
        hypotheses = [json.loads(line) for line in args.hyp_file.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError:
        hypotheses = json.loads(args.hyp_file.read_text())

    # Load references (JSON array or JSONL — official supports both).
    try:
        references = json.loads(args.ref_file.read_text())
    except json.JSONDecodeError:
        references = [json.loads(line) for line in args.ref_file.read_text().splitlines() if line.strip()]

    qid_to_qtype: dict[str, str] = {r["question_id"]: r["question_type"] for r in references}
    qid_to_qdata: dict[str, dict] = {r["question_id"]: r for r in references}
    qtypes = sorted(set(qid_to_qtype.values()))

    n_total = len(hypotheses)
    print(f"Judging {n_total} hypotheses with {args.workers} workers ({metric_model})...")

    # Dispatch concurrently; preserve original input order in output.
    results: list[dict | None] = [None] * n_total

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {
            pool.submit(grade_entry, client, metric_model, dict(entry),
                        qid_to_qtype, qid_to_qdata): i
            for i, entry in enumerate(hypotheses)
        }
        n_done = 0
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            row = fut.result()
            results[i] = row
            n_done += 1
            if not args.quiet:
                qid = row.get("question_id", "?")
                label = row.get("autoeval_label", {}).get("label", "?")
                print(f"  [{n_done}/{n_total}] {qid}  → {label}", flush=True)

    # Write output JSONL in input order, matching the official judge.
    result_path = Path(str(args.hyp_file) + f".eval-results-{args.metric_model}")
    with result_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print per-category accuracy (matches official judge's final block).
    qtype_acc: dict[str, list[int]] = {t: [] for t in qtypes}
    for r in results:
        if r is None:
            continue
        qid = r.get("question_id")
        if qid not in qid_to_qtype:
            continue
        label = r.get("autoeval_label", {}).get("label", False)
        qtype_acc[qid_to_qtype[qid]].append(1 if label else 0)

    correct = sum(
        1 for r in results
        if r is not None and r.get("autoeval_label", {}).get("label", False)
    )
    overall_acc = correct / max(n_total, 1)
    print(f"\nAccuracy: {round(overall_acc, 4)}")
    for k in qtypes:
        v = qtype_acc[k]
        if v:
            print(f"\t{k}: {round(sum(v)/len(v), 4)} ({len(v)})")
    print(f"Saved to {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
