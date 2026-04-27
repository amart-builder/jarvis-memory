"""One-off diagnostic for LongMemEval question 0a995998.

Inspects what our retrieval pipeline returns vs the oracle's
answer_session_ids — answers "did we miss the relevant sessions, or
did the LLM over-prune?"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.longmemeval.classifier import classify, is_counting_question
from scripts.run_longmemeval import (
    LME_NEO4J_LABEL,
    _extract_session_id,
    apply_ppr_overrides,
    ingest_question_haystack,
    retrieve_with_omega_recipe,
    setup_resources,
)


def main():
    qid = "0a995998"
    # Load question from s_cleaned (the real test set, full haystack)
    with open("data/longmemeval/longmemeval_s_cleaned.json") as f:
        for q in json.load(f):
            if q["question_id"] == qid:
                question = q
                break

    # Load oracle for ground-truth answer_session_ids
    with open("data/longmemeval/longmemeval_oracle.json") as f:
        for o in json.load(f):
            if o["question_id"] == qid:
                oracle_q = o
                break

    answer_session_ids = set(oracle_q["answer_session_ids"])
    print(f"Question: {question['question']}")
    print(f"Truth: {oracle_q['answer']}")
    print(f"Total haystack sessions: {len(question['haystack_session_ids'])}")
    print(f"Ground-truth answer sessions ({len(answer_session_ids)}): {sorted(answer_session_ids)}")
    print()

    apply_ppr_overrides()
    driver, embedding_store, chroma_collection = setup_resources()

    group_id = f"lme_q_{qid}"
    classification = classify(question["question"])
    counting = is_counting_question(question["question"])

    print(f"Classified as: {classification.label} (rule: {classification.rule}, counting={counting})")
    print()

    print(f"Ingesting haystack into :{LME_NEO4J_LABEL}...")
    n = ingest_question_haystack(
        driver=driver,
        chroma_collection=chroma_collection,
        question_data=question,
        group_id=group_id,
    )
    print(f"  Ingested {n} sessions")
    print()

    print("Retrieving (OMEGA recipe)...")
    hits = retrieve_with_omega_recipe(
        query=question["question"],
        group_id=group_id,
        category=classification.label,
        counting=counting,
        driver=driver,
        embedding_store=embedding_store,
        chroma_collection=chroma_collection,
    )
    print(f"  Retrieved {len(hits)} hits after filter+sort")
    print()

    print(f"{'rank':>4} {'session_id':<35} {'truth?':<7} {'note_idx':<9} {'similarity':<11} content_preview")
    print("-" * 130)

    retrieved_session_ids = set()
    for rank, h in enumerate(hits, 1):
        uuid = str(h.get("uuid", ""))
        # Strip both group_id prefix AND the {idx:03d}_ index that ingestion
        # adds to handle duplicate session_ids in the haystack.
        sid = _extract_session_id(uuid, group_id)
        retrieved_session_ids.add(sid)
        is_truth = "✓" if sid in answer_session_ids else ""
        note_idx = h.get("note_index", "?")
        sim = float(h.get("similarity") or h.get("score") or 0.0)
        content_preview = (h.get("content") or "")[:80].replace("\n", " ")
        print(f"{rank:>4} {sid:<35} {is_truth:<7} {note_idx!s:<9} {sim:<11.4f} {content_preview}")

    print()
    print("=" * 60)
    found = retrieved_session_ids & answer_session_ids
    missed = answer_session_ids - retrieved_session_ids
    print(f"Ground-truth sessions retrieved: {len(found)}/{len(answer_session_ids)}")
    print(f"  Found: {sorted(found)}")
    print(f"  MISSED: {sorted(missed)}")
    print("=" * 60)

    driver.close()


if __name__ == "__main__":
    main()
