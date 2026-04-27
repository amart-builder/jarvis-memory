#!/usr/bin/env python3
"""LongMemEval adapter for jarvis-memory v1.1.

Per pre-registered protocol (docs/eval/longmemeval-v1.1-protocol.md):
- Test set: longmemeval_s_cleaned.json (NOT oracle).
- Single-shot per question, temperature=0.
- Question classifier (regex/heuristic) — not reading question_type.
- OMEGA's 5 prompt templates verbatim.
- OMEGA's triple fan-out retrieval recipe.
- AR1: PPR damping α=0.5 (HippoRAG-2 paper value).
- AR2: PPR seed broadening (noun phrases, not only proper nouns).
- AR3: counting enumeration (already in OMEGA's MULTISESSION prompt).
- Per-question isolation via group_id=lme_q_<id> + label :LMETestEpisode.

Usage:
    JARVIS_LME_ANSWERER=opus python scripts/run_longmemeval.py \\
        --output runs/lme_opus_v1.1.jsonl

    # Validate on 10 stratified questions:
    JARVIS_LME_ANSWERER=opus python scripts/run_longmemeval.py \\
        --output runs/lme_opus_validate.jsonl --validate

Resume:
    Re-running with the same --output skips already-answered question_ids.

Cost note: full 500 × 1 answerer ≈ $30-50 in API; ~6-10hr wall time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Make our scripts package importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.longmemeval.classifier import (  # noqa: E402
    ABSTENTION_FILTER,
    COUNTING_K_FLOOR,
    FILTER_CONFIG,
    K_FLOORS,
    classify,
    is_counting_question,
)
from scripts.longmemeval.prompts import (  # noqa: E402
    answer_to_str,
    format_session_for_prompt,
    format_session_text,
    parse_longmemeval_date,
    render_prompt,
)


# ── Constants ─────────────────────────────────────────────────────────


LME_NEO4J_LABEL: str = "LMETestEpisode"
LME_CHROMA_COLLECTION: str = "jarvis_lme_v1"
LME_AGENT_ID: str = "benchmark-longmemeval"

DEFAULT_DATASET: Path = Path("data/longmemeval/longmemeval_s_cleaned.json")
DEFAULT_OUTPUT: Path = Path("runs/lme_run.jsonl")

# Stoplist for AR2 (PPR seed broadening). Common English words that
# should NOT seed a graph walk — they appear too often.
_AR2_STOPLIST: set[str] = {
    "what", "when", "where", "which", "while", "with", "would", "have",
    "this", "that", "than", "then", "they", "them", "there", "these",
    "those", "from", "their", "your", "yours", "mine", "much", "many",
    "more", "most", "less", "some", "such", "since", "still", "thing",
    "things", "time", "times", "year", "years", "month", "months", "week",
    "weeks", "day", "days", "long", "ago", "after", "before", "between",
    "about", "ever", "very", "often", "into", "over", "under", "also",
    "been", "were", "been", "does", "did", "doing", "done", "had", "has",
    "having", "tell", "told", "said", "say", "saying", "asked", "answer",
    "good", "best", "first", "last", "next", "now", "currently", "current",
    "recent", "recently", "still", "yet", "anymore", "longer", "started",
    "begin", "began", "begun", "made", "make", "making", "took", "take",
    "taking", "taken", "give", "giving", "gave", "given", "going", "gone",
    "want", "wanting", "wanted", "needs", "need", "needing", "needed",
    "tried", "trying", "tries", "remember", "remembering", "remembered",
    "recall", "recalls", "recalled", "remind", "reminding", "reminded",
    "spend", "spent", "spending", "find", "finding", "found",
    "find", "lost", "lose", "losing", "shows", "show", "showing", "showed",
    "play", "plays", "played", "playing",
}


# ── AR1 + AR2: PPR overrides (monkey-patch) ───────────────────────────


def apply_ppr_overrides() -> None:
    """Apply pre-registered protocol additions AR1 + AR2 to PPR.

    AR1: damping α=0.85 → 0.5 (HippoRAG-2 paper value, spreads
    activation further across the graph for multi-hop).

    AR2: PPR seed broadening — extract noun phrases (lowercase common
    nouns ≥4 chars not in stoplist) in addition to the existing proper-
    noun extraction. So "how often do I exercise" seeds PPR on
    "exercise" instead of returning [].

    Applied via monkey-patch — no production code touched. The
    overrides revert when this process exits.
    """
    from jarvis_memory.search import ppr as ppr_mod

    _orig_extract = ppr_mod._extract_query_entities
    _orig_ppr = ppr_mod.personalized_pagerank

    def broadened_extract(query: str) -> list[str]:
        # Original proper-noun seeds first (preserves prior behavior).
        seeds = list(_orig_extract(query))
        seen = set(seeds)
        # Broaden with lowercase common nouns ≥4 chars.
        for word in re.findall(r"\b[a-z]{4,}\b", query.lower()):
            if word in seen or word in _AR2_STOPLIST:
                continue
            seeds.append(word)
            seen.add(word)
        return seeds

    def ppr_with_alpha(query, **kwargs):
        kwargs.setdefault("damping", 0.5)
        return _orig_ppr(query, **kwargs)

    ppr_mod._extract_query_entities = broadened_extract
    ppr_mod.personalized_pagerank = ppr_with_alpha


# ── Resume / output handling ──────────────────────────────────────────


def load_done_question_ids(output_path: Path) -> set[str]:
    """Read existing JSONL output to find already-answered question_ids."""
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                qid = row.get("question_id")
                if qid:
                    done.add(qid)
            except json.JSONDecodeError:
                # Tolerate a single bad trailing line, but no further.
                continue
    return done


# ── Ingestion ─────────────────────────────────────────────────────────


def ingest_question_haystack(
    *,
    driver: Any,
    chroma_collection: Any,
    question_data: dict,
    group_id: str,
) -> int:
    """Ingest the question's haystack sessions as :LMETestEpisode nodes.

    Wipes any prior LMETestEpisode rows for this group_id first
    (idempotent — supports re-runs after a mid-run crash). Each session
    becomes ONE node:
      - content = role-prefixed concat of turns (OMEGA recipe)
      - referenced_date = ISO from haystack_dates[i]
      - group_id = unique-per-question lme_q_<id>
      - agent_id = "benchmark-longmemeval"
      - namespace = :LMETestEpisode

    Returns the number of sessions ingested.
    """
    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    session_dates = question_data["haystack_dates"]

    # Wipe stale nodes from a prior run of this exact question.
    with driver.session() as db:
        db.run(
            f"MATCH (n:{LME_NEO4J_LABEL} {{group_id: $gid}}) DETACH DELETE n",
            gid=group_id,
        )

    # Drop any prior Chroma rows for this group.
    try:
        chroma_collection.delete(where={"group_id": group_id})
    except Exception:
        # Collection may be fresh — ignore.
        pass

    ids_batch: list[str] = []
    docs_batch: list[str] = []
    meta_batch: list[dict[str, Any]] = []
    n_ingested = 0

    for i, (turns, sid, raw_date) in enumerate(zip(sessions, session_ids, session_dates)):
        content = format_session_text(turns)
        if not content.strip():
            continue
        ref_date = parse_longmemeval_date(raw_date)
        # Use a unique per-question UUID so retrieval doesn't collide
        # with another question's same session_id.
        uid = f"{group_id}__{sid}"

        with driver.session() as db:
            db.run(
                f"""
                CREATE (n:{LME_NEO4J_LABEL} {{
                    uuid: $uid,
                    content: $content,
                    group_id: $gid,
                    memory_type: 'session_summary',
                    episode_type: 'session_summary',
                    referenced_date: $ref_date,
                    created_at: datetime($created_at),
                    t_created: datetime($created_at),
                    importance: 0.5,
                    lifecycle_status: 'active',
                    access_count: 0,
                    agent_id: $agent_id,
                    note_index: $idx
                }})
                """,
                uid=uid,
                content=content,
                gid=group_id,
                ref_date=ref_date,
                created_at=ref_date if "T" in ref_date else "2024-01-01T00:00:00",
                agent_id=LME_AGENT_ID,
                idx=i,
            )

        ids_batch.append(uid)
        docs_batch.append(content)
        meta_batch.append({
            "wing": group_id,        # eval.py uses `wing` for group_id
            "group_id": group_id,
            "memory_type": "session_summary",
            "referenced_date": ref_date,
            "created_at": ref_date if "T" in ref_date else "2024-01-01T00:00:00",
            "note_index": i,
        })
        n_ingested += 1

    if ids_batch:
        chroma_collection.upsert(ids=ids_batch, documents=docs_batch, metadatas=meta_batch)

    return n_ingested


# ── Retrieval (OMEGA's triple fan-out + classifier-driven K) ──────────


def retrieve_with_omega_recipe(
    *,
    query: str,
    group_id: str,
    category: str,
    counting: bool,
    driver: Any,
    embedding_store: Any,
    chroma_collection: Any,
) -> list[dict[str, Any]]:
    """OMEGA-style retrieval: triple fan-out + per-category K floor.

    Returns enriched-hit dicts sorted CHRONOLOGICALLY by referenced_date
    (oldest first), as required by the prompt rules ("higher note
    numbers are more recent").
    """
    from jarvis_memory.scoring import scored_search

    # K floor per OMEGA recipe: counting=45, multi/temporal=25, default=20.
    if counting:
        k = COUNTING_K_FLOOR
    else:
        k = K_FLOORS.get(category, 20)

    def _vector_search_fn(q: str, n: int) -> list[dict]:
        """Bind chroma to the per-question collection."""
        try:
            res = chroma_collection.query(
                query_texts=[q],
                n_results=min(n, 100),
                where={"group_id": group_id},
            )
        except Exception:
            return []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        out: list[dict] = []
        for uid, doc, dist, meta in zip(ids, docs, dists, metas):
            similarity = max(0.0, 1.0 - float(dist))  # cosine distance → similarity
            out.append({
                "uuid": uid,
                "id": uid,
                "content": doc,
                "similarity": similarity,
                "score": similarity,
                "group_id": (meta or {}).get("group_id", group_id),
                "memory_type": (meta or {}).get("memory_type", "session_summary"),
                "referenced_date": (meta or {}).get("referenced_date", ""),
                "created_at": (meta or {}).get("created_at", ""),
                "note_index": (meta or {}).get("note_index", 0),
            })
        return out

    # OMEGA does triple fan-out — primary + secondary unfiltered + tertiary
    # with raw query. We approximate with TWO calls: scored_search already
    # does query expansion + RRF over vector + keyword + (PPR if intent
    # multi_hop), so the "primary" is a strong fused ranking. We add a
    # second "raw, no expansion" pass to widen recall on edge cases where
    # expansion misses.
    primary = scored_search(
        query=query,
        group_id=group_id,
        namespace=LME_NEO4J_LABEL,
        limit=k,
        driver=driver,
        embedding_store=embedding_store,
        vector_search_fn=_vector_search_fn,
        include_expansion=True,
    )
    seen_ids = {h.get("uuid") or h.get("id") for h in primary}

    secondary = scored_search(
        query=query,
        group_id=group_id,
        namespace=LME_NEO4J_LABEL,
        limit=k,
        driver=driver,
        embedding_store=embedding_store,
        vector_search_fn=_vector_search_fn,
        include_expansion=False,  # raw query
    )
    for h in secondary:
        hid = h.get("uuid") or h.get("id")
        if hid and hid not in seen_ids:
            primary.append(h)
            seen_ids.add(hid)

    # Apply OMEGA's adaptive filter (per-category min_rel / min_res / max_res).
    cfg = FILTER_CONFIG.get(category, FILTER_CONFIG["single-session-user"])
    min_rel = float(cfg["min_rel"])
    min_res = int(cfg["min_res"])
    max_res = int(cfg["max_res"])

    def _score_of(h: dict) -> float:
        # Prefer composite_score (RRF), fall back to similarity.
        for k_ in ("composite_score", "score", "similarity"):
            v = h.get(k_)
            if v is not None:
                return float(v)
        return 0.0

    primary.sort(key=_score_of, reverse=True)

    # Keep top max_res; ensure at least min_res survive even if scores
    # are all below min_rel — better noisy context than empty.
    above = [h for h in primary if _score_of(h) >= min_rel]
    if len(above) >= min_res:
        kept = above[:max_res]
    else:
        kept = primary[:max(min_res, len(primary))][:max_res]

    # Recency boost for knowledge-update — OMEGA recipe (line 945).
    if category == "knowledge-update" and kept:
        # Sort by note_index ascending so we know the oldest/newest.
        with_idx = sorted(
            kept, key=lambda h: int(h.get("note_index") or 0)
        )
        n = len(with_idx)
        if n > 1:
            for i, h in enumerate(with_idx):
                # Linearly scale 1.0× (oldest) to 1.5× (newest).
                frac = i / (n - 1)
                h["_kept_score"] = _score_of(h) * (1.0 + 0.5 * frac)
            kept = sorted(with_idx, key=lambda h: h["_kept_score"], reverse=True)[:max_res]

    # Sort by referenced_date ascending (oldest → newest) for the prompt.
    def _date_key(h: dict) -> str:
        return str(h.get("referenced_date") or h.get("created_at") or "")

    kept.sort(key=_date_key)
    return kept


# ── Generation ────────────────────────────────────────────────────────


_OPUS_MODEL: str = "claude-opus-4-7"
_GPT4O_MODEL: str = "gpt-4o-2024-08-06"
_GPT41_MODEL: str = "gpt-4.1"


def call_llm(*, answerer: str, prompt: str, max_tokens: int) -> str:
    """Single-shot LLM generation. Temperature=0. No best-of-N.

    Supports three answerers:
      - opus  → Anthropic claude-opus-4-7 via ANTHROPIC_API_KEY
      - gpt4o → OpenAI gpt-4o-2024-08-06 via OPENAI_API_KEY
      - gpt41 → OpenAI gpt-4.1 via OPENAI_API_KEY

    Errors are logged and return an empty string — the runner records
    a failure row for that question and moves on.
    """
    if answerer == "opus":
        try:
            from anthropic import Anthropic
        except ImportError:
            raise RuntimeError("anthropic SDK not installed; pip install anthropic")
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=_OPUS_MODEL,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    if answerer in ("gpt4o", "gpt41"):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai SDK not installed; pip install openai")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model = _GPT4O_MODEL if answerer == "gpt4o" else _GPT41_MODEL
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    raise ValueError(f"Unknown answerer: {answerer!r} (use opus|gpt4o|gpt41)")


# ── Per-question pipeline ─────────────────────────────────────────────


def run_one_question(
    *,
    q: dict,
    answerer: str,
    driver: Any,
    embedding_store: Any,
    chroma_collection: Any,
) -> dict:
    """Execute the full pipeline for a single LongMemEval question.

    Returns the JSONL row to write. On crash, returns a row with
    ``hypothesis=""`` and ``error=<traceback>`` so the run continues.
    """
    qid = q["question_id"]
    question = q["question"]
    raw_qdate = q.get("question_date", "")
    qdate = parse_longmemeval_date(raw_qdate) if raw_qdate else ""

    group_id = f"lme_q_{qid}"
    classification = classify(question)
    category = classification.label
    counting = is_counting_question(question)

    cfg = FILTER_CONFIG.get(category, FILTER_CONFIG["single-session-user"])
    max_tokens = int(cfg["max_tokens"])

    t0 = time.time()
    try:
        # 1. Ingest haystack into isolated namespace.
        n_sessions = ingest_question_haystack(
            driver=driver,
            chroma_collection=chroma_collection,
            question_data=q,
            group_id=group_id,
        )

        # 2. Retrieve via OMEGA recipe.
        hits = retrieve_with_omega_recipe(
            query=question,
            group_id=group_id,
            category=category,
            counting=counting,
            driver=driver,
            embedding_store=embedding_store,
            chroma_collection=chroma_collection,
        )

        # 3. Confidence diagnostics ONLY — do not suppress max_tokens
        # or truncate hits based on our scoring scale. OMEGA's 0.20
        # threshold is calibrated to their similarity score; our RRF
        # composite + cosine similarity live on different scales, so
        # a hard cutoff would over-abstain. Instead we trust the
        # prompt rule "If the question cannot be answered ... say so"
        # — gpt-4o / Opus / gpt-4.1 all honor it reliably.
        if hits:
            top_score = max(
                float(h.get("similarity") or h.get("score") or 0.0)
                for h in hits
            )
        else:
            top_score = 0.0

        # 4. Format prompt.
        sessions_text = "\n\n".join(
            format_session_for_prompt(
                content=h.get("content", ""),
                date_str=str(h.get("referenced_date") or h.get("created_at") or ""),
                index=i + 1,  # 1-indexed for [Note N] readability
            )
            for i, h in enumerate(hits)
        )
        prompt = render_prompt(
            category=category,
            sessions=sessions_text,
            question=question,
            question_date=qdate,
        )

        # 5. Generate.
        hypothesis = call_llm(
            answerer=answerer,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        elapsed = time.time() - t0
        return {
            "question_id": qid,
            "hypothesis": hypothesis,
            "predicted_category": category,
            "classifier_rule": classification.rule,
            "counting": counting,
            "n_sessions_ingested": n_sessions,
            "n_hits_used": len(hits),
            "top_score": round(top_score, 4),
            "max_tokens": max_tokens,
            "answerer": answerer,
            "elapsed_sec": round(elapsed, 2),
        }
    except Exception:
        elapsed = time.time() - t0
        return {
            "question_id": qid,
            "hypothesis": "",
            "predicted_category": category,
            "answerer": answerer,
            "elapsed_sec": round(elapsed, 2),
            "error": traceback.format_exc(),
        }


# ── Setup ─────────────────────────────────────────────────────────────


def setup_resources():
    """Connect to Neo4j + ChromaDB + EmbeddingStore. Returns the trio."""
    from neo4j import GraphDatabase
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    from jarvis_memory.config import (
        CHROMADB_PATH,
        EMBEDDING_MODEL,
        NEO4J_PASSWORD,
        NEO4J_URI,
        NEO4J_USER,
    )
    from jarvis_memory.embeddings import EmbeddingStore

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Isolated Chroma collection — separate from prod jarvis_memories.
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try:
        collection = chroma_client.get_collection(LME_CHROMA_COLLECTION, embedding_function=ef)
    except Exception:
        collection = chroma_client.create_collection(
            name=LME_CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    embedding_store = EmbeddingStore()  # uses prod Chroma — required for scoring helpers
    return driver, embedding_store, collection


def stratified_subset(dataset: list[dict], n_per_cat: int = 2) -> list[dict]:
    """Pick a stratified validation subset across all 6 categories."""
    by_cat: dict[str, list[dict]] = {}
    for q in dataset:
        if q["question_id"].endswith("_abs"):
            # one abstention question per pass too
            by_cat.setdefault("_abs", []).append(q)
        else:
            by_cat.setdefault(q["question_type"], []).append(q)
    out: list[dict] = []
    for cat, qs in by_cat.items():
        out.extend(qs[:n_per_cat])
    return out


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LongMemEval adapter for jarvis-memory v1.1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL path (resume-safe).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to longmemeval_s_cleaned.json (or oracle for diagnostics).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many NEW questions (after resume).")
    parser.add_argument("--validate", action="store_true",
                        help="Run on a stratified 14-question subset (2 per category + 2 abs).")
    parser.add_argument("--question-id", type=str, default=None,
                        help="Run only this question_id (for debugging).")
    parser.add_argument("--answerer", type=str, default=None,
                        help="Override JARVIS_LME_ANSWERER env var. opus|gpt4o|gpt41.")
    args = parser.parse_args()

    answerer = args.answerer or os.environ.get("JARVIS_LME_ANSWERER", "")
    if answerer not in ("opus", "gpt4o", "gpt41"):
        print("ERROR: set JARVIS_LME_ANSWERER=opus|gpt4o|gpt41 (or pass --answerer)",
              file=sys.stderr)
        return 2

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    # Apply pre-registered AR1 + AR2 PPR overrides BEFORE any retrieval runs.
    apply_ppr_overrides()

    print(f"Loading dataset: {args.dataset}")
    with args.dataset.open() as f:
        dataset = json.load(f)
    print(f"  Loaded {len(dataset)} questions")

    if args.question_id:
        dataset = [q for q in dataset if q["question_id"] == args.question_id]
        if not dataset:
            print(f"ERROR: question_id {args.question_id} not in dataset", file=sys.stderr)
            return 2
    elif args.validate:
        dataset = stratified_subset(dataset, n_per_cat=2)
        print(f"  Validation subset: {len(dataset)} questions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_question_ids(args.output)
    if done_ids:
        print(f"Resume: {len(done_ids)} questions already answered in {args.output}")

    todo = [q for q in dataset if q["question_id"] not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"To process: {len(todo)} questions with answerer={answerer}")

    if not todo:
        print("Nothing to do — output already complete.")
        return 0

    print("Setting up Neo4j + Chroma + embedding store...")
    driver, embedding_store, chroma_collection = setup_resources()

    n_done = 0
    n_failed = 0
    start = time.time()

    try:
        with args.output.open("a") as out:
            for q in todo:
                row = run_one_question(
                    q=q,
                    answerer=answerer,
                    driver=driver,
                    embedding_store=embedding_store,
                    chroma_collection=chroma_collection,
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_done += 1
                if row.get("error"):
                    n_failed += 1
                    print(f"  [{n_done}/{len(todo)}] {row['question_id']}  ERROR ({row['elapsed_sec']:.1f}s)")
                else:
                    cat = row.get("predicted_category", "?")
                    hyp_preview = (row.get("hypothesis") or "")[:80].replace("\n", " ")
                    print(f"  [{n_done}/{len(todo)}] {row['question_id']:30s} "
                          f"cat={cat:25s} t={row['elapsed_sec']:5.1f}s  → {hyp_preview}")
    finally:
        driver.close()

    elapsed = time.time() - start
    print(f"\nDone. {n_done} processed, {n_failed} failed, {elapsed:.0f}s total.")
    print(f"Output: {args.output}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
