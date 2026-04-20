"""Retrieval evaluation harness for jarvis-memory search.

Spec: brain/projects/jarvis-memory/plans/runs/2026-04-20-eval-harness-and-routing/spec.md

Implements standard IR metrics (Precision@k, Recall@k, MRR, nDCG@k) and a
CLI that runs them against jarvis-memory's current search path
(`scored_search`) on a committed synthetic corpus.

Pure-Python metrics — no DB dependency. The CLI is the only surface that
touches Neo4j/Chroma; when invoked with ``--ingest-corpus-first`` it writes
corpus episodes into an isolated namespace (label ``:TestEpisode`` + a
dedicated Chroma collection) so evaluation never pollutes production data.

Callable forms
--------------
- ``python -m jarvis_memory.eval --help``
- ``from jarvis_memory.eval import run_eval, precision_at_k, ...``
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ── Metric primitives ─────────────────────────────────────────────────


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    """Drop duplicates while preserving first-seen order.

    IR metrics treat the ranked list as an ordered set of unique ids;
    callers sometimes pass lists with repeats (duplicate search hits),
    which skews @k numerators if counted naively.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved ids that are relevant.

    Args:
        retrieved: Ranked list of document ids (best first).
        relevant: Set of ground-truth relevant ids.
        k: Cutoff.

    Returns:
        Value in [0, 1]. Returns 0.0 for ``k <= 0`` or empty ``retrieved``.
    """
    if k <= 0 or not retrieved:
        return 0.0
    top = _dedup_preserve_order(retrieved)[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant set captured in the top-k retrieved ids.

    Args:
        retrieved: Ranked list of document ids (best first).
        relevant: Set of ground-truth relevant ids.
        k: Cutoff.

    Returns:
        Value in [0, 1]. Returns 0.0 for empty ``relevant``.
    """
    if not relevant:
        return 0.0
    if k <= 0 or not retrieved:
        return 0.0
    top = set(_dedup_preserve_order(retrieved)[:k])
    hits = len(top & relevant)
    return hits / len(relevant)


def mrr(
    retrieved_lists: list[list[str]],
    relevant_lists: list[set[str]],
) -> float:
    """Mean Reciprocal Rank across a batch of queries.

    Reciprocal rank for a single query is 1/rank of the first relevant
    hit, or 0 if no relevant doc is retrieved.

    Args:
        retrieved_lists: Per-query ranked id lists.
        relevant_lists: Per-query relevant id sets, aligned by position.

    Returns:
        Mean of per-query reciprocal ranks. Zero if the inputs are empty
        or lengths differ.
    """
    if not retrieved_lists or len(retrieved_lists) != len(relevant_lists):
        return 0.0

    total = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        if not retrieved or not relevant:
            continue
        for rank, doc_id in enumerate(_dedup_preserve_order(retrieved), start=1):
            if doc_id in relevant:
                total += 1.0 / rank
                break
    return total / len(retrieved_lists)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance).

    Uses the standard log2-based DCG with binary gains (relevant=1, else=0).
    IDCG is the best possible ordering: all relevant docs at the top.

    Args:
        retrieved: Ranked list of document ids (best first).
        relevant: Set of ground-truth relevant ids.
        k: Cutoff.

    Returns:
        Value in [0, 1]. Returns 0.0 for empty relevant set or k <= 0.
    """
    if k <= 0 or not retrieved or not relevant:
        return 0.0

    top = _dedup_preserve_order(retrieved)[:k]

    dcg = 0.0
    for i, doc_id in enumerate(top, start=1):
        if doc_id in relevant:
            # gain=1, discount=log2(i+1). Rank-1 → denom=log2(2)=1.
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: place min(k, |relevant|) relevant docs at top positions.
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ── I/O helpers ───────────────────────────────────────────────────────


def parse_qrels(path: str | Path) -> dict[str, set[str]]:
    """Parse a JSONL qrels file into a ``query_id -> set(doc_ids)`` map.

    The JSONL format is flexible: each line must be a JSON object with a
    ``query_id`` key. Relevant doc ids may live under any of:
    ``relevant_ids``, ``relevant``, or ``doc_ids`` (first present wins).

    Args:
        path: Path to a .jsonl file.

    Returns:
        Mapping from query id to set of relevant doc ids.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If any line is not a JSON object with ``query_id``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"qrels file not found: {p}")

    qrels: dict[str, set[str]] = {}
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno} invalid JSON: {e}") from e
            if not isinstance(row, dict) or "query_id" not in row:
                raise ValueError(
                    f"{p}:{lineno} missing 'query_id' (got {type(row).__name__})"
                )
            qid = str(row["query_id"])
            relevant = (
                row.get("relevant_ids")
                or row.get("relevant")
                or row.get("doc_ids")
                or []
            )
            if not isinstance(relevant, (list, set, tuple)):
                raise ValueError(
                    f"{p}:{lineno} relevant ids must be a list (got {type(relevant).__name__})"
                )
            qrels[qid] = {str(x) for x in relevant}
    return qrels


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts. Skips blank lines."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl file not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno} invalid JSON: {e}") from e
    return rows


# ── Core harness ──────────────────────────────────────────────────────


SearchFn = Callable[[str, int], list[str]]
"""Search callable signature: (query_text, k_max) -> ranked list of doc ids."""


def run_eval(
    search_fn: SearchFn,
    corpus_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Run the retrieval eval end-to-end.

    Args:
        search_fn: Callable ``(query, k_max) -> [doc_id, ...]`` that returns
            a ranked list of doc ids from a store that has already been
            loaded with the corpus.
        corpus_path: Path to corpus JSONL (each row must include ``uuid``).
            Used only to measure corpus size in the report.
        queries_path: Path to queries JSONL (each row must include
            ``query_id`` and ``query``).
        qrels_path: Path to qrels JSONL (see :func:`parse_qrels`).
        k_values: The cutoffs to report for P@k / R@k / nDCG@k.

    Returns:
        Dict with keys:

        - ``precision_at_k``: ``{k: mean precision}``
        - ``recall_at_k``: ``{k: mean recall}``
        - ``mrr``: scalar MRR over all queries
        - ``ndcg_at_k``: ``{k: mean nDCG}``
        - ``n_queries``: int
        - ``n_corpus``: int
        - ``k_values``: list[int]
    """
    corpus_rows = _load_jsonl(corpus_path)
    query_rows = _load_jsonl(queries_path)
    qrels = parse_qrels(qrels_path)

    for row in query_rows:
        if "query_id" not in row or "query" not in row:
            raise ValueError(
                f"query row missing 'query_id' or 'query': {row}"
            )

    ks = sorted({int(k) for k in k_values if int(k) > 0})
    if not ks:
        raise ValueError("k_values must contain at least one positive int")
    k_max = max(ks)

    per_query_retrieved: list[list[str]] = []
    per_query_relevant: list[set[str]] = []

    for row in query_rows:
        qid = str(row["query_id"])
        query_text = str(row["query"])
        relevant = qrels.get(qid, set())
        try:
            retrieved = search_fn(query_text, k_max)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("search_fn raised on query %s: %s", qid, e)
            retrieved = []
        retrieved = [str(x) for x in (retrieved or [])]
        per_query_retrieved.append(retrieved)
        per_query_relevant.append(relevant)

    n = max(len(per_query_retrieved), 1)
    precision_scores = {
        k: sum(
            precision_at_k(r, rel, k)
            for r, rel in zip(per_query_retrieved, per_query_relevant)
        )
        / n
        for k in ks
    }
    recall_scores = {
        k: sum(
            recall_at_k(r, rel, k)
            for r, rel in zip(per_query_retrieved, per_query_relevant)
        )
        / n
        for k in ks
    }
    ndcg_scores = {
        k: sum(
            ndcg_at_k(r, rel, k)
            for r, rel in zip(per_query_retrieved, per_query_relevant)
        )
        / n
        for k in ks
    }
    mrr_score = mrr(per_query_retrieved, per_query_relevant)

    return {
        "precision_at_k": {str(k): round(v, 4) for k, v in precision_scores.items()},
        "recall_at_k": {str(k): round(v, 4) for k, v in recall_scores.items()},
        "mrr": round(mrr_score, 4),
        "ndcg_at_k": {str(k): round(v, 4) for k, v in ndcg_scores.items()},
        "n_queries": len(query_rows),
        "n_corpus": len(corpus_rows),
        "k_values": ks,
    }


# ── Test-namespace corpus loader (CLI only) ───────────────────────────

TEST_NEO4J_LABEL = "TestEpisode"
TEST_CHROMA_COLLECTION = "jarvis_eval_v1"


def _load_corpus_into_test_namespace(corpus_path: str | Path) -> tuple[Any, Any, Any]:
    """Create an isolated Neo4j+Chroma test namespace and load the corpus.

    Writes to a dedicated Neo4j label (``:TestEpisode``) and a separate
    ChromaDB collection (``jarvis_eval_v1``). Returns the resources the
    caller must tear down: ``(driver, embed_store, test_collection)``.

    Each episode is written as a ``:TestEpisode`` node with fields mirroring
    a real ``:Episode`` (uuid, content, group_id, memory_type, created_at,
    room, hall, importance) so room/hall filters in ``scored_search`` work.
    """
    from neo4j import GraphDatabase
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    from .config import (
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        CHROMADB_PATH,
        EMBEDDING_MODEL,
    )
    from .classifier import classify_memory
    from .rooms import detect_room, get_hall

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Isolated ChromaDB collection — separate from prod jarvis_memories.
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    # Drop any prior collection so re-runs are clean.
    try:
        client.delete_collection(TEST_CHROMA_COLLECTION)
    except Exception:  # noqa: BLE001 — ok if it doesn't exist
        pass
    collection = client.create_collection(
        name=TEST_CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Wipe any prior TestEpisode nodes.
    with driver.session() as db:
        db.run(f"MATCH (n:{TEST_NEO4J_LABEL}) DETACH DELETE n")

    rows = _load_jsonl(corpus_path)
    logger.info("loading %d episodes into :%s + %s", len(rows), TEST_NEO4J_LABEL, TEST_CHROMA_COLLECTION)

    ids_batch: list[str] = []
    docs_batch: list[str] = []
    meta_batch: list[dict[str, Any]] = []

    for row in rows:
        uid = str(row.get("uuid") or row.get("id") or "")
        if not uid:
            continue
        content = str(row.get("content", ""))
        if not content:
            continue
        group_id = str(row.get("group_id", "system"))
        ep_type = row.get("episode_type") or classify_memory(content)
        importance = float(row.get("importance", 0.8))
        created_at = str(row.get("created_at", ""))
        room = detect_room(content, group_id)
        hall = get_hall(str(ep_type))

        with driver.session() as db:
            db.run(
                f"""
                CREATE (n:{TEST_NEO4J_LABEL} {{
                    uuid: $uid,
                    content: $content,
                    group_id: $gid,
                    memory_type: $mt,
                    episode_type: $mt,
                    room: $room,
                    hall: $hall,
                    importance: $imp,
                    created_at: datetime($created_at),
                    lifecycle_status: 'active',
                    access_count: 0
                }})
                """,
                uid=uid,
                content=content,
                gid=group_id,
                mt=str(ep_type),
                room=room,
                hall=hall,
                imp=importance,
                created_at=created_at or "2026-04-01T00:00:00+00:00",
            )

        ids_batch.append(uid)
        docs_batch.append(content)
        meta_batch.append(
            {
                "wing": group_id,
                "room": room,
                "hall": hall,
                "memory_type": str(ep_type),
                "created_at": created_at or "2026-04-01T00:00:00+00:00",
            }
        )

    if ids_batch:
        collection.upsert(ids=ids_batch, documents=docs_batch, metadatas=meta_batch)

    return driver, client, collection


def _teardown_test_namespace(driver: Any, client: Any) -> None:
    """Delete all TestEpisode nodes + the isolated Chroma collection."""
    try:
        with driver.session() as db:
            db.run(f"MATCH (n:{TEST_NEO4J_LABEL}) DETACH DELETE n")
    except Exception as e:  # noqa: BLE001
        logger.warning("neo4j teardown failed: %s", e)
    try:
        client.delete_collection(TEST_CHROMA_COLLECTION)
    except Exception as e:  # noqa: BLE001
        logger.warning("chroma teardown failed: %s", e)
    try:
        driver.close()
    except Exception:  # noqa: BLE001
        pass


def _make_scored_search_fn(collection: Any, driver: Any) -> SearchFn:
    """Build a search closure that exercises prod ``scored_search`` logic.

    Run 3 update: the eval now drives :func:`jarvis_memory.scoring.scored_search`,
    which in turn runs the hybrid RRF pipeline (Chroma + Neo4j full-text +
    expansion + Page boosts). The isolated :TestEpisode namespace is
    passed through so the search stays namespaced.

    ``JARVIS_SEARCH_LEGACY=1`` routes internally to the Run 1 composite
    path for baseline comparison.
    """
    from .scoring import scored_search as _scored_search

    def _vector_fn(q: str, n: int) -> list[dict[str, Any]]:
        try:
            cr = collection.query(query_texts=[q], n_results=n)
        except Exception as e:  # pragma: no cover
            logger.warning("chroma query failed: %s", e)
            return []
        ids = cr.get("ids", [[]])[0] if cr else []
        distances = cr.get("distances", [[]])[0] if cr else [0.0] * len(ids)
        metadatas = cr.get("metadatas", [[]])[0] if cr else [{}] * len(ids)
        out: list[dict[str, Any]] = []
        for idx, uid in enumerate(ids):
            dist = distances[idx] if idx < len(distances) else 0.0
            similarity = max(0.0, 1.0 - (dist / 2.0))
            meta = metadatas[idx] if idx < len(metadatas) else {}
            out.append(
                {
                    "id": uid,
                    "uuid": uid,
                    "similarity": similarity,
                    "metadata": dict(meta) if meta else {},
                }
            )
        return out

    def _search(query: str, k_max: int) -> list[str]:
        results = _scored_search(
            query,
            limit=k_max,
            driver=driver,
            namespace=TEST_NEO4J_LABEL,
            vector_search_fn=_vector_fn,
        )
        return [str(r.get("uuid") or r.get("id") or "") for r in results if (r.get("uuid") or r.get("id"))]

    return _search


# ── CLI ────────────────────────────────────────────────────────────────


def _parse_k_values(arg: str) -> list[int]:
    try:
        return [int(x.strip()) for x in arg.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid --k value: {arg}") from e


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis_memory.eval",
        description=(
            "Run retrieval eval (P@k / R@k / MRR / nDCG@k) against the "
            "current scored_search on a JSONL corpus+qrels."
        ),
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus JSONL (each row: uuid, content, group_id, ...).",
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="Path to queries JSONL (each row: query_id, query).",
    )
    parser.add_argument(
        "--qrels",
        required=True,
        help=(
            "Path to qrels JSONL "
            "(each row: query_id, relevant_ids=[uuid,...])."
        ),
    )
    parser.add_argument(
        "--k",
        default="1,3,5,10",
        type=_parse_k_values,
        help="Comma-separated k values (default: 1,3,5,10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report to stdout.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON report to.",
    )
    parser.add_argument(
        "--ingest-corpus-first",
        action="store_true",
        help=(
            "Load the corpus into an isolated test namespace "
            f"(:{TEST_NEO4J_LABEL} + Chroma '{TEST_CHROMA_COLLECTION}') "
            "before running the eval. Tears down on exit."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Python log level (default WARNING).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    driver = None
    chroma_client = None
    search_fn: SearchFn

    if args.ingest_corpus_first:
        driver, chroma_client, collection = _load_corpus_into_test_namespace(args.corpus)
        search_fn = _make_scored_search_fn(collection, driver)
    else:
        # Assume corpus already loaded. Fall back to the production
        # EmbeddingStore — caller is responsible for isolating the
        # namespace if they don't want prod contamination.
        from .embeddings import EmbeddingStore

        store = EmbeddingStore()
        if not store.health_check():
            print(
                "ERROR: ChromaDB not available and --ingest-corpus-first was not set.",
                file=sys.stderr,
            )
            return 2

        def search_fn(query: str, k_max: int) -> list[str]:  # type: ignore[no-redef]
            hits = store.search(query=query, limit=k_max)
            return [str(h.get("id", "")) for h in hits if h.get("id")]

    try:
        report = run_eval(
            search_fn=search_fn,
            corpus_path=args.corpus,
            queries_path=args.queries,
            qrels_path=args.qrels,
            k_values=args.k,
        )
    finally:
        if args.ingest_corpus_first and driver is not None and chroma_client is not None:
            _teardown_test_namespace(driver, chroma_client)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json or args.out:
        if args.out:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        if args.json:
            print(payload)
    else:
        # Human-friendly summary.
        print(f"Corpus: {report['n_corpus']} episodes, {report['n_queries']} queries, k={report['k_values']}")
        print("Precision@k:", report["precision_at_k"])
        print("Recall@k:   ", report["recall_at_k"])
        print("nDCG@k:     ", report["ndcg_at_k"])
        print("MRR:        ", report["mrr"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
