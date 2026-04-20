"""Tests for jarvis_memory.eval — IR metrics + harness plumbing.

Pure-Python. No Neo4j / Chroma / Anthropic calls.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from jarvis_memory.eval import (
    mrr,
    ndcg_at_k,
    parse_qrels,
    precision_at_k,
    recall_at_k,
    run_eval,
)


class TestPrecisionAtK:
    """Precision@k: fraction of top-k retrieved that are relevant."""

    def test_all_relevant(self):
        # All 3 in top-3 are relevant → P@3 = 1.0
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0

    def test_partial_hits(self):
        # 2 of top-4 relevant → 0.5
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_k_greater_than_retrieved(self):
        # 1 relevant out of 2 retrieved; P@5 divides by k=5.
        assert precision_at_k(["a", "x"], {"a"}, 5) == pytest.approx(1 / 5)

    def test_duplicate_retrieved_ids_dedup(self):
        # Duplicates collapse before @k cutoff.
        # Unique = ['a','x'], top-2 has one hit → 0.5.
        assert precision_at_k(["a", "a", "a", "x"], {"a"}, 2) == 0.5

    def test_k_one_hit(self):
        assert precision_at_k(["a", "b", "c"], {"a"}, 1) == 1.0

    def test_k_one_miss(self):
        assert precision_at_k(["x", "a"], {"a"}, 1) == 0.0


class TestRecallAtK:
    """Recall@k: fraction of the relevant set captured in top-k."""

    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0

    def test_half_retrieved(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_empty_relevant_returns_zero(self):
        # Undefined by definition; by convention we return 0 so callers
        # don't blow up on queries with no qrel entry.
        assert recall_at_k(["a", "b"], set(), 5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}, 5) == 0.0

    def test_k_too_small_to_capture_all(self):
        # 2 relevant exist but only k=1 → best we can do is 0.5.
        assert recall_at_k(["a", "b"], {"a", "b"}, 1) == 0.5

    def test_k_zero(self):
        assert recall_at_k(["a"], {"a"}, 0) == 0.0

    def test_duplicate_retrieved_dedup(self):
        # Dupes don't inflate recall.
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == 0.5


class TestMRR:
    """Mean Reciprocal Rank."""

    def test_single_query_first_position(self):
        # Hit at rank 1 → 1.0
        assert mrr([["a", "b", "c"]], [{"a"}]) == 1.0

    def test_single_query_third_position(self):
        # Hit at rank 3 → 1/3
        assert mrr([["x", "y", "a"]], [{"a"}]) == pytest.approx(1 / 3)

    def test_no_hits(self):
        assert mrr([["x", "y"]], [{"a"}]) == 0.0

    def test_multi_query_average(self):
        # q1 hit at 1, q2 hit at 2 → mean of 1.0 and 0.5 = 0.75
        rl = [["a", "x"], ["x", "b"]]
        rel = [{"a"}, {"b"}]
        assert mrr(rl, rel) == pytest.approx(0.75)

    def test_empty_inputs(self):
        assert mrr([], []) == 0.0

    def test_mismatched_lengths(self):
        assert mrr([["a"]], []) == 0.0

    def test_first_hit_wins(self):
        # Only the FIRST relevant doc contributes; later duplicates ignored.
        # Hit at rank 2 → 0.5
        assert mrr([["x", "a", "a"]], [{"a"}]) == 0.5


class TestNDCGAtK:
    """nDCG@k with binary relevance."""

    def test_perfect_ranking(self):
        # All k retrieved are relevant, in rank-1..k.
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_no_hits(self):
        assert ndcg_at_k(["x", "y", "z"], {"a"}, 3) == 0.0

    def test_single_hit_at_rank_one(self):
        # DCG = 1/log2(2) = 1; IDCG = 1 → nDCG = 1.0
        assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) == 1.0

    def test_single_hit_at_rank_two(self):
        # DCG = 1/log2(3); IDCG = 1 → nDCG = 1/log2(3).
        expected = 1.0 / math.log2(3)
        assert ndcg_at_k(["x", "a", "y"], {"a"}, 3) == pytest.approx(expected)

    def test_reverse_perfect_ranking(self):
        # 2 relevant but placed at ranks 2 and 3 instead of 1 and 2.
        # DCG = 1/log2(3) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3).
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 + 1 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3) == pytest.approx(expected)

    def test_empty_retrieved(self):
        assert ndcg_at_k([], {"a"}, 5) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), 5) == 0.0

    def test_k_one(self):
        assert ndcg_at_k(["a", "b", "c"], {"a"}, 1) == 1.0
        assert ndcg_at_k(["x", "a", "c"], {"a"}, 1) == 0.0


class TestParseQrels:
    """Qrels JSONL → dict loader."""

    def test_basic(self, tmp_path: Path):
        fpath = tmp_path / "qrels.jsonl"
        fpath.write_text(
            '\n'.join([
                json.dumps({"query_id": "q1", "relevant_ids": ["a", "b"]}),
                json.dumps({"query_id": "q2", "relevant_ids": ["c"]}),
            ]) + "\n",
            encoding="utf-8",
        )
        qrels = parse_qrels(fpath)
        assert qrels == {"q1": {"a", "b"}, "q2": {"c"}}

    def test_blank_lines_skipped(self, tmp_path: Path):
        fpath = tmp_path / "qrels.jsonl"
        fpath.write_text(
            '\n'.join([
                json.dumps({"query_id": "q1", "relevant_ids": ["a"]}),
                '',
                json.dumps({"query_id": "q2", "relevant_ids": ["b"]}),
            ]) + "\n",
            encoding="utf-8",
        )
        qrels = parse_qrels(fpath)
        assert len(qrels) == 2

    def test_accepts_alternate_key(self, tmp_path: Path):
        fpath = tmp_path / "qrels.jsonl"
        fpath.write_text(
            json.dumps({"query_id": "q1", "doc_ids": ["a", "b"]}) + "\n",
            encoding="utf-8",
        )
        qrels = parse_qrels(fpath)
        assert qrels == {"q1": {"a", "b"}}

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_qrels(tmp_path / "does-not-exist.jsonl")

    def test_missing_query_id_raises(self, tmp_path: Path):
        fpath = tmp_path / "qrels.jsonl"
        fpath.write_text('{"relevant_ids": ["a"]}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            parse_qrels(fpath)


class TestRunEval:
    """End-to-end harness wiring (with a stub search_fn, no DB)."""

    @pytest.fixture
    def corpus_files(self, tmp_path: Path) -> dict[str, Path]:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            '\n'.join([
                json.dumps({"uuid": "a", "content": "apple"}),
                json.dumps({"uuid": "b", "content": "banana"}),
                json.dumps({"uuid": "c", "content": "cherry"}),
            ]) + "\n",
            encoding="utf-8",
        )

        queries = tmp_path / "queries.jsonl"
        queries.write_text(
            '\n'.join([
                json.dumps({"query_id": "q1", "query": "fruit a"}),
                json.dumps({"query_id": "q2", "query": "fruit b"}),
            ]) + "\n",
            encoding="utf-8",
        )

        qrels = tmp_path / "qrels.jsonl"
        qrels.write_text(
            '\n'.join([
                json.dumps({"query_id": "q1", "relevant_ids": ["a"]}),
                json.dumps({"query_id": "q2", "relevant_ids": ["b"]}),
            ]) + "\n",
            encoding="utf-8",
        )

        return {"corpus": corpus, "queries": queries, "qrels": qrels}

    def test_perfect_search(self, corpus_files):
        """Stub that always returns the relevant doc first."""
        lookup = {"q1": "a", "q2": "b"}

        def search(query: str, k_max: int) -> list[str]:
            # Map query text to the right uuid heuristically.
            if query.endswith("a"):
                return ["a", "b", "c"][:k_max]
            if query.endswith("b"):
                return ["b", "a", "c"][:k_max]
            return []

        report = run_eval(
            search_fn=search,
            corpus_path=corpus_files["corpus"],
            queries_path=corpus_files["queries"],
            qrels_path=corpus_files["qrels"],
            k_values=[1, 3],
        )
        assert report["mrr"] == 1.0
        assert report["precision_at_k"]["1"] == 1.0
        assert report["recall_at_k"]["1"] == 1.0
        assert report["ndcg_at_k"]["1"] == 1.0
        assert report["n_queries"] == 2
        assert report["n_corpus"] == 3

    def test_keys_present(self, corpus_files):
        report = run_eval(
            search_fn=lambda q, k: [],
            corpus_path=corpus_files["corpus"],
            queries_path=corpus_files["queries"],
            qrels_path=corpus_files["qrels"],
            k_values=[1, 5, 10],
        )
        assert set(report.keys()) >= {
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "ndcg_at_k",
            "n_queries",
            "n_corpus",
            "k_values",
        }
        # All-empty search → all zeros, but keys exist.
        assert report["mrr"] == 0.0

    def test_search_fn_exception_does_not_crash(self, corpus_files):
        def bad_search(query: str, k_max: int) -> list[str]:
            raise RuntimeError("db down")

        report = run_eval(
            search_fn=bad_search,
            corpus_path=corpus_files["corpus"],
            queries_path=corpus_files["queries"],
            qrels_path=corpus_files["qrels"],
            k_values=[1],
        )
        assert report["mrr"] == 0.0


class TestCommittedCorpusShape:
    """The shipped synthetic corpus must stay well-formed."""

    REPO_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = REPO_ROOT / "tests" / "eval_data"

    def test_corpus_exists(self):
        assert (self.DATA_DIR / "synthetic_corpus_v1.jsonl").exists()
        assert (self.DATA_DIR / "synthetic_queries_v1.jsonl").exists()
        assert (self.DATA_DIR / "synthetic_qrels_v1.jsonl").exists()

    def test_line_counts(self):
        corpus = (self.DATA_DIR / "synthetic_corpus_v1.jsonl").read_text(encoding="utf-8").splitlines()
        queries = (self.DATA_DIR / "synthetic_queries_v1.jsonl").read_text(encoding="utf-8").splitlines()
        qrels = (self.DATA_DIR / "synthetic_qrels_v1.jsonl").read_text(encoding="utf-8").splitlines()
        assert 140 <= len(corpus) <= 160, f"corpus has {len(corpus)} rows, expected ~150"
        assert 40 <= len(queries) <= 60, f"queries has {len(queries)} rows, expected ~50"
        assert len(qrels) == len(queries), "qrels must be 1:1 with queries"

    def test_episode_types_valid(self):
        from jarvis_memory.classifier import MEMORY_TYPES

        corpus = self.DATA_DIR / "synthetic_corpus_v1.jsonl"
        with corpus.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                assert row.get("episode_type") in MEMORY_TYPES, (
                    f"line {lineno} has invalid episode_type: {row.get('episode_type')!r}"
                )

    def test_qrels_point_at_corpus_uuids(self):
        corpus_uuids: set[str] = set()
        with (self.DATA_DIR / "synthetic_corpus_v1.jsonl").open(encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    corpus_uuids.add(json.loads(raw)["uuid"])

        with (self.DATA_DIR / "synthetic_qrels_v1.jsonl").open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                for uid in row["relevant_ids"]:
                    assert uid in corpus_uuids, (
                        f"qrel line {lineno} references unknown uuid {uid}"
                    )
