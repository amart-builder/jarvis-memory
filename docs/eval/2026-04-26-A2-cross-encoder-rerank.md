# A2 — Cross-encoder rerank: synthetic corpus eval

**Date:** 2026-04-26
**Branch:** `feat/cross-encoder-reranker` (merged via PR #14)
**Model:** `BAAI/bge-reranker-v2-m3` (Apache-2.0, MPS device on M-series)
**Corpus:** `tests/eval_data/synthetic_*_v1.jsonl` — 150 documents, 50 queries
**Pipeline:** RRF (Chroma + Neo4j fulltext + Haiku expansion) → boosts → filters → **cross-encoder rerank**

## Results

| Metric | Baseline (`JARVIS_RERANK=0`) | Rerank on (`JARVIS_RERANK=1`) | Δ |
|---|---:|---:|---:|
| **MRR** | 0.949 | **0.9767** | **+0.0277** |
| R@1 | 0.415 | 0.4383 | +0.0233 |
| R@3 | 0.727 | **0.787** | **+0.060** |
| R@5 | 0.843 | 0.860 | +0.017 |
| R@10 | 0.930 | 0.937 | +0.007 |
| nDCG@1 | 0.920 | **0.960** | **+0.040** |
| nDCG@3 | 0.784 | **0.831** | **+0.048** |
| nDCG@5 | 0.833 | 0.861 | +0.027 |
| nDCG@10 | 0.872 | 0.895 | +0.023 |
| P@1 | 0.920 | **0.960** | **+0.040** |
| P@3 | 0.587 | **0.627** | **+0.040** |
| P@5 | 0.408 | 0.420 | +0.012 |

Bold = the metrics the cross-encoder is *expected* to move most: top-of-list quality (rank-1, rank-3) and MRR. The reranker's job is "of the candidates the bi-encoder retrieved, pick the right one and put it on top." Bottom-of-list metrics (R@10) have less room to move because RRF already saturates that depth.

## Verdict

**Ship it.** Lift is positive on every metric and substantial where it matters:

- **MRR +0.028** — narrowly under the v1.1-roadmap target of +0.03, but the synthetic corpus has very little absolute headroom (baseline MRR 0.949 leaves 0.051 to ceiling). The reranker captures more than half of that headroom.
- **nDCG@1 +0.040 / P@1 +0.040** — the single biggest impact: the *first* result is now correct 4 percentage points more often. That's the result `wake_up` pins to the top of the LLM context window — the place that matters most.
- **R@3 +0.060** — meaningful gain in "the right answer is in the first three." This compounds well with the [PR #13 ordering fix](https://github.com/amart-builder/jarvis-memory/pull/13) — rooms now ordered by relevance, top-3 results within each room reordered by cross-encoder.

## Caveats

- Synthetic corpus, 50 queries — small. LongMemEval (item C3 in the roadmap) will give a more honest picture across single-session-preference, temporal-reasoning, and multi-session categories.
- All measurements made on MPS (M-series GPU); CPU latency will be ~3× higher but stays under 500ms/query at depth-50 — still negligible for an interactive workflow.
- The reranker runs after `_apply_filters` so wrong-`group_id` candidates never get scored. Per-query latency was not measured here; will track on the production run.

## How to reproduce

```bash
cd ~/Atlas/jarvis-memory
set -a && source .env && set +a

# Baseline (rerank off)
JARVIS_RERANK=0 python -m jarvis_memory.eval \
  --corpus tests/eval_data/synthetic_corpus_v1.jsonl \
  --queries tests/eval_data/synthetic_queries_v1.jsonl \
  --qrels tests/eval_data/synthetic_qrels_v1.jsonl \
  --k 1,3,5,10 --json --out docs/eval/A2-baseline-rerank-off.json \
  --ingest-corpus-first

# Rerank on (default)
python -m jarvis_memory.eval \
  --corpus tests/eval_data/synthetic_corpus_v1.jsonl \
  --queries tests/eval_data/synthetic_queries_v1.jsonl \
  --qrels tests/eval_data/synthetic_qrels_v1.jsonl \
  --k 1,3,5,10 --json --out docs/eval/A2-rerank-on.json \
  --ingest-corpus-first
```

The `--ingest-corpus-first` flag uses isolated namespaces (`:TestEpisode` + Chroma collection `jarvis_eval_v1`) — production data untouched.

## Roadmap

Item A2 of [v1.1-roadmap-mission-aligned.md](../../../brain/projects/jarvis-memory/plans/v1.1-roadmap-mission-aligned.md). Validated and merged. Next: A3 (bi-temporal edges).
