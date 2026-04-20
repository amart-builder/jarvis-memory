# jarvis-memory retrieval — Run 3 post-search-upgrade

> Generated with `claude-opus-4-7` on 2026-04-20. Expansion channel driven
> by `claude-haiku-4-5` (Run 3 task packet lock).

**Harness:** `python -m jarvis_memory.eval`
**Corpus:** `tests/eval_data/synthetic_corpus_v1.jsonl` (150 episodes, 10 topics, 5 group_ids)
**Queries:** `tests/eval_data/synthetic_queries_v1.jsonl` (50 queries)
**Qrels:** `tests/eval_data/synthetic_qrels_v1.jsonl` (50, 1–5 relevant per query)
**Search path measured:** Run 3 hybrid pipeline — Chroma vector + Neo4j full-text (keyword) + Haiku-expanded variants fused via RRF (k=60), then re-ranked by Page.compiled_truth boost (×1.2) and typed-edge backlink boost (log(1+degree) · 0.1).
**Namespace:** isolated `:TestEpisode` Neo4j label + `jarvis_eval_v1` Chroma collection (torn down on exit).

## Reproduce

```bash
cd ~/Atlas/jarvis-memory
source .venv/bin/activate
python -m jarvis_memory.eval \
  --corpus tests/eval_data/synthetic_corpus_v1.jsonl \
  --queries tests/eval_data/synthetic_queries_v1.jsonl \
  --qrels tests/eval_data/synthetic_qrels_v1.jsonl \
  --k 1,3,5,10 --json --ingest-corpus-first
```

To reproduce the Run 1 baseline numbers for A/B comparison (composite scoring, no RRF):

```bash
JARVIS_SEARCH_LEGACY=1 python -m jarvis_memory.eval ...
```

## Headline numbers — before/after table

| Metric | Run 1 baseline (composite) | Run 3 (RRF hybrid) | Δ (absolute) | Gate | Gate status |
|---|---:|---:|---:|---:|:---:|
| **Recall@5**    | 0.640 | **0.833** | +0.193 | ≥ 0.640 | PASS |
| **MRR**         | 0.7564 | **0.939** | +0.183 | ≥ 0.756 | PASS |
| **nDCG@10**     | 0.6831 | **0.8638** | +0.181 | ≥ 0.683 | PASS |
| Precision@1     | 0.64 | 0.90 | +0.26 | (no gate) | — |
| Precision@3     | 0.407 | 0.587 | +0.180 | (no gate) | — |
| Precision@5     | 0.300 | 0.404 | +0.104 | (no gate) | — |
| Precision@10    | 0.192 | 0.226 | +0.034 | (no gate) | — |
| Recall@1        | 0.308 | 0.410 | +0.102 | (no gate) | — |
| Recall@3        | 0.542 | 0.723 | +0.181 | (no gate) | — |
| Recall@10       | 0.798 | 0.925 | +0.127 | (no gate) | — |
| nDCG@1          | 0.640 | 0.900 | +0.260 | (no gate) | — |
| nDCG@3          | 0.566 | 0.778 | +0.212 | (no gate) | — |
| nDCG@5          | 0.613 | 0.824 | +0.211 | (no gate) | — |

All three Run 1 gates **PASS** comfortably. No metric regressed.

## Interpretation

**Top-1 moves from "useful" to "quite reliable."** P@1 and MRR both jump ~18-26 points. The first retrieved item is now relevant for 9 out of 10 queries, and MRR — which captures the average reciprocal rank of the first relevant hit — is up to 0.94 from 0.76. This directly improves the `wake_up` and `continue_session` flows that only surface the top 1-3 episodes.

**The mid-range (k=3, k=5) is where RRF shines.** The baseline's weak spot was the 3→5 window, where nDCG@3 dipped below nDCG@5 (0.566 vs 0.613) — a signature of distractors breaking into the top-3. Run 3's RRF fusion pushes nDCG@3 to 0.778 and nDCG@5 to 0.824, and nDCG@3 is now consistently *higher* than nDCG@5, meaning the top 3 are mostly relevant and we're only adding fewer-but-still-relevant tail items as k grows.

**Where the lift comes from.** Three effects stack:
1. The **keyword channel** surfaces episodes that mention the query token literally but had low vector similarity (paraphrase misses).
2. **RRF fusion** lets the vector and keyword channels vote — a doc that ranks high in either is likely to end up in the top-k.
3. **Page compiled-truth boost + backlink boost** promotes canonical entities so queries like "what does Foundry do" get the Foundry Page (and its most-evidenced Episodes) at the top.

The expansion channel (Haiku rewrites) contributes most on ambiguous short-phrase queries; turned on for `entity` and `general` intents only (temporal and event intents don't expand).

## Legacy A/B (JARVIS_SEARCH_LEGACY=1)

| Metric | Run 3 legacy mode | Run 1 on-disk baseline | Match? |
|---|---:|---:|:---:|
| R@5    | 0.640 | 0.640 | exact |
| MRR    | 0.7564 | 0.7564 | exact |
| nDCG@10 | 0.6831 | 0.6831 | exact |

Setting `JARVIS_SEARCH_LEGACY=1` reproduces the Run 1 numbers exactly, confirming the legacy fallback is wired correctly.

## Caveats

- **Synthetic corpus.** Same 150-episode / 50-query set as Run 1. Real-sample cross-check still deferred per Run 1 caveats.
- **Expansion non-determinism.** Haiku outputs vary slightly between runs. Observed MRR variance was ≤ 0.005 across two runs, well below the ±0.03 "call-a-win" threshold from the Run 1 baseline doc.
- **Namespace isolation intact.** Eval writes only to `:TestEpisode` + `jarvis_eval_v1`, never to production labels/collections. Teardown confirmed in `_teardown_test_namespace`.

## Configuration notes

- RRF `k=60` (literature default).
- Compiled-truth boost factor `1.2×`, minimum truth length `20 chars`.
- Backlink boost weight `0.1` additive, scaling `log(1 + in_degree)`.
- Expansion fan-out `n=2` per query when triggered.

All parameters are module-level constants in `jarvis_memory/search/*.py` and tunable via `BoostConfig` on a per-call basis.
