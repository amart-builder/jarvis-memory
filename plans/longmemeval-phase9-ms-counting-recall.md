# LongMemEval Phase 9 - MS Counting Recall Filter

## Goal

Beat OMEGA's 95.4% honestly on the matched LongMemEval setup without bundling unrelated interventions.

## Current Evidence

- Phase 8 targeted gate: 48/74 baseline-wrongs fixed, 26 still wrong, 1/30 sampled-right regression.
- Naive projection: 473/500 = 94.6%.
- Needed to beat 95.4%: 4 additional net fixes.
- Needed to hit 96.0%: 7 additional net fixes.

## Error Atlas

The largest remaining mechanical cluster is multi-session counting partial retrieval:

- 8 still-wrong MS-counting questions have at least one gold session retrieved by an upstream stage but filtered out before the prompt.
- These are not zero-retrieval misses; they are relevance-threshold drops.
- The two-pass counting prompt is designed to enumerate candidates, so recall is more valuable than threshold pruning for this subset.

## Intervention

For `category == "multi-session"` and `counting == true`, keep the top `max_res` reranked hits directly instead of dropping candidates below `min_rel`.

Do not change:

- non-counting multi-session questions
- temporal-reasoning
- knowledge-update
- observation extraction
- prompt templates
- generator or judge setup

## Validation Gate

1. Unit-test the counting-only filter behavior.
2. Run 3-5 affected qids first and inspect judged results.
3. If the sample has net positive fixes with no obvious regression signal, run the same 104-question targeted gate.
4. Full 500 only after the 104q gate beats 95.4% naive projection with acceptable regressions.
