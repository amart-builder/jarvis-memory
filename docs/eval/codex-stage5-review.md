# Stage 5 Plan Review

## TL;DR

Revise before shipping. The bucket counts in the Stage 5 spec are real, but the main conclusion is wrong: Bucket C is not mostly "gold outside the prompt." In this adapter, the retrieval diagnostics are computed after final filtering and after the currently disabled context trim, so the 20 Bucket C questions already have all gold sessions in the prompt. Widening `max_res` from 20 to 30 may add distractors, but it does not directly fix the dominant failure mode. I would drop 5A as the headline patch, skip/rewrite 5B because raw + expanded fan-out already exists, add better pre/post-filter diagnostics, and spend the remaining time on evidence-packet / salience rescue for the all-gold-visible failures.

## 1. Plan correctness

The good news: Catalyst's A/B/C/D/E bucket counts reproduce exactly from `runs/lme_gpt41_stage4d_targeted.jsonl.eval-results-gpt-4o`: A=5, B=6, C=20, D=1, E=0. I checked the JSONL directly.

The bad news: Bucket C is being interpreted incorrectly. The spec says Bucket C means "reranker/trim leaves gold below cap" and treats widening `n_hits_used` as the direct fix (`docs/eval/stage5-spec.md:61-66`, `docs/eval/stage5-spec.md:83-103`). But `compute_retrieval_diagnostics()` explicitly computes ranks over the final `hits` list that enters the prompt (`scripts/run_longmemeval.py:412-466`, `scripts/run_longmemeval.py:1062-1073`). `n_hits_used` is also just `len(hits)` after filtering/trim (`scripts/run_longmemeval.py:1055-1061`, `scripts/run_longmemeval.py:1165-1167`).

The current context trim is a no-op for every category because `CONTEXT_BUDGET_CHARS = {}` (`scripts/longmemeval/classifier.py:327-338`), and `trim_to_context_budget()` returns the input unchanged when the category has no budget (`scripts/run_longmemeval.py:361-365`). The effective cap is `FILTER_CONFIG[*].max_res`, applied inside retrieval (`scripts/run_longmemeval.py:834-846`; `scripts/longmemeval/classifier.py:283-292`). So when a row says `n_hits_used=20` and a gold rank is 17, that gold session is already in the prompt.

Concrete examples:

- `88432d0a` is wrong with `n_hits_used=20`; all 4 gold sessions are in the prompt at ranks 6, 11, 16, and 17 (`runs/lme_gpt41_stage4d_targeted.jsonl.eval-results-gpt-4o:16`). The model still counted 3 instead of 4. That is not a context-cap miss.
- `gpt4_7abb270c` is wrong with all 6 museum sessions in the prompt at ranks 2, 5, 7, 9, 14, and 20 (`runs/lme_gpt41_stage4d_targeted.jsonl.eval-results-gpt-4o:63`). The spec says the model "literally cannot see" ranks 14 and 20 (`docs/eval/stage5-spec.md:51-55`), but the diagnostics show the opposite.
- `2788b940` is wrong with all 4 gold sessions in the prompt, including one at rank 1 and the rest at ranks 16, 18, and 19 (`runs/lme_gpt41_stage4d_targeted.jsonl.eval-results-gpt-4o:22`). Again, widening to 30 is not the direct lever.

This changes the plan materially. 5A might still help a few cases if non-oracle supporting context is missing, but the projected 8-12 fixes is not supported by the diagnostic. The safer estimate is 0-3, with real regression risk.

5B also needs correction. The spec says the adapter "currently retrieves only once with the expanded query" (`docs/eval/stage5-spec.md:45-47`) and proposes raw + expanded query fan-out. The current code already does this: primary retrieval uses the expanded query (`scripts/run_longmemeval.py:747-756`), secondary retrieval uses the raw query (`scripts/run_longmemeval.py:759-773`). OMEGA's local reference also uses precedence append/merge, not RRF, for raw + expanded results (`/tmp/omega-memory/scripts/longmemeval_official.py:982-1007`, `/tmp/omega-memory/scripts/longmemeval_official.py:1018-1028`). So revised 5B is at best a merge-strategy experiment, not a missing OMEGA port.

5G is directionally plausible, but under-instrumented. Current retrieval has several caps: `k` passed into `scored_search` (`scripts/run_longmemeval.py:695-699`, `scripts/run_longmemeval.py:747-763`), internal vector/keyword overfetch in `scored_search` (`jarvis_memory/scoring.py:329-360`), adapter pure-channel rerank depth (`scripts/run_longmemeval.py:789-801`), and final `max_res` (`scripts/run_longmemeval.py:834-846`). The current diagnostics only tell us what survived into the prompt. They do not tell us whether Bucket B misses are beyond initial `k`, dropped by weighted rerank, dropped by `min_rel`, or absent from all channels.

Parallelism is useful if safe, but it is not a score intervention. Do not let it consume the Stage 5 critical path until the score plan is fixed.

## 2. Risks I identified

The widening-to-30 risk is worse than Catalyst frames it. The model already fails on 20-note prompts where the gold is present. Adding 10 more full sessions makes the actual problem harder: more near-duplicates, more temporal distractors, and more chances for the two-pass MS extractor to omit a valid item. Because final prompt order is chronological (`scripts/run_longmemeval.py:864-866`), the original retrieval rank signal is lost before generation. A rank-17 gold session is visible, but not highlighted.

The proposed 30-question parallelism parity check is not enough. Comparing only `hypothesis` can miss retrieval-order drift that happens not to flip the final answer on that sample. The check should compare prompt hashes, ordered hit UUIDs, diagnostics, and judge labels. The sample should be stratified across Bucket A/B/C/D plus high-context stable-rights, not purely random.

Per-worker Chroma collection suffixes are not full isolation. All workers would still write through the same persistent Chroma path (`scripts/run_longmemeval.py:1249-1259`). Multi-process writes to the same Chroma store are the scary part, not just collection-name collision. If adapter parallelism ships, use a per-worker Chroma path or a proven read/write lock, not just collection suffixes.

Neo4j label suffixes are safer than Chroma suffixes, but still add cleanup and query-surface risk. Every namespace call must take the suffix, including keyword and PPR paths (`scripts/run_longmemeval.py:750`, `scripts/run_longmemeval.py:762`, `scripts/run_longmemeval.py:798`). Crash cleanup needs a separate janitor or stale labels will accumulate.

RRF is not obviously the right way to merge raw + expanded query results. OMEGA uses "primary first, then fill from secondary/tertiary" precedence merge (`/tmp/omega-memory/scripts/longmemeval_official.py:1001-1007`). RRF can elevate generic raw-query distractors above expanded-query hits, which is exactly the failure pattern in questions like `8550ddae` where the raw phrase is vague.

The tolerance of "2 flips out of 30" for parallelism is too loose if prompts and retrieval are identical. With `temperature=0` and `seed=42`, any differences should be investigated. If the team wants to tolerate OpenAI nondeterminism, compare retrieval and prompt hashes separately from final text.

## 3. Methodology issues

The 104-question targeted validation is fine as a cheap directional gate, but it is not strong enough to justify a narrow 80% confidence interval or an 85% probability of clearing 95.4%. The regression side is only 30 of 426 baseline-right questions (`scripts/run_targeted_validation.py:251-288`). That is useful for catching obvious breakage, not for ruling out category-specific regressions.

The biggest methodology problem is definition laundering around "pure generation failure." If "pure generation failure" means "all gold sessions are in top 5," then yes, E=0 by definition. But operationally, 21/32 still-wrong questions have all gold sessions in the final prompt: 20 Bucket C plus 1 Bucket D. Those are at least partly generation, prompt-salience, or evidence-assembly failures. Calling them retrieval failures is only fair if the proposed fix improves ranking/salience, not if it merely appends more context.

The gold-rank diagnostic is also insufficient for the decisions Stage 5 wants to make. It only records the final prompt list. Before another scoring patch, add diagnostics for: raw `scored_search` expanded-query ranks, raw-query ranks, merged pre-rerank ranks, weighted-rerank ranks, post-filter ranks, final prompt ranks, and prompt hashes. Without that, 5G cannot distinguish "not retrieved" from "retrieved then filtered out."

There is real overfitting risk. Stage 4E-4H has already iterated prompt rules against known wrongs, and Stage 5 is now tuning around the same 32 still-wrongs. The mitigation is not another prompt rule. Freeze prompt changes unless a new diagnostic proves a narrow failure, and include a targeted stable-right sample that resembles the modified cases: high-context MS/TR/KU rights, not just random rights.

My honest estimate: Catalyst's projected +16 net is very squishy. With the plan as written, I would expect 5A + 5G + 5B to land more like +3 to +7 net, and it could be lower if the wider contexts break high-context rights. The current plan does not deserve an 85% clear-95.4 claim.

## 4. Alternatives worth considering

Highest ROI: replace 5A with an evidence-packet / salience-rescue patch for MS/TR/KU. Keep the 20 existing chronological notes, but prepend a compact "High-signal evidence" block built from the retrieved sessions before chronological sorting. It should preserve note numbers and include short user-turn snippets with dates, quantities, named entities, and temporal phrases. This directly attacks Bucket C: gold is present but ignored or underweighted.

This is the cheap adapter-level version of what Mastra and AgentMemory are doing. Mastra's public writeup says Observational Memory replaces raw message history with dense observations, keeping a stable observation log in context rather than dynamically dumping raw messages (https://mastra.ai/research/observational-memory). AgentMemory's README describes event extraction for temporal reasoning, broad recall up to `limit=500`, and context building with session-balanced/topic-dense selection, date labels, and coreference hints (https://github.com/JordanMcCann/agentmemory). We do not need to port those systems wholesale; we need one adapter-local evidence builder that gives the generator a better intermediate representation.

Second-highest ROI: temporal two-lane context. For any inferred temporal window, split prompt context into "in-window evidence" first and "other retrieved notes" second. Do not hard-drop out-of-window notes. This keeps Catalyst's concern about hard-filter misses, but makes questions like `88432d0a` and `gpt4_7abb270c` easier for the model.

Third: add rerank-rescue diagnostics and only then tune caps. For Bucket B, run a diagnostic with `k=75` and `max_res=75` without changing generation, record where missing gold appears, and then decide whether to widen initial `k`, final `max_res`, temporal boost, or keyword weighting. Blind widening is not engineering; it is hoping.

Fourth: if you still want a raw + expanded merge experiment, match OMEGA first: precedence merge with expanded/temporal primary, raw fallback. Do not jump to RRF unless a small ablation proves it helps the five A cases without hurting stable rights.

I would not spend the next 4 hours on a production-quality graph-memory rewrite. But I also would not pretend Stage 5A is a mechanical win. The local, cheap path is better context construction over the already-retrieved sessions.

## 5. Engineering critique

The parallelism design is under-scoped. `--tenant-suffix` has to reach the Neo4j label constant, Chroma collection name, setup, ingest delete, create, every retrieval namespace call, pure keyword search, PPR, cleanup, diagnostics, and tests. That is not a 60-line adapter change unless it is done sloppily.

The proposed tenant isolation test is necessary but not sufficient. Counting `:LMETestEpisode_w0` and `:LMETestEpisode_w1` after a 4-question run checks label routing, but not Chroma isolation, prompt equality, retrieval order equality, crash cleanup, resume behavior, or same-qid collision behavior.

A safer parallel design is: per-worker process, per-worker Chroma path, per-worker Neo4j label, separate output JSONL, no cleanup until the coordinator has verified outputs, then a janitor cleanup. If that is too much, defer adapter parallelism and use the already-built parallel judge only.

1.5 hours is optimistic. I would budget 3-5 hours to build and verify adapter parallelism safely. In the current race to publish, parallelism is only worth doing after the score intervention is credible. Otherwise it speeds up the wrong experiment.

The empirical parity test should be upgraded to:

1. Run serial and parallel on the same stratified 40-question set.
2. Compare ordered hit UUIDs, diagnostics, prompt hashes, hypotheses, and judge labels.
3. Include duplicate same-qid stress where two workers intentionally process the same qid into different tenant suffixes.
4. Verify no leftover `LMETestEpisode*` labels and no worker Chroma directories/collections after cleanup.

## 6. If I were starting fresh

With 3-4 hours and ~$200 left, I would do this:

1. Judge Stage 4E-4H immediately with the existing parallel judge. If it is net-negative, rollback or bisect before stacking anything.
2. Add retrieval-stage diagnostics. This is a small, non-scoring patch and prevents more motivated reasoning.
3. Implement MS/TR/KU evidence packets instead of widening raw context. Keep `max_res=20` initially. Prepend top-ranked snippets and temporal/entity/quantity cues while preserving the current chronological notes.
4. Add temporal two-lane context for inferred windows: in-window evidence first, fallback notes second.
5. Run the 104-question targeted validation. If net lift is strong and regressions are low, then run the full 500.
6. Only if validation wall time becomes the blocker, implement minimal parallelism with per-worker Chroma paths. I would not start with parallelism.

This plan is less flashy than "widen to 30," but it matches the actual failure evidence. The model is not starved for more raw notes in Bucket C. It is failing to use relevant notes that are already present.

## Recommended ordering of changes

1. Judge `runs/lme_gpt41_stage4eh_targeted.jsonl` and decide whether Stage 4E-4H stays. Do not stack on an unjudged prompt patch.
2. Add diagnostics for retrieval stages and prompt hashes; rerun targeted only if needed to fill missing diagnostics.
3. Replace 5A with evidence-packet / salience-rescue for MS/TR/KU, starting with Bucket C qids like `88432d0a`, `gpt4_7abb270c`, and `2788b940`.
4. Add temporal two-lane context for inferred temporal windows.
5. Re-evaluate 5G using the new diagnostics. Widen `k` or `max_res` only where the missing gold is proven to be just beyond a specific cap.
6. Drop revised 5B unless an ablation proves RRF beats the existing raw + expanded precedence merge. The current adapter already does dual fan-out.
7. Defer adapter parallelism unless the score patch validates. If built, isolate by Chroma path as well as Neo4j label.
