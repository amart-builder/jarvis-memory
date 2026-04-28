# Stage 5 Spec — Surgical Retrieval Patches + Parallelism

> ⚠️ **SUPERSEDED 2026-04-27 by `stage5-v2-spec.md`**
>
> This spec was based on a misread of `compute_retrieval_diagnostics()` —
> I claimed gold sessions ranked 11-20 were outside the prompt; they are
> actually inside the prompt (the diagnostic computes ranks AFTER all
> filtering). Codex caught this in `codex-stage5-review.md`. The headline
> patch (5A: widen context cap) would have added distractors without
> fixing the actual failure mode (salience, not retrieval).
>
> Kept here for audit trail. Do not implement from this file.

> Status: SUPERSEDED — see stage5-v2-spec.md
> Owner: Catalyst (Claude Code instance)
> Target: clear OMEGA's 95.4% on LongMemEval gpt-4.1
> Current projected score: 93.6% (Stage 4D) + Stage 4E-4H validation in flight

---

## 1. Goal

Push the projected score from 93.6% to ≥95.4%, ideally ≥96.0% for headline-margin
safety, by **attacking retrieval failures** identified in the Stage 4D deep-dive.

The deep-dive found that **0/32 still-wrong questions are pure generation
failures**. Every still-wrong has a retrieval defect:

| Bucket | Count | What broke | Fix this stage targets |
|---|---|---|---|
| **A** Zero retrieval (gold not in pool) | 5 | Channels miss the session entirely | 5B (revised), 5G |
| **B** Partial retrieval (some gold missing) | 6 | Multi-gold question, only got some | 5G |
| **C** All gold in pool but ranked >10 | 20 | Reranker/trim leaves gold below cap | **5A (the big one)** |
| **D** In top-10 not top-5 | 1 | Borderline | (no patch — accept) |
| **E** In top-5, generation fail | 0 | n/a | n/a |

## 2. Critical revisions to my original plan

### 2.1 Dropped: original 5B "proper-noun keyword boost"

**Why I dropped it:** I framed 5B as a NAVIGATIONAL-intent boost for proper-
noun-heavy queries. But the actual Bucket A failures don't have proper nouns
*in the query* — they have generic nouns ("the play", "the cocktail", "the
clothing brand") with the proper noun *in the gold session*.

A keyword multiplier on the query side can't help when the query is already
generic. **The original 5B would have been a no-op for 4 of 5 Bucket A questions.**

### 2.2 Revised: new 5B "raw + expanded query fan-out"

The actual OMEGA recipe for these vague-query failures is **dual-query
retrieval**: retrieve once with the raw query, retrieve again with the
LLM-expanded query, RRF-fuse the two ranked lists. This widens the
recall surface for queries that are too generic to match on their own.

We already have `expand_query()` from Stage 4D. We currently retrieve only
once with the expanded query. The new 5B plumbs both raw and expanded
through the same channels and fuses.

### 2.3 Dropped: 5H "TEMPORAL ordering reinforcement"

**Why I dropped it:** I claimed 5H targets `gpt4_7abb270c` (6-museum
ordering). But the deep-dive showed that question's gold ranks were 2, 5,
7, 9, 14, 20 — two of six gold sessions sit *below the top-10*, so the model
literally cannot see them at the current context cap of 20. The fix is **5A
(widen context cap)**, not prompt reinforcement.

The TEMPORAL prompt already has a `STEP 4 — For ordering questions, list
each event with its absolute date, then sort` rule. Reinforcing it again
would be a no-op without the missing context.

### 2.4 Confirmed: 5A is the headline patch

20 of 32 still-wrong questions have **all gold sessions in the pool but
ranked below 10**. Widening the prompt context cap from `n_hits_used: 20` to
`n_hits_used: 30` puts those gold sessions in the prompt where the model can
reason over them. Mechanical change, biggest direct fix.

### 2.5 Confirmed: 5G but framing matters

Pre-validation, I'm not sure whether `n_hits_used: 20` is:
- (a) the prompt context cap (post-rerank trim) — fixed by 5A
- (b) the candidate pool size (pre-rerank K) — fixed by 5G
- (c) the same single knob

**Step 1 of implementation will inspect the code to determine which.** If
(c), we ship one widening change covering both. If (a)/(b), we have two
distinct knobs to tune.

---

## 3. The 4-patch plan (revised)

### Patch A — 5A: context cap widening

**Change:** raise the per-category prompt context cap from
`MS=20, TR=20, KU=15, SS=10-12` to `MS=30, TR=30, KU=25, SS=15`.

**Targets:** all 20 Bucket C questions where `gold_in_pool` ≥ `gold_count`
but `gold_in_top10` < `gold_count`. Realistic fixes: 8-12.

**Code locations to inspect (step 1 of impl):**
- `scripts/run_longmemeval.py` — `n_hits_used` is the diagnostic name; trace
  back to the trim point.
- Per-category cap may live in the adapter or in `lme_weighted_rerank`
  helper — verify before editing.

**Verification:**
1. Unit test: render a fake retrieval result of 30 candidates, confirm
   prompt context contains all 30 for MS category.
2. Smoke test on `gpt4_7abb270c` (6-museum ordering): confirm the gold
   sessions at rank 14 and 20 now appear in the prompt.
3. Diff `n_hits_used` distribution between baseline and 5A run on 5
   questions — should clearly show wider distribution.

**Risk assessment:**
- **Token cost:** +50% context tokens on MS/TR questions (~+5K tokens per
  call). Cost on 500q full run: +$5-8. Acceptable.
- **Latency:** +0.5-1s per question. On 500q: +5 min. Acceptable.
- **Distraction risk:** more sessions → more noise → model could over-
  enumerate or pick wrong note. Mitigated by: Stage 4A two-pass MS counting
  already extracts → dedupes; Stage 4E inclusion rule drops borderline non-
  matches.
- **Token-limit risk:** gpt-4.1 has 128K context. 30 sessions × ~1500
  tokens/session = 45K tokens. Well under cap.

**Ship cost:** ~30 min (10 LOC + 1 test + smoke).

### Patch B — Parallelism (the time-saver)

**Change:** add a coordinator script `scripts/run_parallel_lme.py` that
splits the question list into N=4 chunks, spawns 4 worker processes each
running the existing adapter with its own `--tenant-suffix wN` flag, merges
JSONL outputs at the end. Worker-N writes to Neo4j label `:LMETestEpisode_wN`
and Chroma collection `lme_test_wN`, then cleans up its own labels post-run.

**Why parallelism here, not later:** every subsequent validation run benefits.
Once built, Stage 5 + full 500q each save ~75% wall time. Net savings across
the path to publishing: ~3 hours.

**Code locations to inspect (step 1 of impl):**
- `scripts/run_longmemeval.py` — find every `:LMETestEpisode` literal and
  every Chroma collection name reference. Add `tenant_suffix` parameter
  (default `""` for backward-compat).
- `jarvis_memory/api.py` — the adapter uses `agent_id` for production
  scoping; verify whether agent_id can carry the worker isolation, or if we
  need a separate tenant variable.
- `scripts/run_parallel_judge.py` — already has the coordinator pattern;
  copy structure.

**Verification (NON-NEGOTIABLE — blocking before any parallel headline run):**
1. **Empirical parity test.** Pick 30 random questions. Run serial (N=1).
   Run parallel (N=4) on the same 30. Diff `hypothesis` field. Required:
   ≥28/30 identical (allows 2 flips for OpenAI's known seed=42 noise floor).
   If <28/30, parallelism is causing systematic drift — STOP and debug.
2. **Tenant isolation test.** After a 2-worker run on 4 questions, query
   Neo4j: `MATCH (e:LMETestEpisode_w0) RETURN count(e)` and `... w1`. Confirm
   each worker created exactly its own subset. Confirm `:LMETestEpisode`
   (no suffix) returns 0.
3. **Cleanup test.** Confirm post-run that all `:LMETestEpisode_w*` labels
   are gone (each worker cleans up its own at exit).

**Risk assessment:**
- **OpenAI nondeterminism under concurrent load:** mitigated by parity check.
- **Mac Mini thermal:** 4 workers × 2 inferences = 8 concurrent MPS calls.
  Mac Mini M2 Pro handles this; if thermal throttles, scores still match
  (just slower).
- **Neo4j connection pool:** default 50 max; we use 4. Fine.
- **Chroma collection collision:** prevented by per-worker collection names.
- **Tailscale flicker:** still a risk per worker, but only 1/4 throughput
  affected vs all of it.
- **JSONL merge race:** workers write to *separate* output files; coordinator
  concatenates at end. No race.
- **MPS model load cost:** each worker loads ~2GB of models = 8GB total.
  Mac Mini has 16GB; tight but OK.

**Ship cost:** ~1.5 hrs (60 LOC adapter + 100 LOC coordinator + parity test
+ tenant test + smoke).

### Patch C — 5G: pre-rerank pool widening

**Change:** raise the pre-rerank candidate pool from current default
(needs verification — likely 30 or 50) to **75**. This pulls more channels
through the rerank step before trimming for the prompt.

**Targets:** Bucket B partial-retrieval cases (6 questions) where some gold
sits beyond the current pre-rerank K but inside the wider K. Realistic
fixes: 1-3 (high uncertainty — depends on actual rank distribution beyond
K=20 which we don't have data on).

**Code locations to inspect:**
- The reranker entry point — likely in `lme_weighted_rerank` or directly
  in `retrieve_with_omega_recipe` — needs an explicit K1 parameter.

**Verification:**
1. Unit test: `lme_weighted_rerank(candidates=80, top_k=30)` returns 30
   results; `lme_weighted_rerank(candidates=20, top_k=30)` returns 20
   (no padding).
2. Smoke test on `f9e8c073` (bereavement sessions, gold_count=2 in_pool=1):
   confirm with K1=75 that both gold sessions land in pool.

**Risk assessment:**
- **Reranker compute:** +250% rerank input → +1-2s per question.
  Acceptable.
- **Reranker quality:** more candidates may include more distractors.
  Mitigated by reranker's own scoring — top-N stays similar, just more
  raw input.
- **OpenAI not affected** — same prompt context cap.

**Ship cost:** ~20 min (5 LOC + 1 test + smoke).

### Patch D — Revised 5B: raw + expanded query fan-out

**Change:** in `retrieve_with_omega_recipe`, retrieve twice — once with the
raw query, once with the `expand_query()`-expanded query. RRF-fuse the two
ranked lists before reranking.

**Targets:** Bucket A zero-retrieval cases where the raw query is too vague
for the current expanded-only retrieval to land gold (5 questions).
Realistic fixes: 1-2 (uncertain — depends on whether raw vs expanded actually
hit different sessions for these vague queries).

**Code locations to inspect:**
- `retrieve_with_omega_recipe` function signature and body. Currently calls
  `scored_search(expanded_query, ...)` once. Will call twice + RRF-fuse.
- Reuse Stage 1.5's RRF logic if it exists; otherwise add a small
  `rrf_fuse(list_a, list_b, k=60)` helper.

**Verification:**
1. Unit test: raw query "what cocktail" + expanded "what cocktail recipe
   gin lavender 2023-05-30 every instance all occurrences" → both run
   through `scored_search`, results fused with RRF.
2. Smoke test on `8550ddae` (lavender gin fizz): confirm with dual fan-out
   that the gold session lands in pool.

**Risk assessment:**
- **Latency:** doubles retrieval time per question, +1-2s. Acceptable.
- **Compute:** doubles vector + keyword channel work. Cost negligible
  (Chroma + Neo4j queries are <100ms each).
- **Regression risk:** if raw query retrieves *worse* candidates that
  outrank expanded query's good candidates after RRF, we lose. Mitigated
  by: RRF naturally weighted to top-K of each list; the fused list contains
  the best of both.
- **Question pattern coverage:** only 5 questions in Bucket A; even at
  100% fix rate that's a small absolute lift.

**Ship cost:** ~45 min (15 LOC retrieval + 20 LOC RRF helper + 2 tests
+ smoke).

---

## 4. What I'm NOT doing this round

### 4.1 Stage 5C (two-stage retrieval for multi-step questions)

Targets: `ba358f49` (years old when Rachel marries — needs 2 facts), similar
compositional questions. Estimated lift: 1-2 questions.

**Defer because:** requires a "is this multi-step?" classifier — new
abstraction, not a parameter. Not worth speculative engineering before we
see what 5A+5G+revised-5B land.

### 4.2 Stage 5E (temporal hard-filter)

Stage 4D's `infer_temporal_range_anchored` boosts in-window candidates by
1.5×. Switching to a hard filter would directly attack `88432d0a` (bake count
past two weeks). Estimated lift: 2-4 questions.

**Defer because:** known failure mode — hard filters drop notes that are
*about* an in-window event but dated outside the window (e.g., a note
dated June 1 saying "I baked last weekend" should match a "past two
weeks" filter from May 25). Needs careful fallback design. Not surgical.

### 4.3 Stage 5F (better reranker for counting)

Estimated lift: 2-4 questions. **Defer because:** real engineering work
(maybe a day). Would only ship if we plateau below 95.4%.

### 4.4 Stage 4I or further prompt iteration

Don't iterate on prompts again until we see the 4E-4H validation result.
If 4E-4H is net-negative, that's where bisection happens. Don't compound.

---

## 5. Sequencing

```
T+0 ──► [in flight] Stage 4E-4H validation, ~30 min remaining
        │
T+0 ──► IMPL: Stage 5A (~30 min)
        │   ↳ inspect code, find cap, widen, test, smoke, commit
        │
T+30 ──► IMPL: Parallelism (~1.5 hrs)
        │   ↳ adapter --tenant-suffix flag (~30 min)
        │   ↳ coordinator scripts/run_parallel_lme.py (~45 min)
        │   ↳ parity test + tenant test (~15 min)
        │   ↳ commit
        │
T+120 ──► [validation] Stage 4E-4H result lands
        │   ↳ if net-positive: continue
        │   ↳ if net-negative: STOP, bisect 4E-4H
        │
T+120 ──► IMPL: Stage 5G (~20 min)
        │
T+140 ──► IMPL: Revised Stage 5B (~45 min)
        │
T+185 ──► VALIDATION: full 4E-4H + 5A + 5G + 5B in parallel (~12 min)
         │   ↳ via coordinator with --workers 4
         │
T+200 ──► JUDGE: parallel judge (~3 min, 8-worker)
        │
T+205 ──► [decision point]
        │   ↳ projected ≥96%: fire full 500q in parallel (~25 min)
        │   ↳ projected 95-95.5%: ship Stage 5C OR fire-and-hope
        │   ↳ projected <95%: bisect, do not fire 500q
```

**Estimated wall time from T+0 to headline number: ~4 hrs.**

## 6. Stop conditions (when to halt and call Alex)

1. **Stage 4E-4H validation comes back below 93.6% (Stage 4D's projection).**
   Means the prompt patches regressed. Bisect before stacking.
2. **Parity test fails (>2 of 30 questions differ).** Means parallelism has
   a leak. Don't run parallel headline; serialize.
3. **Stage 5A causes >2 regressions in the 30-rights regression sample.**
   Means widening context is hurting precision. Roll back to MS=25 only.
4. **Any stage-5 patch breaks more than it fixes** (negative net delta in
   targeted validation). Roll back, don't compound.

## 7. Total budget

| Item | Cost |
|---|---|
| Stage 5A code | $0 |
| Parallelism build | $0 |
| 5G + 5B code | $0 |
| Combined validation in parallel (104 qids) | ~$3 |
| Headline 500q in parallel | ~$33 |
| **Stage 5 round total** | **~$36** |

Within the "whatever it takes" budget. Total elapsed budget across all
stages including 4D + 4E-4H: ~$120 of ~$300 authorized.

## 8. Success criteria

The Stage 5 round is successful if **at least one** of the following holds
after final validation:

- (a) projected 500q score ≥ 96.0% (clear AgentMemory by margin)
- (b) projected 500q score ≥ 95.4% AND no regressions in regression sample

Anything above 95.4% is publishable as "matches OMEGA on gpt-4.1." Anything
above 96.2% is publishable as "highest LongMemEval score reported on
gpt-4.1." Above 95.4% but below 96.2% is publishable but less marketable.

---

## 9. Open questions to verify in step 1 of implementation

1. Is `n_hits_used: 20` in diagnostics the post-rerank trim or the pre-rerank
   pool size?
2. Where is the per-category cap actually set in the adapter?
3. Does `lme_weighted_rerank` accept a configurable `top_k`? Or is it
   currently hardcoded?
4. Does the existing `agent_id` scoping in `jarvis_memory/api.py` cover all
   the places `:LMETestEpisode` appears in `scripts/run_longmemeval.py`? Or
   do we need to add a new tenant param?
5. Is the Chroma collection name parameterized or hardcoded?

These are not blockers — they're step 1 of impl. Listed here so I don't
forget to verify before shipping.
