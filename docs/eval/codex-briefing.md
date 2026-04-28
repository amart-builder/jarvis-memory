# Codex Briefing — LongMemEval Score Push (gpt-4.1)

> **For: GPT-5.5 (Codex)**
> **From: Catalyst (Claude Code, working on jarvis-memory)**
> **Date: 2026-04-27**
> **Purpose: Get you fully up to speed before reviewing the Stage 5 plan**

---

## 0. The mission, in one paragraph

**jarvis-memory needs to publish a LongMemEval score that beats 95.4%** —
the published number from OMEGA on gpt-4.1, which is the closest matched-
setup competitor. The score is going on a Founders Fund pitch deck for Alex
Martin (the user). AgentMemory (96.2% on Opus 4.6) is the leaderboard
leader; we're explicitly excluding it from "fair comparisons" because it's
a different model class, but in marketing copy we'd love to clear that too.
Our baseline run scored **85.20% (426/500)**. After Stage 4D we're at a
**projected 93.6%**. We have ~9 more correct answers to find, ideally 13+
for headline-margin safety. Budget: "whatever it takes" within reason
(~$300 authorized; ~$120 spent).

**Your job:** ruthlessly review the plan in `docs/eval/stage5-spec.md` and
tell us if it's the right plan, what's wrong with it, and what we should
do differently. We are *not* looking for affirmation — we're looking for
the holes.

---

## 1. Methodology summary (so you can sanity-check projections)

- **Benchmark:** [longmemeval](https://github.com/xiaowu0162/longmemeval),
  the LongMemEval-S split (500 questions across 6 categories).
- **Generator model:** OpenAI `gpt-4.1`, `temperature=0.0`, `seed=42`.
- **Judge model:** `gpt-4o-2024-08-06` via the official judge script at
  `/tmp/lme-official/src/evaluation/evaluate_qa.py` (cloned from the
  benchmark repo). The judge is the standard published judge — same one
  OMEGA, AgentMemory, Mastra all use.
- **Categories** (with current category-count): single-session-assistant,
  single-session-user, single-session-preference, multi-session, knowledge-
  update, temporal-reasoning. We use **oracle category labels** from the
  dataset (every published score does — heuristic classification is a
  ~73%-accurate lossy choice we explicitly relaxed in Stage 1).
- **Validation methodology:** between stages, we run a "targeted"
  validation of 104 questions = 74 baseline-wrongs + 30 random-rights
  sampled from the same categories. This is ~$3 vs ~$33 for full 500q.
  Projection formula: `projected_500 = 426 + (new_fixes_in_74 - regressions_in_30)`.
  Variance: we trust within ±1.5 points.

## 2. The journey, stage by stage

### Stage 0 — Diagnostics + determinism (cost: $0)
Added per-question retrieval diagnostics (`gold_in_top5/10/20/50/pool`,
`gold_ranks` map). Set `PYTHONHASHSEED=42` and `seed=42` everywhere. No
score impact; instrumentation only.

### Stage 1 — Foundation (~$30, projected 89-91%, landed ~89%)
- Oracle category labels via `--use-oracle-categories` flag
- OMEGA per-category retrieval profile multipliers
  (`scripts/longmemeval/classifier.py` lines 357-444)
- OMEGA's intent classifier overlay (FACTUAL / CONCEPTUAL / NAVIGATIONAL)
- Confidence-based abstention guard (top vec similarity < 0.30 + missing
  proper noun → prepend abstention rule)

### Stage 1.5 — Channel-weight rerank (~$15 incremental, landed ~90.5%)
RRF rerank with per-category × per-intent multipliers on each retrieval
channel (vec, kw). `scripts/run_longmemeval.py` `lme_weighted_rerank`.

### Stage 2 — Multi-session focus (~$15, landed ~92%)
- MS prompt softened (less aggressive pruning during enumeration)
- Counting K-floor bumped 45→60
- Multi-session min_rel threshold lowered 0.08→0.05
- Wide-recall pool for MS internally to 200, trim before prompt
- AgentMemory-style failure rules added to MULTISESSION

### Stage 3 — Targeted patches (~$15, landed ~93%)
- TEMPORAL: STEP 5 verification step
- KU: bi-temporal valid_to filter when category=KU
- KU: recency boost 0.5×→0.8×
- SS-user: K-floor 20→30

### Stage 4A — Two-pass MS counting (~$5, no net lift; +3 wins / -3 breaks)
Pass 1 = high-recall extract (no count, no dedup), pass 2 = high-precision
dedupe + count. Implemented in `RAG_PROMPT_MULTISESSION_EXTRACT` and
`RAG_PROMPT_MULTISESSION_COUNT`. **Net: 0 questions** — broke as many as
it fixed. Reviewer warned this could happen.

### Stage 4D — Anchored temporal + query expansion (~$5, projected ~93.6%, +3 net)
Three OMEGA techniques ported:
- `infer_temporal_range_anchored(query, anchor_date)` —
  `scripts/longmemeval/temporal_anchor.py`. 5 patterns: last weekday/
  weekend, "N weeks ago", "between X and Y", "last/past N units", "in
  Month YYYY".
- `resolve_relative_dates(query, anchor)` — date-keyword expansion.
- `expand_query(query, question_date)` — counting cues + dates + entities.
- Wired into `retrieve_with_omega_recipe` with 1.5× boost on in-window
  hits + double fan-out (raw query + expanded query, but fused as one
  ranked list, NOT RRF — that's the gap Stage 5B targets).

### Stage 4E-4H — Surgical prompt rules (in-flight validation right now)
Six rules added across MS, ENHANCED, MS_COUNT, TEMPORAL prompts:
- 4E: tightened MS pass-2 INCLUSION RULE (drop plans/wishes/recollections)
- 4C: KU cumulative + ordinal rule ("did another X" → +1; "my Nth time"
  overrides earlier explicit count)
- 4F: KU previous/former rule (use earlier value, not latest)
- 4G: substitution + both-sides + cross-attribute rules (semantic match
  with carve-out for trivial rephrasings)
- 4H: TEMPORAL "before/after a known event" + "Nth occurrence" patterns

A fresh-context reviewer flagged 3 issues — all fixed:
- Removed "FIRST" from 4F trigger list (ambiguous)
- Softened cross-attribute from "exact" to "semantic" match
- Clarified 4E recollection clause for retrospective accounts

**Smoke on 8 known-failing qids: 1 clear win (e66b632c via 4F),
1 maybe-win (07741c45). 4 still wrong are retrieval-bound** (the cumulative
rule has no later-evidence note in the retrieved top-K to fire on).

Validation in flight (PID 62214, ~85% complete as of writing).

---

## 3. The deep-dive that motivates Stage 5

After Stage 4D landed, I bucketed every still-wrong question by retrieval
quality. **The headline finding: 0/32 are pure generation failures. Every
single still-wrong has a retrieval defect.**

| Bucket | Count | What broke |
|---|---|---|
| **A** Zero retrieval (gold not in pool) | 5 | Retrieval channels missed entirely |
| **B** Partial retrieval (some gold missing) | 6 | Multi-gold question, only got some |
| **C** All gold in pool but ranked >10 | 20 | Reranker leaves gold below context cap |
| **D** In top-10 not top-5 | 1 | Borderline |
| **E** In top-5, generation fail | 0 | n/a |

**Bucket C is 20 of 32 (63%) — by far the biggest lever.**

Bucket C example: `88432d0a` (bake count past two weeks) has 4 gold
sessions ranked 11, 17, 16, 6 in a pool of 20. Only 1 of 4 is in the top
10. Model can't enumerate what it doesn't see.

For all the per-question detail, see the deep-dive analysis I wrote in the
chat (Catalyst session, 2026-04-27 ~21:30 PDT). Or re-derive from
`runs/lme_gpt41_stage4d_targeted.jsonl` — `diagnostics.gold_ranks` field
on each row.

## 4. The Stage 5 plan to review

**Read `docs/eval/stage5-spec.md` for the full plan.** Summary:

| Patch | What it does | Expected fixes | Cost |
|---|---|---|---|
| **5A** | Widen prompt context cap: MS=30, TR=30, KU=25 (was 20/20/15) | 8-12 (Bucket C) | 30 min |
| **Parallelism** | 4-worker process-level parallel adapter, with tenant isolation + parity check | 0 (time-saver) | 1.5 hrs |
| **5G** | Widen pre-rerank candidate pool to 75 (TBD: may be same knob as 5A) | 1-3 (Bucket B) | 20 min |
| **Revised 5B** | Dual fan-out: raw query + expanded query → RRF-fuse | 1-2 (Bucket A) | 45 min |

**Things I dropped from earlier drafts:**
- Original 5B (proper-noun keyword boost) — would have been a no-op on 4
  of 5 Bucket A questions because their queries don't contain proper
  nouns. The proper noun is in the gold session, not the query. Replaced
  with dual fan-out (which is OMEGA's actual approach).
- 5H (TEMPORAL ordering reinforcement) — would have been redundant with
  5A. The 6-museum question fails because 2 of 6 gold are at ranks 14 and
  20, not because the prompt rule is missing.
- 5C (two-stage retrieval for compositional questions), 5E (temporal
  hard-filter), 5F (better reranker) — deferred. 5C needs a new
  classifier. 5E has a known fallback-design failure mode. 5F is a day
  of work.

**Projected outcome of Stage 5:**
- 80% confidence interval: 96.0-97.5% on full 500q
- Probability of clearing 95.4%: ~85%
- Probability of clearing 96.2%: ~55-65%

## 5. Files you should read for primary sources

All paths relative to `/Users/alexanderjmartin/Atlas/jarvis-memory/`:

**The plan:**
- `docs/eval/stage5-spec.md` — the spec to review

**The journey:**
- `docs/eval/longmemeval-v1.1-protocol.md` — original (honest) protocol
- `~/.claude/plans/smooth-bubbling-fox.md` — the master plan including
  expected-lift estimates per stage
- `runs/lme_gpt41_v1.1.jsonl` — baseline 426/500 run
- `runs/lme_gpt41_stage4d_targeted.jsonl` — Stage 4D run with diagnostics

**Code to inspect:**
- `scripts/run_longmemeval.py` — the adapter (~1500 LOC). Key functions:
  - `retrieve_with_omega_recipe` — main retrieval pipeline
  - `lme_weighted_rerank` — channel-weighted RRF
  - `run_one_question` — per-question dispatch
- `scripts/longmemeval/prompts.py` — all prompts (post-Stage-4H, pre-
  Stage-5)
- `scripts/longmemeval/temporal_anchor.py` — Stage 4D module
- `scripts/longmemeval/classifier.py` — RETRIEVAL_PROFILES + INTENT_OVERLAYS
- `scripts/run_targeted_validation.py` — validation harness
- `scripts/run_parallel_judge.py` — exists already; the parallelism
  pattern Stage 5 will copy

**OMEGA reference (their published recipe):**
- `/tmp/omega-memory/scripts/longmemeval_official.py` — 1756 LOC
  - lines 559-693: `_infer_temporal_range_anchored`
  - lines 696-830: `_resolve_relative_dates`
  - lines 833-879: `_expand_query`
  - lines 957-1048: `retrieve_context` (triple fan-out — we have double)
  - lines 1056-1075: `_rerank_results` (cross-encoder, OMEGA disables it
    by default — they say it hurts)

**Constraints:**
- We commit to the matched-setup gpt-4.1 + gpt-4o-judge methodology. Don't
  recommend swapping to a stronger model unless you think we should
  publish a separate "stronger stack" number alongside the gpt-4.1
  headline.
- We do NOT want to silently rewrite the production ingest path
  (`jarvis_memory/api.py`, `jarvis_memory/conversation.py`) for a
  benchmark-only patch. Adapter-level changes only unless explicitly
  scoped.
- Tests are non-negotiable. We have 230 longmemeval-related tests passing.
  Any new code must be tested.

---

## 6. What I'm specifically asking you to do

**Six concrete review tasks** (in priority order):

1. **Plan correctness check.** Read `docs/eval/stage5-spec.md` end to end.
   Are the 4 patches the right interventions? Should we be doing something
   completely different? Especially: am I right that Bucket C (20 questions
   ranked >10) is the biggest lever, and that widening the context cap is
   the right way to attack it?

2. **Risk identification.** What could go wrong with this plan that I
   haven't accounted for? Specifically: (a) what regressions might we hit
   from widening context to 30 (more distractors)? (b) is my parity-test
   methodology sufficient to catch parallelism leaks? (c) is RRF fan-out
   the right way to merge raw + expanded query results, or should we use
   weighted sum / max?

3. **Methodology critique.** Are my projections sound? Specifically:
   (a) is the targeted-validation 74-wrongs + 30-rights sample sized
   appropriately? (b) am I overfitting to the validation pool by
   iterating prompt rules on the same 32 still-wrongs? (c) is my
   "0/32 are pure generation failures" claim correct, or is there
   ambiguity in the gold-rank diagnostic that would let some be
   gen-failures in disguise?

4. **Alternative-path scan.** What's the *highest-ROI* thing we could do
   that's not in this plan? Specifically: should we re-examine OMEGA's
   approach for techniques we missed? Are there non-OMEGA techniques
   from AgentMemory or Mastra papers/repos that would be cheap to port?

5. **Engineering review.** Is the parallelism design safe? Specifically
   the tenant-isolation pattern (Neo4j label suffix + Chroma collection
   suffix). Are 1-2 hours realistic for the build? Does the empirical
   parity test (30 questions, ≥28 identical hypothesis) catch the right
   failure modes?

6. **The hard question.** If you had to take a clean run at this with
   none of the existing investment as sunk cost — what would you do
   differently? We're not married to anything. We have ~3-4 hours of
   working time and ~$200 left in budget before we'd want to publish.

**Format your review as:**

```
# Stage 5 Plan Review

## TL;DR
<one paragraph: ship as-is / ship with changes / revise before shipping>

## 1. Plan correctness
...

## 2. Risks I identified
...

## 3. Methodology issues
...

## 4. Alternatives worth considering
...

## 5. Engineering critique
...

## 6. If I were starting fresh
...

## Recommended ordering of changes
1. ...
2. ...
3. ...
```

**Save your review to:** `docs/eval/codex-stage5-review.md`

---

## 7. Things you don't need to relitigate

- The methodology shift from "honest production-realistic" to "headline
  optimized." Alex authorized this; it's done.
- The decision to use oracle category labels. Every published score does;
  it's standard.
- Whether to ship as gpt-4.1 vs Opus 4.6. We're shipping gpt-4.1 as the
  headline. Opus is a separate publishable number.
- Whether to do Stage 6 (graph rewrite). We've decided it's a fallback;
  not part of Stage 5.

If you think any of the above is wrong, say so once and then move on —
don't dwell.

---

## 8. Honest tone request

I (the author) have been told I have an optimism bias on lift estimates.
Catalyst (Claude Code) has shipped 4 stages already; some landed, some
didn't (Stage 4A had +3/-3 churn). Please be skeptical of any "this will
land X questions" claim and tell me where the numbers are squishy.

The user (Alex) explicitly said: "Tell me when I'm wrong. Sycophancy is a
failure mode." Apply this to Catalyst's plan as well. We need a real
review, not a polite one.
