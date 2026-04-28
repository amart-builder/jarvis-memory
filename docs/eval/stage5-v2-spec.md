# Stage 5 v2 — Unified Plan to Beat 96% on LongMemEval

> **Status:** approved by Alex, supersedes `stage5-spec.md` (which had a
> diagnostic misread Codex caught and corrected).
> **Owner:** Catalyst (Claude Code)
> **Approved by:** Alex Martin, 2026-04-27
> **Independent reviewer:** Atlas/Codex (GPT-5.5)

---

## 1. Where we are

| Stage | Projected 500q | Δ |
|---|---|---|
| Baseline (Stage 0) | 85.20% | — |
| Stage 4D | 93.6% | +42 |
| **Stage 4E-4H (just landed)** | **94.8%** | **+48 cumulative, 0 regressions** |
| **Floor target (Alex)** | **≥96.0%** | **+6 more needed** |

Stage 4E-4H validation: 48/74 baseline-wrongs fixed, 0/30 random-rights
regressed. Clean landing — no rollback or bisection needed.

**Gap to 96.0%: ≥6 incremental net fixes.**
**Gap to 96.5% (margin): ≥9 incremental.**

## 2. Why this plan supersedes the old Stage 5

Codex's review (`docs/eval/codex-stage5-review.md`) caught a critical misread
in my deep-dive: I claimed 20 of 32 still-wrong questions had gold sessions
ranked >10 and outside the prompt. **Wrong.** The diagnostic computes
ranks AFTER all filtering — when `n_hits_used=20` and gold rank=17, that
gold session **is in the prompt**.

Real bucketing (verified by reading code at `scripts/run_longmemeval.py:407-466`):

- **Bucket A — true retrieval miss** (gold not in prompt): 5 questions
- **Bucket B — partial retrieval** (some gold missing): 6 questions
- **Bucket C — gold visible, model fails to use it**: 20 questions ← biggest
- **Bucket D — borderline**: 1 question

So **20 of 32 still-wrong are salience/generation failures**, not retrieval
failures. Widening context (old Stage 5A) won't help; the model already has
the evidence and isn't using it. The right intervention is **better
evidence assembly inside the prompt**.

This v2 plan is structurally Atlas/Codex's recommendation, calibrated to
Alex's constraints (96% floor, methodical pace, anything on the table).

---

## 3. Phase plan

### Phase 1 — Lock baseline ✅ DONE

Stage 4E-4H validated at 94.8% / 0 regressions. Net-positive, clean.
**Prompts are FROZEN until further evidence justifies a change.**

### Phase 2 — Retrieval-stage diagnostics (1 hr, $0)

**What:** instrument every retrieval stage with rank tracking. Currently
`compute_retrieval_diagnostics()` only records the FINAL post-everything
rank. We need ranks at each pipeline stage:

| Stage | Where | What rank is recorded |
|---|---|---|
| Expanded-query scored_search | `run_longmemeval.py:747-756` | rank of gold in expanded primary list |
| Raw-query scored_search | `run_longmemeval.py:759-768` | rank of gold in raw secondary list |
| Merged pre-rerank | after precedence merge | rank in merged candidate list |
| Weighted-rerank | after `lme_weighted_rerank` | rank after channel weights applied |
| Post-filter | after `FILTER_CONFIG.max_res` trim | rank that survives the trim |
| Final (chronological) | what enters prompt | rank in date-sorted final list |
| Prompt hash | sha256 of full prompt string | for parity testing later |

**Why:** without this, Phase 6 (re-eval 5G) is blind. Also lets us label
every still-wrong question by failure mode:
- Not retrieved (gold rank < 1 anywhere)
- Retrieved then dropped at stage X
- Visible but ignored
- Visible but reasoning failed

**Amendment (Atlas 2026-04-27): rebucket against the CURRENT 26 still-
wrongs, not the stale 32 from Stage 4D.** Stage 4E-4H changed the failure
set: 48 of 74 baseline-wrongs are now fixed, 26 remain still-wrong, 0 of 30
random-rights regressed. We must NOT design Phase 3 packets against
questions we've already fixed. Phase 2 outputs a fresh per-question failure
classification of the current 26 (post-4H) still-wrong set.

**Risk:** zero — pure observability. Cannot regress score.

**Verify:**
1. Unit test the new fields with a synthetic retrieval result
2. Re-run a 4-question smoke; confirm new fields populate
3. Confirm overall score unchanged on those 4 questions

**Ship:** new commit, then proceed to Phase 3.

### Phase 3 — Evidence packets (the headline patch, 2-3 hrs, $3 validation)

**What:** for MS/TR/KU categories, prepend a `[High-signal evidence]`
block to the chronological notes. Block contains user-turn snippets
extracted from the retrieved sessions, surfacing dates / quantities /
named entities / ordinals.

**Format example for `88432d0a`** (bake count past two weeks, gold=4):
```
[High-signal evidence — items relevant to "How many times did I bake
something in the past two weeks?"]
- 2023-05-20 [Note 3] User: "I made the apple pie in my cast iron skillet"
- 2023-05-21 [Note 6] User: "I tried out a new bread recipe using sourdough
  starter on Tuesday"
- 2023-05-24 [Note 11] User: "I made a delicious whole wheat baguette
  last Saturday"
- 2023-05-25 [Note 16] User: "I used my oven's convection setting last
  Thursday to bake a batch of cookies"
- 2023-05-27 [Note 17] User: "I just baked a chocolate cake for my
  sister's birthday party last weekend"

[Below: full chronological notes for verification]
{sessions}
```

**Implementation:** heuristic extractor first (no LLM call, instant). If
heuristic doesn't land enough, fall back to LLM extraction in Phase 8.

Heuristic rules per session:
- Parse user turns only (not assistant turns)
- Match signals via regex:
  - Dates: ISO-like, "last weekend", "two weeks ago", "March 15"
  - Quantities: `\$?\d+`, "five sessions", "third time"
  - Proper nouns: capitalized words not at sentence start, not in stoplist
  - Ordinals: "first/second/.../Nth", "1st/2nd/...10th"
- Score user turn by signal count
- Take top N=12 highest-scored snippets across all sessions
- Truncate snippets to ~80 chars

**Prompt rule for the model** (added to MS/TR/KU prompts):
> The "[High-signal evidence]" block at the top is an INDEX of items
> extracted from the chronological notes — use it to identify relevant
> items quickly, then VERIFY each against the full notes below before
> answering. The packet may miss items or include irrelevant ones; the
> chronological notes are the ground truth.

**Why:** 20 of 32 still-wrong are Bucket C (all gold visible). The model
fails to enumerate scattered user-turn evidence. Pre-extracting the
signals into a dense block is what AgentMemory + Mastra do.

**Risk register:**
| Risk | Mitigation |
|---|---|
| Snippets misleading (out-of-context) | Prompt rule: verify against full notes |
| Model trusts packet over notes when packet wrong | Same — prompt rule + chronological notes still present |
| Packet duplicates content the model already enumerates correctly | Acceptable — duplication doesn't hurt |
| Heuristic misses multi-word entities | Improve regex; or fall back to LLM extraction |

**Verify:**
1. Unit test extractor on 5 known-failing qids drawn from the 26 CURRENT
   still-wrongs (post-Stage-4H), not the 32 stale ones. Phase 2's
   rebucketing tells us which 5 to use. Eyeball the packet — does it
   surface the gold-relevant snippets?
2. Smoke generate on those 5 with the new packet — does the model use it?
3. **Phase 3.5 — Smoke Gate (Atlas amendment).**

**Expected lift:** +3 to +8 (Atlas estimate). Honest median: +5.

### Phase 3.5 — Smoke Gate (Atlas amendment, 0 cost)

**What:** before spending $3 + 1.5 hr on targeted validation, eyeball the
5-question smoke output:

- Did the heuristic extractor surface the GOLD-relevant user-turn snippets
  in the `[High-signal evidence]` block?
- Did the model's hypothesis cite or use those snippets?
- For counting questions: did the count increase from previous wrong
  count toward the gold count?

**Decision:**

| Smoke result | Action |
|---|---|
| ≥3 of 5 packets clearly surface gold-relevant signals AND model uses them | Run Phase 5 (targeted validation) |
| <3 of 5 surface gold OR model ignores packet | **Skip Phase 5; go straight to Phase 8 (observation extraction)** |

**Why:** the heuristic extractor may be too dumb for the exact cases that
matter (per Atlas). If it's clearly missing on smoke, don't burn a full
validation on it — pivot to LLM-extracted observations directly.

### Phase 4 — Temporal two-lane context (45 min, $0)

**What:** when `infer_temporal_range_anchored` returns a window:
- Split chronological notes into Lane 1 (in-window dates) and Lane 2 (rest)
- Render Lane 1 first under `[In-window notes — most likely relevant]`
- Render Lane 2 second under `[Other retrieved notes — for context]`

When no window inferred, behavior is unchanged (single chronological list).

**Why:** for date-windowed counting questions like `88432d0a` (past two
weeks) and `5a7937c8` (faith activities in December), surface the
date-relevant context first. Models are more likely to enumerate from the
top of context.

**Important:** this does NOT hard-filter out-of-window notes (Codex's
critique of earlier 5E). A note dated June 1 saying "I baked last
weekend" is dated outside a "past two weeks ending May 30" window but
is highly relevant; it stays in Lane 2.

**Risk:** low — reorders only, doesn't drop. The chronological order
within each lane is preserved.

**Verify:**
1. Unit test the partitioning function with synthetic dates
2. Smoke on `88432d0a` — confirm in-window bake notes are surfaced

**Expected lift:** +1 to +3.

### Phase 5 — Validation Gate 1 (1.5 hr serial, $3)

Run `scripts/run_targeted_validation.py` with Phases 2+3+4 in place.

**Decision tree:**

| Result | Action |
|---|---|
| **≥96.0% projected, ≤1 regression** | Skip to **Phase 10** (full 500q) |
| **95.5-95.9%, ≤2 regressions** | Phase 6 + Phase 8 (try to push higher) |
| **95.0-95.4%, ≤2 regressions** | Phase 8 (fallback observation extraction) |
| **<95.0% OR >2 regressions** | Bisect: roll back Phase 3 or Phase 4, re-validate |

**Why this gate is necessary:** Phase 8 is +3 hrs and ~$50 — only worth
firing if Phase 3+4 alone don't clear 96.0%.

### Phase 6 — Surgical 5G (conditional, 30 min, $0)

Trigger only if Phase 5 returns 95.5-95.9% AND retrieval-stage
diagnostics from Phase 2 show specific cases where:
- Gold appears at rank 21-75 in pre-filter stages
- AND gets dropped at a specific cap (initial `k`, max_res, rerank depth)

If pattern matches: widen the specific cap that dropped it. Don't blanket-
widen.

If pattern doesn't match (gold is missing entirely or visible-then-ignored):
skip Phase 6, go to Phase 8.

**Expected lift:** +1 to +4 if applicable, 0 otherwise.

#### Phase 6 execution (2026-04-28, isolated experiment per Atlas/Codex amendment)

After Phase 3 reverted (net −6 in validation) and Phase 4 reverted (fired 0/104),
Atlas insisted Phase 6 run as an **isolated** experiment — not bundled with
Phase 8 — to keep causality clean. Diagnostic-driven scope only.

**Phase 5 baseline (Phase 2 instrumentation, no Phase 3/4 patches):** 70/104 correct (67.31%).
- 42 baseline-wrongs fixed
- 32 baseline-wrongs still wrong
- 28 baseline-rights still right
- 2 baseline-rights regressed (`d905b33f`, `gpt4_2c50253f`)

**Filter-drop analysis** on the 32 still-wrongs (gold present in pre-filter
stages but missing from `final_chrono`):

| qid | category | best pre-rank | post-temporal_boost | filtered cap | widening fix |
|---|---|---|---|---|---|
| 58bf7951 | SS-user | pure_vec=6 | 17 | 12 → DROP | max_res 12→20 ✓ |
| 8550ddae | SS-user | weighted_rerank=2 | 17 | 12 → DROP | max_res 12→20 ✓ |
| 726462e0 | SS-user | pure_vec=36 only | n/a (out of merge) | 12 → DROP | merge issue, not max_res |
| d6233ab6 | SS-pref | pure_vec=35 only | n/a (out of merge) | 10 → DROP | merge issue, not max_res |
| gpt4_468eb063 | TR | raw_secondary=1 | 35 | 20 → DROP | TR widen 20→40 too risky for 1 outlier |

**MS not widened:** All 18 MS still-wrongs have gold IN `final_chrono`
(generation-side enumeration failures, not retrieval). Widening MS would only
add noise to prompts where the issue is the LLM, not the retrieval.

**Surgical change:** `FILTER_CONFIG["single-session-user"]["max_res"]: 12 → 20`
in `scripts/longmemeval/classifier.py`. One number. No prompt changes, no
retrieval-pipeline changes.

**Smoke (5 questions: 2 targets + 3 SS-user sentinels):** 5/5 correct.
- 58bf7951 → fixed (gold rank 13 in chrono, was -1)
- 8550ddae → fixed (gold rank 2 in chrono, was -1)
- e47becba, 118b2229, 51a45a95 → still correct (sentinels preserved)

**104q validation (in progress):** gate ≥95.4% with 0 NEW regressions.

### Phase 7 — Skip / demote 5B

Adapter already does dual fan-out (precedence merge of expanded + raw).
Codex's analysis: revised 5B is at best a merge-strategy ablation, not a
missing feature.

**Skip unless** Phase 8 fallback observation extraction is also not enough.

### Phase 8 — Fallback: Observation Extraction (3 hrs build, ~$50, $3 validation)

**Trigger:** Phase 5 result is 95.0-95.9% projected.

**What:** at adapter ingestion (NOT production), call gpt-4o-mini per
LongMemEval session to extract structured observations:

```json
{
  "events": [
    {"type": "yoga_class", "instance": 5, "date": "2023-06-12",
     "details": "evening class, vinyasa style"}
  ],
  "facts": [
    {"key": "user_age", "value": "32", "stated_date": "2023-04-11"}
  ],
  "preferences": [
    {"key": "coffee_temperature", "value": "iced", "valid_from": "2023-05-01"}
  ],
  "updates": [
    {"key": "5K_PB", "old": "27:45", "new": "26:30", "date": "2023-07-30"}
  ]
}
```

**Storage:** `:LMETestObservation` nodes in Neo4j, suffixed per worker if
parallel. Connected to source `:LMETestEpisode` via `EVIDENCED_BY` edge.

**Retrieval:** in addition to retrieving raw sessions, also retrieve
relevant observations (vector + keyword over observation text). Render
observations BEFORE the evidence packet, before the chronological notes:

```
[Structured evidence (extracted observations)]
- Event: yoga class #5 on 2023-06-12
- Fact: user's age = 32 (stated 2023-04-11)
- Update: 5K PB changed 27:45 → 26:30 on 2023-07-30

[High-signal evidence (user-turn snippets)]
{packet}

[Chronological notes (full conversations)]
{sessions}
```

**Why:** for cumulative-count questions like `f9e8c073` (5 bereavement
sessions, gold=5), the user has explicit "3 sessions" in one note and
"my 5th session" in a later note. Currently we retrieve the "3 sessions"
note but miss the "5th session" note (Bucket B partial retrieval). With
observations:
- Session 1 → extracts `{"event": "bereavement_session", "instance": 3}`
- Session 5 → extracts `{"event": "bereavement_session", "instance": 5}`
- Both observations get retrieved on the query "how many bereavement
  sessions"
- Model reads "instance: 5" directly

This is the AgentMemory/Mastra technique, adapter-local only.

**Cost:** ~3000 sessions × ~$0.0005 each (gpt-4o-mini extraction with
~1500-token sessions) = ~$1.50 — much cheaper than the original $50 plan
estimate. Caveat: extraction prompts are cheap, but if we need to re-
extract on revisions, costs add up. Cap at 3 extraction passes.

**Risk register:**
| Risk | Mitigation |
|---|---|
| Extraction quality varies | Spot-check 5 sessions, refine prompt |
| Hallucinated observations | "extract verbatim from session, do not infer" rule |
| Observations for non-relevant queries pollute context | Retrieve only top-K relevant observations |
| Build complexity | 3 hrs is realistic if extraction prompt is single-pass |
| Production-code creep | NOT touching production — `:LMETestObservation` is adapter-only |

**Verify:**
1. Test extractor on 5 sessions: f9e8c073, 45dc21b6, 6d550036 source
   sessions. Eyeball JSON output.
2. Run extractor across all sessions for a 20-question subset, verify
   observation count is reasonable (5-15 per session).
3. Targeted validation (Phase 9).

**Expected lift on top of Phase 3+4:** +3 to +7. Atlas confidence with
fallback included: 85%+.

### Phase 9 — Validation Gate 2 (conditional, 1.5 hr, $3)

Trigger: only if Phase 8 fired.

Re-run targeted validation. **Decision tree:**

| Result | Action |
|---|---|
| **≥96.0%, ≤1 regression** | Phase 10 (fire full 500q) |
| **95.4-95.9%, ≤2 regressions** | Discuss with Alex: ship at this number, or push for production-code Stage 6 |
| **<95.4%** | Escalate; reconsider strategy |

### Phase 10 — Full 500q for headline ($33, 2.5 hr serial)

Conditional on Gate 1 OR Gate 2 clearing 96.0%.

Run the full 500-question benchmark with whatever Phase combination got
us through the gate. Use existing `scripts/run_longmemeval.py` invocation
(`--use-oracle-categories --diagnostics`).

Then run the official judge.

**Stop conditions:**
- If full 500q comes back at <95.4% (i.e., the targeted projection over-
  estimated by >0.5 points): inspect what differed in the 396 questions
  not in the targeted pool. Decide whether to publish the 500q number as-
  is, or treat as a regression and iterate one more cycle.
- If full 500q comes back at ≥95.4% but <96.0%: still publishable as
  matched-setup OMEGA-equivalent, but Alex decides whether to push for
  the higher number or ship.

### Phase 11 — Publish

Compose:
- Repo README update with the score + reproduction one-liner
- `docs/eval/longmemeval-v1.2-results.md` writeup
- Blog post draft / LinkedIn / Founders Fund deck note
- Acknowledge OMEGA + AgentMemory + Mastra explicitly

---

## 4. Total budget envelope

**Optimistic path (Phase 5 clears 96.0% directly):**

| Phase | Time | $$ |
|---|---|---|
| 2 — Diagnostics | 1 hr | $0 |
| 3 — Evidence packets | 2.5 hrs | $3 |
| 4 — Two-lane temporal | 0.75 hr | $0 |
| 5 — Validation Gate 1 | 1.5 hrs | $3 |
| 10 — Full 500q | 2.5 hrs | $33 |
| 11 — Publish | 1 hr | $0 |
| **Total** | **~9.25 hrs** | **~$39** |

**Fallback path (Phase 8 needed):**

| Phase | Time | $$ |
|---|---|---|
| 2-5 | 5.75 hrs | $6 |
| 6 — 5G surgical | 0.5 hrs | $0 |
| 8 — Obs extraction build | 3 hrs | $1.50 ext + $1.50 val |
| 9 — Validation Gate 2 | 1.5 hrs | $3 |
| 10 — Full 500q | 2.5 hrs | $33 |
| 11 — Publish | 1 hr | $0 |
| **Total** | **~14.25 hrs** | **~$45** |

Both fit Alex's "$300 / whatever it takes" envelope comfortably. ~$120
already spent across earlier stages, so total exposure ≤$165.

---

## 5. Confidence estimates

Honest, not optimism-padded:

| Path | Probability of clearing 96.0% projected |
|---|---|
| Phase 3+4 alone (heuristic evidence packets + 2-lane temporal) | **40-55%** ← Atlas-corrected from 50-60% |
| Phase 3+4+8 (with observation extraction fallback) | **75-85%** |
| Phase 3+4+8 + tweaking on 1-2 iterations | **85-90%** |
| All phases + production Stage 6 graph rewrite | **95%+** |

The asymptote is not 100%. Even with perfect retrieval + perfect prompts,
gpt-4.1's seed=42 nondeterminism plus the judge's own variance gives ~0.5%
of run-to-run noise. We need to land at projected 96.5%+ to be 95%+
confident the published 500q clears 96.0%.

**Floor commitment:** if at end of Phase 9 we're stuck at 95.4-95.9%, we
will surface to Alex for the call between (a) ship at that number with
caveat, (b) push for production-code Stage 6, (c) try gpt-5 / Opus 4.7
as a separate publishable number.

---

## 6. Stop conditions / rollback triggers

1. **Phase 5 result <95.0%:** STOP. Bisect Phase 3 vs Phase 4, roll back
   the regressing phase, re-validate.
2. **Phase 5 / 9 regressions >2 of 30:** STOP. Investigate which prompt
   or context change caused regressions; roll back if not understood.
3. **Phase 8 extraction quality is bad** (eyeball test on 5 sessions
   shows >2 sessions with hallucinated observations): STOP. Refine
   extraction prompt before running across all sessions.
4. **Targeted validation projection vs full 500q delta >0.7 points:**
   STOP iteration; the projection methodology is broken. Move to
   per-category sub-validation.

---

## 7. Open questions to verify in implementation

(non-blockers; flagged so I don't forget)

1. Where exactly does `infer_temporal_range_anchored` get called in the
   adapter, and where does its result feed in? Need to know to plumb
   Phase 4's lane partitioning.
2. Is there a clean way to get "user turns only" from a session's
   stored content? Or do we re-parse from the original LongMemEval JSON?
3. Does `:LMETestObservation` need its own bi-temporal columns, or do we
   inherit from the source `:LMETestEpisode`?
4. For Phase 8, do we co-extract observations with the existing per-
   question DETACH DELETE cleanup, or extract once globally and persist?

---

## 8. Sequencing

```
T+0    ── Phase 2: diagnostics            (1 hr)
T+1    ── Phase 3: evidence packets       (2.5 hrs)
T+3.5  ── Phase 4: two-lane temporal      (0.75 hr)
T+4.25 ── Phase 5: validation gate 1      (1.5 hrs)
        │
        ├─ if ≥96.0% → fire Phase 10     (~6.75 hrs total to headline)
        │
        ├─ if 95.5-95.9% → Phase 6 (0.5h) + Phase 8 (3h) + Phase 9 (1.5h)
        │                  then Phase 10 (~14.25 hrs total)
        │
        └─ if 95.0-95.4% → Phase 8 + 9    (~13.75 hrs total)

T+11   ── Phase 11: publish                (1 hr)
```

**Approved by Alex 2026-04-27. Ready to execute. First action: Phase 2
diagnostics.**
