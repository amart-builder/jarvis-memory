# Prompt for Codex (GPT-5.5) — Stage 5 Plan Review

> Paste this entire file into your Codex session, or feed it as the
> initial system/user message.

---

You are reviewing a working plan from another AI agent (Catalyst, a Claude
Code instance) for pushing jarvis-memory's score on the LongMemEval
benchmark from a projected 93.6% to ≥95.4%. The user (Alex Martin, a non-
technical founder) has asked for your independent ruthless review.

## Step 1 — Get up to speed

Read this file first, in full:

`/Users/alexanderjmartin/Atlas/jarvis-memory/docs/eval/codex-briefing.md`

It contains the mission, methodology, complete journey through Stages
0-4D, the failure-mode deep-dive that motivates Stage 5, the Stage 5
plan summary, and a list of supporting source files.

## Step 2 — Read the actual plan

`/Users/alexanderjmartin/Atlas/jarvis-memory/docs/eval/stage5-spec.md`

This is the document under review. Read it end to end.

## Step 3 — Pull the primary sources you need

The briefing lists every file you might need under §5. At minimum you'll
likely want:

- `scripts/run_longmemeval.py` — the adapter (how retrieval actually works
  today)
- `scripts/longmemeval/prompts.py` — the current prompts (post-Stage-4H)
- `scripts/longmemeval/temporal_anchor.py` — the Stage 4D temporal module
- `runs/lme_gpt41_stage4d_targeted.jsonl` — the Stage 4D validation output
  with full retrieval diagnostics on every still-wrong question
- `/tmp/omega-memory/scripts/longmemeval_official.py` — OMEGA's reference
  implementation (the 95.4% baseline we're trying to beat)

Don't try to read the entire adapter — use grep/glob to pull the specific
functions Catalyst names in the briefing (`retrieve_with_omega_recipe`,
`lme_weighted_rerank`, etc.).

## Step 4 — Write the review

The briefing §6 lists six specific review tasks. Address each one. Save
to:

`/Users/alexanderjmartin/Atlas/jarvis-memory/docs/eval/codex-stage5-review.md`

Use the format prescribed in the briefing §6. The most important sections
are:

1. **Risks I identified** — Catalyst will fix these. Don't pull punches.
2. **Alternatives worth considering** — if there's a higher-ROI path that
   isn't in the plan, name it. We're not married to the current approach.
3. **The hard question** — if you'd start fresh, what would you do? This
   is where we need your independent judgment most.

## Step 5 — Honest-tone enforcement

Catalyst (the plan author) has self-disclosed an optimism bias on lift
estimates. Apply heavy skepticism to any "this will fix X questions"
claim. Specifically:

- The deep-dive bucketing (32 still-wrongs into Buckets A/B/C/D/E):
  verify the categorization is real, not motivated reasoning. Pull a
  few questions from each bucket and check the diagnostic against the
  bucket assignment.
- The "0/32 are pure generation failures" claim: scrutinize. Is there
  a definition of "generation failure" that would change this number?
- Stage-by-stage projected lifts vs landed lifts: the briefing admits
  Stage 4A landed +0 net despite optimistic projections. How much of
  Stage 5's projected ~16-question lift is real vs hopeful?

## Step 6 — Don't be polite

Alex's house style is "tell me when I'm wrong; sycophancy is a failure
mode." Apply this to Catalyst's plan. Polite reviews that don't change
the plan are wasted compute. We need to know where the plan is brittle.

If you think the plan is fundamentally sound and only needs minor edits,
that's a fine answer — but make it explicit so we don't second-guess.

---

## Constraints on your review

1. **No model swap recommendations** unless you have a strong belief that
   gpt-4.1 cannot clear 95.4% no matter what. We've committed to gpt-4.1
   as the headline.

2. **No production-code changes** — we want adapter-only patches in
   `scripts/longmemeval/` and `scripts/run_longmemeval.py`. Production
   `jarvis_memory/` code stays untouched.

3. **Don't recommend Stage 6 (graph rewrite)** unless you're convinced
   none of Stage 5 will work. Stage 6 is a known fallback option.

4. **Total Stage 5 budget is ~$36 in API calls + ~4 hours of engineering
   time.** Recommendations must fit this envelope or explicitly justify
   exceeding it.

5. **The validation harness is the targeted 74+30 = 104-question
   methodology**, with full 500q only fired once after Stage 5 lands.
   Don't recommend more frequent full-500q runs unless you can justify
   the cost.

---

## What success looks like

A review that, when Catalyst reads it, causes Catalyst to either:

(a) Ship Stage 5 with confidence because it's been pressure-tested, OR
(b) Modify Stage 5 in specific ways before shipping (e.g., "drop 5G,
    add Stage 5C in this revised form, adjust 5A's cap to 25 not 30 to
    avoid distraction risk")
(c) Abandon Stage 5 in favor of a different approach you'll specify.

Anything else is a wasted review. Be specific. Cite line numbers and
filenames. Disagree if you disagree.

Begin.
