# LongMemEval v1.1 — Pre-Registered Benchmark Protocol

**Date pre-registered:** 2026-04-26
**System under test:** jarvis-memory v1.1 (commit hash captured at run time below)
**Branch:** `feat/longmemeval-runner`
**Authors:** Catalyst Claude Code instance (Opus 4.7)
**Status:** ⚠ COMMITTED BEFORE ANY ADAPTER CODE LANDS — this is the contract.

---

## Why pre-register

LongMemEval is a 500-question public benchmark with a one-shot grading script. If we run, see failures, tweak the adapter, and re-run on the same questions, the final number is overfit to the test set — exactly the failure mode that turned the MemPalace 100% claim into a credibility disaster.

This document commits to the methodology **before** any adapter code is written. The git log of this repo, with this file's commit timestamp predating every commit in `scripts/run_longmemeval.py` and `scripts/longmemeval/*`, is the receipt that proves we didn't tune to failures.

When we publish a number, this doc is linked from the report. Anyone re-running the benchmark from this commit gets the same setup we did.

---

## Decisions locked (do not reopen during the run)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Test set:** `data/longmemeval/longmemeval_s_cleaned.json` (the standard 500-question benchmark). NOT `longmemeval_oracle.json`. | Oracle is "easy mode" — only relevant sessions visible. Using oracle as the test set is dishonest marketing. Oracle is for the JUDGE only (it has the ground-truth answers). |
| D2 | **Judge:** `xiaowu0162/LongMemEval` `src/evaluation/evaluate_qa.py` with `gpt-4o-2024-08-06`, exactly as published. | Same judge as Zep (71%), Mem0 (93%), OMEGA (95.4%), Mastra (84.2%/94.9%). Without the official judge, scores aren't comparable. |
| D3 | **Answer models — three runs:** (a) Claude Opus 4.7 (`claude-opus-4-7`); (b) gpt-4o (`gpt-4o-2024-08-06`); (c) gpt-4.1 (`gpt-4.1`). Publish all three numbers. | OMEGA's headline 95.4% used gpt-4.1, not gpt-4o (their script default) — so we run gpt-4.1 for an apples-to-apples vs OMEGA. We run gpt-4o for apples-to-apples vs Mastra and the broader literature. We run Opus for the strongest stack. Three numbers > one. |
| D4 | **Run all 500.** No subsampling, no cherry-picking categories. | Anything else is benchmark engineering. |
| D5 | **Single-shot per question.** No best-of-N, no retry-with-different-prompt, no self-consistency voting. | Test-time compute games aren't generalizable; the LLM gets one chance per question. |
| D6 | **Question classification: build a regex/heuristic classifier in `scripts/longmemeval/classifier.py`. Do NOT read `question_type` from the dataset.** | OMEGA reads `question_type` directly (their cheat — won't work in production). We build a classifier and report classifier accuracy on the 500-question oracle as a separate diagnostic. The published number is the one with our classifier. |
| D7 | **No test-set engineering after first run.** The first complete run is the published number. We do NOT inspect failures and patch the adapter to fix them, then re-run. | The MemPalace 100% catastrophe started exactly this way. Anything we want to add must be added BEFORE any 500-question run. |
| D8 | **Bug-fix-only iterations on the validation pass.** During the 10-question validation step, only crashes (NameError, JSON parse errors, etc.) and obvious format errors get fixed. Anything semantic (a category seems weak; a prompt seems off) is left alone — that is test-set engineering. | Same MemPalace risk; pre-registered guard. |
| D9 | **Publish:** adapter code, prompt templates, classifier code, raw model outputs (JSONL), judge outputs (JSONL), per-category scores, and the seed used for any non-determinism. | Reproducibility = credibility. |
| D10 | **Per-question isolation:** unique `group_id = lme_q_<question_id>` and a dedicated Neo4j label `:LMETestEpisode` + dedicated Chroma collection `jarvis_lme_v1`. Cleanup query removes all `:LMETestEpisode` nodes after the run completes. | Zero pollution of production memory; zero cross-question leakage. |

---

## Pre-run additions (the "armor" — locked in BEFORE any 500-question run)

These are the high-ROI changes we apply on top of the v1.1 baseline. They are added to the adapter before the FIRST 500-run, so they cannot be construed as test-set engineering. Lift estimates are honest, grounded in published numbers from cited papers.

| ID | Addition | Source | Expected lift on multi-session category | Expected lift on headline |
|---|---|---|---|---|
| AR1 | **PPR damping α: 0.85 → 0.5** | HippoRAG-2 paper (Gutiérrez et al., ICML 2025). 0.5 spreads activation further, helps multi-hop. | +1–2 pts | +0.2 pts |
| AR2 | **PPR seed broadening:** extract noun phrases (not only proper nouns). For "how often do I exercise", "exercise" must seed PPR. | HippoRAG-2 indexing prompt. | +3–5 pts | +0.5–0.8 pts |
| AR3 | **Counting prompt enumeration:** add to MULTISESSION prompt — "Before answering, output a numbered list of every distinct match. Then count the list." | Chain-of-Note (Yu et al. 2023) + Wei CoT. | +2–4 pts on counting subset | +0.3 pts |
| AR4 | **Adopt OMEGA's exact retrieval recipe:** triple fan-out (primary temporal + secondary expanded + tertiary raw), per-category K floors, recency boost (0.5×) for knowledge-update, anchored temporal-range parsing. Verbatim per `scripts/longmemeval_official.py` lines 957–1103. | OMEGA: replicate the parts of their stack we don't already cover. | n/a (replication of baseline) | —baseline— |
| AR5 | **Adopt OMEGA's 5+1 prompt templates verbatim** (VANILLA, ENHANCED, MULTISESSION, PREFERENCE, TEMPORAL, ABSTENTION). Use the prompt selected by our classifier output. | OMEGA `longmemeval_official.py` lines 143–276. | n/a (replication) | —baseline— |

**Combined honest expected lift on headline:** +1 to +2 pts over a clean OMEGA replication.

**Stretch goals (only if validation comes back clean and time permits BEFORE first 500-run):**
- AR6: Self-Ask query decomposition for multi-session questions (one extra LLM call to split, retrieve per sub-query, RRF merge). +3–6 pts on multi-session.

---

## Targets to beat (publicly published, legit)

| System | Score | Answer model | Judge | Reproducibility |
|---|---|---|---|---|
| OMEGA | 95.4% | gpt-4.1 | gpt-4o-2024-08-06 (homemade, prompts copied from official) | github.com/omega-memory/omega-memory |
| Mastra Observational | 94.87% | gpt-5-mini | gpt-4o-2024-08-06 | mastra.ai/research/observational-memory |
| Mastra Observational | 84.23% | gpt-4o | gpt-4o-2024-08-06 | same |
| Mem0 token-efficient | 93.4% | unspecified | gpt-4o-2024-08-06 | mem0.ai/research |
| Hindsight (Vectorize) | 91.4% | Gemini-3-Pro | non-standard judge | arxiv 2512.12818 |

**Honest expectation for jarvis-memory v1.1 with all AR1–AR5 shipped:** 92–96% with gpt-4o, 94–97% with gpt-4.1 / Opus 4.7. Beating 95.4% with gpt-4.1 is reachable; we have structural advantages (HippoRAG-style PPR, bi-temporal edges) on the multi-session category where OMEGA is weakest.

---

## Step-by-step run procedure

### Step 1 — Build, then validate on 10 questions ($1 budget)
1. Adapter: `scripts/run_longmemeval.py` (with classifier + prompts in `scripts/longmemeval/`).
2. Smoke-test classifier on all 500 oracle questions: report accuracy vs ground-truth `question_type`. Target ≥85%.
3. Run on stratified 10-question subset (1–2 per category, mixed difficulty).
4. Eyeball outputs for:
   - Crash-free completion.
   - Hypothesis is a string (not None, not a JSON dump).
   - Retrieval pulls some sessions in `answer_session_ids` (oracle reference, ONLY for diagnosis — never fed to generation).
5. Fix any crashes/format errors. **Do not iterate on semantic quality.**

### Step 2 — Full 500-question run, three answerers (~$60–150, ~12hr wall time)
Run sequentially in the background:
```bash
JARVIS_LME_ANSWERER=opus     python scripts/run_longmemeval.py --output runs/lme_opus_v1.1.jsonl
JARVIS_LME_ANSWERER=gpt4o    python scripts/run_longmemeval.py --output runs/lme_gpt4o_v1.1.jsonl
JARVIS_LME_ANSWERER=gpt41    python scripts/run_longmemeval.py --output runs/lme_gpt41_v1.1.jsonl
```
Each writes append-only JSONL; resume on crash skips already-answered question_ids.

### Step 3 — Run the official judge (~$15–25 each)
```bash
cd /tmp/lme-official
pip install -r requirements-lite.txt
for f in lme_opus_v1.1 lme_gpt4o_v1.1 lme_gpt41_v1.1; do
  python src/evaluation/evaluate_qa.py gpt-4o-2024-08-06 \
    ~/Atlas/jarvis-memory/runs/${f}.jsonl \
    ~/Atlas/jarvis-memory/data/longmemeval/longmemeval_oracle.json
done
```

### Step 4 — Aggregate + publish
- `python /tmp/lme-official/src/evaluation/print_qa_metrics.py runs/<file>.eval-results-gpt-4o-2024-08-06`
- Per-category scores + overall.
- Write `docs/eval/2026-04-2X-longmemeval-v1.1.md` with: headline numbers (3 answerers), per-category breakdown, comparison table, reproduction one-liner, classifier accuracy gap, link to commit hash, raw JSONLs, judge outputs.

### Step 5 — Cleanup
```bash
.venv/bin/python -c "
from neo4j import GraphDatabase
import os
d = GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
with d.session() as s:
    s.run('MATCH (n:LMETestEpisode) DETACH DELETE n')
d.close()
"
```
Plus delete the `jarvis_lme_v1` Chroma collection. Plus rotate `OPENAI_API_KEY` at platform.openai.com (it's been in chat).

---

## Honesty floor (what we'll publish even if it's lower than we want)

If our classifier is materially worse than reading `question_type`, our published headline number is the one with the classifier — even if oracle-category routing scores higher. We report both as a credibility marker.

If we score below OMEGA's 95.4%, we publish the lower number. We don't re-run. We don't tune. We report what we got.

If a category result is suspiciously high (>98%), we audit it for accidental leakage (oracle reference accidentally fed to generation, dataset contamination via training data, etc.) before publishing.

---

## Sign-off

**Pre-registered:** 2026-04-26 by Catalyst Claude Code instance.
**First adapter commit MUST come AFTER this doc commits.** Anyone re-running the benchmark from a future commit hash should be able to verify, via git log, that this doc predates all adapter code.

The commit message for this file is the receipt: `docs(eval): pre-register LongMemEval v1.1 protocol`.
