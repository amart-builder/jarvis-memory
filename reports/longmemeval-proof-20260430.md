# LongMemEval Proof - 2026-04-30

## Score

**Jarvis Memory scored 488 / 500 = 97.60% raw correct** on the checkpointed full LongMemEval run.

Run ID: `phase12_full500_chk_20260430-0419`

## Method

- Dataset: LongMemEval-S, 500 questions.
- Answerer: GPT-4.1.
- Judge: GPT-4o.
- Run shape: five checkpointed 100-question chunks, then merged into one canonical 500-row artifact.
- Scoring basis: raw correct out of 500 judged rows.
- Code anchor before proof commit: `38c8baf` (`Fix total-line append for amount questions`) on branch `codex/lme-score-lab-scaffolds`.

## Result

| Chunk | Correct | Total |
| --- | ---: | ---: |
| 1 | 97 | 100 |
| 2 | 97 | 100 |
| 3 | 99 | 100 |
| 4 | 98 | 100 |
| 5 | 97 | 100 |
| **Total** | **488** | **500** |

Accuracy: **97.60%**

## Misses

| Category | Correct | Total | Misses |
| --- | ---: | ---: | ---: |
| single-session-user | 69 | 70 | 1 |
| multi-session | 128 | 133 | 5 |
| single-session-preference | 29 | 30 | 1 |
| temporal-reasoning | 131 | 133 | 2 |
| knowledge-update | 77 | 78 | 1 |
| single-session-assistant | 54 | 56 | 2 |
| **Total** | **488** | **500** | **12** |

Missed question IDs:

`4100d0a0`, `88432d0a`, `d23cf73b`, `b6025781`, `e6041065`, `7405e8b1`, `a96c20ee_abs`, `0bc8ad93`, `gpt4_2f56ae70`, `07741c45`, `4baee567`, `eaca4986`

## Proof Artifacts

Committed proof files:

- `reports/longmemeval-proof-20260430.md`
- `reports/proof/phase12_full500_chk_20260430-0419_merged500.summary.json`
- `reports/proof/phase12_full500_chk_20260430-0419.hashes.txt`
- `reports/longmemeval-raw-score-chart-20260430.html`

Local raw artifacts, intentionally not committed because they are large generated outputs:

- `runs/phase12_full500_chk_20260430-0419_merged500.jsonl`
- `runs/phase12_full500_chk_20260430-0419_merged500.jsonl.eval-results-gpt-4o`
- `runs/phase12_full500_chk_20260430-0419_merged500.summary.json`

The hash file records SHA-256 hashes for the raw answer file, raw judge file, original summary, committed summary copy, and leaderboard HTML.

## Verification Checks

Verified locally:

- Merged answer artifact has exactly 500 rows.
- Merged judge artifact has exactly 500 rows.
- Summary reports `right = 488`, `wrong = 12`, `accuracy = 0.976`, `rows = 500`.
- The committed summary copy has the same SHA-256 hash as the original summary.

## Caveat

This proof pack documents the completed checkpointed full-500 run and its merged judge output. It does not claim a second redundant re-judge of the same merged 500 rows.
