# Parallel LongMemEval Validation Build Plan

## Goal

Cut targeted LongMemEval validation wall-clock time by running adapter shards in parallel without changing retrieval, prompt assembly, generation settings, or judge behavior.

## Success Criteria

- Parallel adapter writes the same canonical qid set as serial targeted validation.
- Each worker uses isolated mutable state:
  - unique Neo4j label
  - unique Chroma persistent path
  - unique Chroma collection
  - unique group-id prefix
  - unique output/log files
- Coordinator refuses to merge if any qid is missing, duplicated, or unexpected.
- Optional parity gate compares serial vs parallel prompt hashes and normalized ordered hit ids.
- Existing targeted-validation flow still works unchanged.
- Unit tests cover sharding, merge validation, isolation spec generation, and parity comparison.

## Implementation

1. Add environment-controlled isolation hooks to `scripts/run_longmemeval.py`:
   - `JARVIS_LME_NEO4J_LABEL`
   - `JARVIS_LME_CHROMA_PATH`
   - `JARVIS_LME_CHROMA_COLLECTION`
   - `JARVIS_LME_GROUP_PREFIX`
2. Add `scripts/run_parallel_targeted_validation.py` as the process-level coordinator.
3. Add unit tests in `tests/longmemeval/test_parallel_targeted_validation.py`.
4. Verify with pytest and a `--dry-run` command before any live API run.

## Operational Gate

Before using this for score-bearing validation, run a 30-40 question parity sample:

1. Serial targeted run with `--diagnostics`.
2. Parallel targeted run with `--diagnostics --parity-against <serial-jsonl>`.
3. Accept parallel mode only if parity passes.

If prompt hashes or normalized hit order differ, do not use adapter parallelism for score-bearing runs.
