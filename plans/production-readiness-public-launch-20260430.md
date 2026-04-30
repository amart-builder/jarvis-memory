# Jarvis Memory Public Launch Readiness Plan - 2026-04-30

## Goal

Make Jarvis Memory credible enough to share publicly with the 97.6% LongMemEval result, while keeping the public repo honest: the production branch should ship the real memory system Alex uses, not benchmark-only answer rules.

## Branch Strategy

- Keep `codex/lme-score-lab-scaffolds` frozen as benchmark proof.
- Build public readiness on `codex/production-readiness`, branched from `main`.
- Port only production-worthy changes from the score branch.
- Copy the benchmark proof docs, but do not copy LongMemEval answer scaffolds into product behavior.

## Audit Findings

### The Context Graph Question

Jarvis does have a context graph:

- `:Page` nodes represent entities/concepts/projects.
- Episodes link to Pages through `EVIDENCED_BY` timeline edges.
- Deterministic typed extraction creates Page-to-Page relationships such as `WORKS_AT`, `FOUNDED`, `DECIDED_ON`, `MENTIONS`, and `REFERS_TO`.
- `scored_search` includes a Personalized PageRank channel for multi-hop queries, so the graph is used during retrieval when the query looks relational.

The accurate caveat is narrower: Page `compiled_truth` summaries are not yet richly auto-authored for every Page, and graph extraction is conservative. The graph exists and is used, but the public docs should describe it as an entity/timeline graph, not as magic automatic human-level knowledge modeling.

### Score-Branch Changes To Keep

- Reranker guardrails: cap rerank candidates, truncate long docs, allow `JARVIS_RERANK_DEVICE=cpu`.
- Compaction guardrails: `JARVIS_SEMANTIC_DEDUP=0` and `JARVIS_SEMANTIC_DEDUP_TIMEOUT` so daily compaction cannot burn CPU indefinitely.
- API guardrails: uvicorn keepalive and concurrency limits.
- Stop-hook guardrail: never auto-write canonical `STATUS.md`; only write an explicit fallback path.
- Benchmark proof docs: summary + hashes + readable proof.

### Score-Branch Changes To Leave Behind

- LongMemEval prompt templates, targeted validators, answer scaffolds, evidence ledgers, oracle-category routing, and benchmark-only extraction caches.
- These improved the benchmark run but are not the normal chief-of-staff memory product. They can live in a future `benchmarks/` package if we want reproducible eval tooling, but they should not be presented as production behavior.

## Documentation Work

1. Rewrite the README around the public promise:
   - Persistent memory for agents.
   - Real graph + vector + keyword retrieval.
   - MCP and REST surfaces.
   - 97.6% LongMemEval proof with a caveat that LongMemEval is a signal, not the whole definition of agent memory.
2. Add an agent-facing install runbook:
   - The user gives the repo to Claude Code, Codex, OpenClaw, or another agent.
   - The agent audits the user's machine, agent surfaces, Neo4j situation, security posture, and desired integrations.
   - The agent decides the right install topology instead of blindly prescribing Alex's setup.
3. Fix public docs that still leak Alex-specific assumptions:
   - Replace Alex's internal group IDs with examples and a "choose your own scopes" model.
   - Update MCP count from stale 27 references to the current 29-tool surface.
   - Fix REST smoke examples to use POST `/api/v2/scored_search` and `episode_type`.
   - Explain API bearer-token security for non-loopback deployments.

## Verification

- Run focused tests for changed operational code:
  - `python -m pytest tests/search/test_rerank.py tests/test_compaction.py`
- Run install/docs smoke:
  - `bash scripts/verify_install.sh` if local `.env`/venv supports it.
  - At minimum, run import and MCP parity tests.
- Review staged diff for public tone, no secrets, no hardcoded Alex-only paths in install docs.
