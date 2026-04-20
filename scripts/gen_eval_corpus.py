#!/usr/bin/env python
"""Generate a synthetic eval corpus (episodes + queries + qrels) via Claude Opus.

Spec: brain/projects/jarvis-memory/plans/runs/2026-04-20-eval-harness-and-routing/spec.md

Produces ~150 synthetic jarvis-memory episodes, ~50 queries, and ~50 qrel
rows mapping each query to 1-5 relevant episode uuids. Writes JSONL to:

    tests/eval_data/synthetic_corpus_v1.jsonl
    tests/eval_data/synthetic_queries_v1.jsonl
    tests/eval_data/synthetic_qrels_v1.jsonl

Reproducibility
---------------
- Python ``random.seed(SEED)`` (default 20260420) fixes uuid generation and
  group/type/date sampling.
- Anthropic calls use ``temperature=0.3`` and fixed prompts per topic; the
  LLM side isn't byte-reproducible, but the COMMITTED corpus is — the
  on-disk JSONL files are the ground truth for the eval harness.

Usage
-----
    source ~/Atlas/jarvis-memory/.venv/bin/activate
    export ANTHROPIC_API_KEY=...    # required
    python scripts/gen_eval_corpus.py                 # default settings
    python scripts/gen_eval_corpus.py --episodes 150 --queries 50
    python scripts/gen_eval_corpus.py --template-only # no API, deterministic

``--template-only`` builds a stand-in corpus from a hand-curated template
bank — useful when the API is flaky or for CI smoke tests. The committed
corpus is still the canonical one.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import uuid as uuidlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("gen_eval_corpus")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests" / "eval_data"

SEED = 20260420

# Canonical project group_ids for jarvis-memory (matches brain/MEMORY_PROTOCOL §1).
GROUP_IDS = ["system", "navi", "foundry", "catalyst", "combinator"]

# jarvis-memory episode_types used in the wild + their tag prefixes.
# The tag (e.g. "[MEETING]") is human-facing; the stored ``episode_type`` must
# be a valid key in ``jarvis_memory.classifier.MEMORY_TYPES``.
EPISODE_TYPES: list[tuple[str, str]] = [
    ("decision", "[DECISION]"),
    ("plan", "[PLAN]"),
    ("fact", "[FACT]"),
    ("correction", "[CORRECTION]"),
    ("event", "[MEETING]"),
    ("outcome", "[COMPLETION]"),
    ("meta", "[HANDOFF]"),
]

# Common LLM aliases → canonical MEMORY_TYPES keys. Applied after the model
# response so the on-disk corpus never ships unknown types.
TYPE_ALIAS: dict[str, str] = {
    "meeting": "event",
    "handoff": "meta",
    "completion": "outcome",
}

OPUS_MODEL = "claude-opus-4-6"  # Opus model that still accepts `temperature`.
# Note: claude-opus-4-7 deprecated the temperature param; we keep 4-6 so the
# documented `temperature=0.3` seed-equivalent still applies.


# ──────────────────────────────────────────────────────────────────────
# Topic bank — each topic maps to a realistic project context. The LLM
# (or template fallback) riffs on these to produce prose episodes.
# ──────────────────────────────────────────────────────────────────────

TOPICS: list[dict[str, Any]] = [
    {
        "id": "navi-auth",
        "group_id": "navi",
        "subject": "Navi auth stack",
        "hint": (
            "Navi (SPV investor portal). Stack: Next.js + Clerk for auth. "
            "Common decisions: Clerk vs Auth0 vs NextAuth, magic-link vs "
            "password, session TTLs, SOC2 audit prep."
        ),
    },
    {
        "id": "navi-stripe",
        "group_id": "navi",
        "subject": "Navi payments + carry distribution",
        "hint": (
            "Stripe Connect for LP contributions and GP carry. Decisions "
            "around test-mode webhooks, idempotency keys, and K-1 reports."
        ),
    },
    {
        "id": "navi-deals",
        "group_id": "navi",
        "subject": "Navi deal flow pipeline",
        "hint": (
            "Postgres schema for deals, term sheets, and LP allocation. "
            "Migration strategy, constraint updates, importance of valid_to."
        ),
    },
    {
        "id": "foundry-lp-portal",
        "group_id": "foundry",
        "subject": "Foundry LP portal relaunch",
        "hint": (
            "Marketing+product site for Foundry. Next.js + Sanity CMS. "
            "Rebrand, mobile-first, investor-facing dashboards."
        ),
    },
    {
        "id": "foundry-email",
        "group_id": "foundry",
        "subject": "Foundry newsletter stack",
        "hint": (
            "SendGrid integration, weekly macro briefing, subscriber "
            "segmentation, unsubscribe flow, DKIM/SPF setup."
        ),
    },
    {
        "id": "catalyst-pipeline",
        "group_id": "catalyst",
        "subject": "Catalyst data pipeline",
        "hint": (
            "ETL pipeline ingesting market data into Postgres. "
            "Backfill strategy, dedup, Airflow DAG design."
        ),
    },
    {
        "id": "catalyst-llm",
        "group_id": "catalyst",
        "subject": "Catalyst LLM research agent",
        "hint": (
            "Agent that reads earnings transcripts and emits summary cards. "
            "Claude Sonnet vs Opus trade-offs, RAG over Neo4j."
        ),
    },
    {
        "id": "combinator-mvp",
        "group_id": "combinator",
        "subject": "Combinator MVP",
        "hint": (
            "Combinator: a tool that mixes strategy. Decisions about UI "
            "framework, hosting, and beta-tester cohort."
        ),
    },
    {
        "id": "system-memory",
        "group_id": "system",
        "subject": "jarvis-memory architecture",
        "hint": (
            "Neo4j + ChromaDB backend for shared Claude+OpenClaw memory. "
            "Composite scoring, 21-type classifier, room/hall/wing metadata, "
            "compaction cron."
        ),
    },
    {
        "id": "system-openclaw",
        "group_id": "system",
        "subject": "OpenClaw on Mac Mini",
        "hint": (
            "Always-on agent. MCP tool surface, session handoff, Tailscale, "
            "launchd crons, SOUL.md bootstrap."
        ),
    },
]


# ──────────────────────────────────────────────────────────────────────
# Template fallback — used when --template-only or when API fails.
# Deterministic, fully reproducible from SEED alone.
# ──────────────────────────────────────────────────────────────────────

TEMPLATE_SEEDS: list[tuple[str, str]] = [
    # (episode_type, body_template)
    ("decision", "Chose {choice_a} over {choice_b} for {subject}. WHY: {reason}. IMPACT: {impact}."),
    ("decision", "Moved {subject} from {old} to {new}. Rationale: {reason}. Deadline: end of week."),
    ("plan", "Next step for {subject}: build {artifact} by {when}. Acceptance: {accept}."),
    ("plan", "Roadmap for {subject}: {phase_a}, then {phase_b}, finally {phase_c}. Owner: Alex."),
    ("fact", "{subject} uses {tool}. {extra_fact}. Contact point: {contact}."),
    ("fact", "As of today, {subject} has {metric}. Last review: 2026-03-15."),
    ("correction", "Correction: {subject} is actually {truth}, not {myth}. Source: {source}."),
    ("correction", "Previous note said {myth} — updated to {truth} after {source} review."),
    ("event", "Meeting with {person} on {subject}. Key discussion: {discussion}. Next step: {next_step}."),
    ("event", "Deployed {subject} to production at {time}. Issue: {issue}. Resolution: {resolution}."),
    ("outcome", "Completed {artifact} for {subject}. Result: {result}. Blocker resolved: {blocker}."),
    ("outcome", "Shipped {subject} v{version}. Users can now {capability}. Follow-up: {followup}."),
    ("meta", "Handing off {subject} to OpenClaw. Status: {status}. Open question: {question}."),
    ("meta", "Session handoff: {subject} is at {state}. Resume with: {resume}."),
]

TEMPLATE_FILLS: dict[str, list[str]] = {
    "choice_a": ["Clerk", "Supabase", "Postgres", "Next.js", "Tailwind", "Neo4j", "Claude Opus", "SendGrid", "Stripe Connect", "Vercel"],
    "choice_b": ["Auth0", "Firebase", "MongoDB", "Remix", "Bootstrap", "Weaviate", "Claude Sonnet", "Mailgun", "Stripe Standard", "Netlify"],
    "old": ["Auth0", "a custom Postgres schema", "the legacy REST API", "a bash cron", "a Google Sheet"],
    "new": ["Clerk", "Prisma + Postgres", "the GraphQL gateway", "launchd + Python", "a Neo4j graph"],
    "subject": ["the SPV deal pipeline", "the LP carry report", "the market-data ETL", "the investor dashboard", "the MCP trust boundary", "the eval harness", "the compaction cron", "the memory router"],
    "reason": ["lower maintenance", "better vendor trust", "SOC2-friendly defaults", "fewer moving parts", "native MCP support", "cheaper at scale"],
    "impact": ["unblocks onboarding", "saves ~6h/month of ops", "removes a compliance gap", "doubles retrieval recall", "shrinks cold-start latency"],
    "artifact": ["the signup flow", "a runbook", "the composite-score benchmark", "the onboarding checklist", "the webhook handler"],
    "when": ["Friday", "next Wednesday", "end of sprint", "April 30"],
    "accept": ["all integration tests pass", "p95 latency under 200ms", "the SOC2 checklist is green", "zero 5xx for 24h"],
    "phase_a": ["scaffold the schema", "write the runbook", "collect baseline metrics"],
    "phase_b": ["ship the MVP to 3 LPs", "run a shadow-deploy for 48h", "publish the dashboards"],
    "phase_c": ["review + commit", "cut v1", "hand off to OpenClaw"],
    "tool": ["ChromaDB", "Neo4j over Tailscale", "Stripe Connect", "Sanity CMS", "Vercel Edge"],
    "extra_fact": ["It is authoritative for the spend ledger", "It drives the nightly compaction job", "It backs the scored_search endpoint"],
    "contact": ["alex@edge-fund.io", "the on-call rotation", "the #jarvis-alerts Slack channel"],
    "metric": ["4,281 active episodes", "97ms p95 write latency", "2 open incidents", "12 recent handoffs"],
    "truth": ["the port is 3500", "Clerk is used, not Auth0", "the Mini is the Neo4j host"],
    "myth": ["the port is 3000", "we use Auth0", "Neo4j runs on the MBP"],
    "source": ["a fresh review", "the deploy logs", "the compaction report"],
    "person": ["a prospective LP", "the Stripe rep", "the security auditor", "a Foundry subscriber"],
    "discussion": ["carry mechanics", "the wire flow", "retention after the pilot", "the rebranding timeline"],
    "next_step": ["schedule the follow-up", "draft the MSA", "ship the fix", "update the runbook"],
    "time": ["11:42 UTC", "just after the daily stand-up", "3:07 PT"],
    "issue": ["a brief Stripe webhook replay", "no issue", "a cold-start timeout", "a stale ChromaDB collection"],
    "resolution": ["reran the cron manually", "auto-retried in 30s", "rolled back and patched", "increased the timeout"],
    "result": ["R@5 improved from 0.61 to 0.74", "all pending signups processed", "the rebrand preview is live"],
    "blocker": ["the Tailscale handshake", "the ChromaDB reindex", "the Stripe account verification"],
    "version": ["0.2", "1.0", "2.1", "3.0"],
    "capability": ["download their K-1", "see their carry allocation", "trigger a manual compaction"],
    "followup": ["monitor for a week", "survey the first 5 users", "open a follow-up issue"],
    "status": ["blocked on a webhook", "ready to ship", "needs a review"],
    "question": ["should we idempotency-key the retry?", "is the dedup threshold too aggressive?", "which group_id owns this?"],
    "state": ["75% done", "awaiting review", "post-merge polish"],
    "resume": ["pytest tests/test_api_parity.py", "curl /api/v2/wake_up?group_id=navi", "git pull && pytest"],
}


def _expand_template(rng: random.Random, episode_type: str, template: str, topic: dict[str, Any]) -> str:
    """Fill a template with random choices from TEMPLATE_FILLS."""
    filled = template
    # Topic-seed the subject whenever the template asks for {subject}.
    if "{subject}" in filled:
        filled = filled.replace("{subject}", topic["subject"])
    for key, choices in TEMPLATE_FILLS.items():
        placeholder = "{" + key + "}"
        while placeholder in filled:
            filled = filled.replace(placeholder, rng.choice(choices), 1)
    tag_map = dict(EPISODE_TYPES)
    tag = tag_map.get(episode_type, f"[{episode_type.upper()}]")
    return f"{tag} {filled}"


# ──────────────────────────────────────────────────────────────────────
# LLM path — uses Anthropic SDK. Returns JSON blob with episodes+queries.
# ──────────────────────────────────────────────────────────────────────


def _llm_generate(
    client: Any,
    topic: dict[str, Any],
    rng: random.Random,
    n_episodes: int,
    n_queries: int,
    model: str = OPUS_MODEL,
) -> dict[str, Any]:
    """Ask Claude Opus to generate episodes + queries for one topic.

    Returns a dict:
        {
          "episodes": [{"episode_type": ..., "body": "..."}],
          "queries":  [{"query": "...", "relevant_indices": [0, 3, 7]}]
        }
    """
    prompt = f"""You are generating a small synthetic slice of engineering memories for a test corpus.

Project: **{topic['subject']}** (group_id: {topic['group_id']})
Context: {topic['hint']}

Write exactly {n_episodes} episodes + {n_queries} queries about this topic.

Rules:
- Each episode is 100-400 characters of realistic prose.
- Tag each episode with ONE of: [DECISION], [PLAN], [FACT], [CORRECTION], [MEETING], [COMPLETION], [HANDOFF].
- Mix episode types roughly evenly.
- Use concrete nouns (tools, people names, numbers). Avoid Claude-isms like "I'll help you".
- Queries should be realistic "I need info about X" questions a developer would ask later. 3-10 words each.
- For each query, list 1-5 episode indices (0-based) whose content is actually relevant.

Return STRICT JSON. No prose before/after. Schema:
{{
  "episodes": [
    {{"episode_type": "decision", "body": "[DECISION] Chose Clerk over Auth0 because..."}},
    ...
  ],
  "queries": [
    {{"query": "which auth provider are we using", "relevant_indices": [0, 2]}},
    ...
  ]
}}
"""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        # Newer Opus models (e.g., 4-7) deprecate `temperature`; retry without.
        if "temperature" in str(e).lower() and "deprecated" in str(e).lower():
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise
    text = response.content[0].text.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("LLM returned invalid JSON (%s). First 200 chars: %s", e, text[:200])
        return {"episodes": [], "queries": []}


# ──────────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────────


def _make_uuid(rng: random.Random) -> str:
    # Deterministic UUID from our seeded RNG — uuidlib.uuid4() pulls from
    # os.urandom and isn't seedable.
    return str(uuidlib.UUID(int=rng.getrandbits(128)))


def _backdated_iso(rng: random.Random) -> str:
    days_back = rng.randint(1, 120)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    ts = datetime(2026, 4, 20, tzinfo=timezone.utc) - timedelta(
        days=days_back, hours=hour, minutes=minute
    )
    return ts.isoformat()


def _generate_for_topic(
    topic: dict[str, Any],
    rng: random.Random,
    n_episodes: int,
    n_queries: int,
    use_llm: bool,
    client: Any,
    model: str = OPUS_MODEL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (episodes, queries, qrels) for one topic."""
    episodes: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []

    llm_result: dict[str, Any] = {}
    if use_llm and client is not None:
        try:
            llm_result = _llm_generate(client, topic, rng, n_episodes, n_queries, model=model)
            logger.info(
                "topic=%s LLM produced ep=%d q=%d",
                topic["id"],
                len(llm_result.get("episodes", [])),
                len(llm_result.get("queries", [])),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM call failed for %s: %s (using templates)", topic["id"], e)
            llm_result = {}

    llm_episodes = llm_result.get("episodes") or []
    llm_queries = llm_result.get("queries") or []

    # Fill shortages from templates.
    if len(llm_episodes) < n_episodes:
        remaining = n_episodes - len(llm_episodes)
        for _ in range(remaining):
            ep_type, tmpl = rng.choice(TEMPLATE_SEEDS)
            body = _expand_template(rng, ep_type, tmpl, topic)
            llm_episodes.append({"episode_type": ep_type, "body": body})

    # Cap to exactly n_episodes.
    llm_episodes = llm_episodes[:n_episodes]

    # Assign uuids + metadata.
    for idx, ep in enumerate(llm_episodes):
        ep_type = ep.get("episode_type") or "fact"
        ep_type = TYPE_ALIAS.get(ep_type, ep_type)
        body = ep.get("body") or ""
        # Ensure tag prefix.
        tag_map = dict(EPISODE_TYPES)
        tag = tag_map.get(ep_type, f"[{ep_type.upper()}]")
        if not body.lstrip().startswith("["):
            body = f"{tag} {body}"
        uid = _make_uuid(rng)
        episodes.append(
            {
                "uuid": uid,
                "content": body,
                "group_id": topic["group_id"],
                "episode_type": ep_type,
                "importance": round(rng.uniform(0.55, 0.95), 2),
                "created_at": _backdated_iso(rng),
                "topic_id": topic["id"],
                "topic_index": idx,
            }
        )

    # Queries: from LLM if usable; else template-derived.
    if not llm_queries:
        # Template queries: generate one per (roughly) every 3 episodes.
        n = max(1, n_queries)
        for _ in range(n):
            subject = topic["subject"]
            template_q = rng.choice(
                [
                    f"what did we decide about {subject}",
                    f"current plan for {subject}",
                    f"most recent update on {subject}",
                    f"open questions on {subject}",
                    f"who owns {subject}",
                    f"what broke in {subject}",
                    f"last meeting about {subject}",
                ]
            )
            # Pick 1-3 random relevant episode indices.
            count = min(len(episodes), rng.randint(1, 3))
            rel_idx = rng.sample(range(len(episodes)), count)
            llm_queries.append({"query": template_q, "relevant_indices": rel_idx})

    llm_queries = llm_queries[:n_queries]

    for q_idx, q in enumerate(llm_queries):
        qid = f"{topic['id']}-q{q_idx:02d}"
        # Translate indices → uuids. Clamp to valid range.
        indices = q.get("relevant_indices") or []
        if isinstance(indices, list):
            relevant_uuids = [
                episodes[i]["uuid"]
                for i in indices
                if isinstance(i, int) and 0 <= i < len(episodes)
            ]
        else:
            relevant_uuids = []
        # Require at least 1 relevant uuid — pick one if LLM returned none.
        if not relevant_uuids and episodes:
            relevant_uuids = [rng.choice(episodes)["uuid"]]
        # Cap at 5.
        relevant_uuids = relevant_uuids[:5]
        queries.append(
            {
                "query_id": qid,
                "query": str(q.get("query", "")).strip() or f"info about {topic['subject']}",
                "group_id": topic["group_id"],
                "topic_id": topic["id"],
            }
        )
        qrels.append(
            {
                "query_id": qid,
                "relevant_ids": relevant_uuids,
            }
        )

    return episodes, queries, qrels


def generate_corpus(
    total_episodes: int,
    total_queries: int,
    use_llm: bool,
    client: Any,
    rng: random.Random,
    model: str = OPUS_MODEL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Distribute totals across TOPICS evenly and generate."""
    n_topics = len(TOPICS)
    eps_per_topic = max(1, total_episodes // n_topics)
    qs_per_topic = max(1, total_queries // n_topics)

    # Remainder goes to the first few topics.
    eps_remainder = max(0, total_episodes - eps_per_topic * n_topics)
    qs_remainder = max(0, total_queries - qs_per_topic * n_topics)

    episodes_all: list[dict[str, Any]] = []
    queries_all: list[dict[str, Any]] = []
    qrels_all: list[dict[str, Any]] = []

    for idx, topic in enumerate(TOPICS):
        n_ep = eps_per_topic + (1 if idx < eps_remainder else 0)
        n_q = qs_per_topic + (1 if idx < qs_remainder else 0)
        print(f"[{idx + 1}/{n_topics}] topic={topic['id']} ep={n_ep} q={n_q}", flush=True)
        eps, qs, qr = _generate_for_topic(topic, rng, n_ep, n_q, use_llm, client, model=model)
        episodes_all.extend(eps)
        queries_all.extend(qs)
        qrels_all.extend(qr)

    return episodes_all, queries_all, qrels_all


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=150, help="Total episodes to generate (default 150).")
    parser.add_argument("--queries", type=int, default=50, help="Total queries to generate (default 50).")
    parser.add_argument("--seed", type=int, default=SEED, help=f"RNG seed (default {SEED}).")
    parser.add_argument("--template-only", action="store_true", help="Skip API; use deterministic templates.")
    parser.add_argument("--model", default=OPUS_MODEL, help=f"Anthropic model (default {OPUS_MODEL}).")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    rng = random.Random(args.seed)

    client: Any = None
    use_llm = not args.template_only
    if use_llm:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            print(
                "ANTHROPIC_API_KEY not set. Either export it, or pass --template-only.",
                file=sys.stderr,
            )
            return 2
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:  # noqa: BLE001
            print(f"anthropic init failed: {e}. Falling back to --template-only.", file=sys.stderr)
            use_llm = False

    out_dir = Path(args.out_dir)
    episodes, queries, qrels = generate_corpus(
        total_episodes=args.episodes,
        total_queries=args.queries,
        use_llm=use_llm,
        client=client,
        rng=rng,
        model=args.model,
    )

    _write_jsonl(out_dir / "synthetic_corpus_v1.jsonl", episodes)
    _write_jsonl(out_dir / "synthetic_queries_v1.jsonl", queries)
    _write_jsonl(out_dir / "synthetic_qrels_v1.jsonl", qrels)

    print(
        f"\nWrote {len(episodes)} episodes, {len(queries)} queries, "
        f"{len(qrels)} qrels to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
