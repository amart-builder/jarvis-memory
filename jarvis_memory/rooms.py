"""Room detection and hall mapping for structured memory metadata.

Rooms are topic-level categories (e.g., "auth", "frontend", "infrastructure").
Halls are memory-type categories (e.g., "decisions", "milestones", "context").

Together with wing (= group_id), these form the metadata triad that
enables retrieval filtering: 60.9% → 94.8% R@10 based on MemPalace benchmarks.
"""
from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Room keyword patterns (70+ domains) ──────────────────────────────

ROOM_KEYWORDS: dict[str, list[str]] = {
    # Engineering domains
    "auth": [
        "auth", "login", "logout", "oauth", "jwt", "token", "session",
        "password", "credential", "sso", "saml", "clerk", "auth0",
        "permission", "rbac", "role", "acl",
    ],
    "frontend": [
        "react", "vue", "angular", "svelte", "css", "html", "tailwind",
        "component", "ui", "ux", "layout", "responsive", "dom", "jsx",
        "tsx", "styled", "theme", "animation", "modal", "button", "form",
    ],
    "backend": [
        "api", "endpoint", "rest", "graphql", "server", "route", "handler",
        "middleware", "controller", "express", "fastapi", "flask", "django",
        "nest", "hono", "trpc",
    ],
    "database": [
        "database", "sql", "postgres", "mysql", "mongo", "redis", "neo4j",
        "migration", "schema", "query", "index", "table", "column", "orm",
        "prisma", "drizzle", "supabase", "sqlite",
    ],
    "infrastructure": [
        "deploy", "ci", "cd", "docker", "kubernetes", "k8s", "terraform",
        "aws", "gcp", "azure", "vercel", "netlify", "cloudflare", "nginx",
        "ssl", "dns", "domain", "cdn", "load balancer", "scaling",
    ],
    "testing": [
        "test", "spec", "jest", "pytest", "vitest", "cypress", "playwright",
        "e2e", "unit test", "integration test", "coverage", "mock", "stub",
        "fixture", "assertion",
    ],
    "devops": [
        "monitoring", "logging", "alert", "grafana", "datadog", "sentry",
        "prometheus", "metrics", "uptime", "incident", "pagerduty", "oncall",
        "observability",
    ],
    "ai": [
        "llm", "gpt", "claude", "openai", "anthropic", "embedding",
        "vector", "rag", "prompt", "fine-tune", "model", "inference",
        "token", "context window", "agent", "mcp", "tool use",
    ],
    "payments": [
        "stripe", "payment", "billing", "invoice", "subscription",
        "checkout", "refund", "charge", "price", "plan", "revenue",
    ],
    "email": [
        "email", "smtp", "sendgrid", "mailgun", "newsletter", "template",
        "notification", "inbox", "unsubscribe",
    ],
    "mobile": [
        "ios", "android", "react native", "flutter", "swift", "kotlin",
        "mobile", "app store", "push notification",
    ],
    "security": [
        "security", "vulnerability", "xss", "csrf", "injection", "encrypt",
        "hash", "salt", "audit", "compliance", "gdpr", "soc2",
    ],
    "performance": [
        "performance", "latency", "throughput", "cache", "optimization",
        "profiling", "benchmark", "slow", "bottleneck", "memory leak",
    ],
    "documentation": [
        "docs", "readme", "guide", "tutorial", "onboarding", "api docs",
        "changelog", "wiki", "specification",
    ],
    "design": [
        "figma", "sketch", "wireframe", "prototype", "mockup", "brand",
        "typography", "color", "icon", "illustration", "design system",
    ],
    # Business domains
    "finance": [
        "spv", "fund", "investor", "capital", "carry", "management fee",
        "portfolio", "deal", "term sheet", "valuation", "equity",
        "quickbooks", "accounting", "budget",
    ],
    "content": [
        "blog", "article", "post", "video", "podcast", "twitter",
        "linkedin", "social media", "newsletter", "content strategy",
    ],
    "legal": [
        "legal", "contract", "terms", "privacy policy", "nda",
        "compliance", "regulation", "trademark", "ip",
    ],
    "hiring": [
        "hiring", "interview", "candidate", "job", "offer", "recruiter",
        "resume", "onboard", "team",
    ],
    "product": [
        "product", "feature", "roadmap", "sprint", "backlog", "user story",
        "stakeholder", "launch", "beta", "feedback", "analytics",
    ],
    # Personal domains
    "health": [
        "health", "exercise", "sleep", "diet", "meditation", "workout",
        "gym", "running", "walking", "nutrition", "mental health",
    ],
    "personal": [
        "personal", "life", "family", "friend", "hobby", "travel",
        "vacation", "birthday", "anniversary",
    ],
}

# ── Hall mapping (21 memory types → 5 halls) ────────────────────────

HALL_MAP: dict[str, str] = {
    # decisions hall
    "decision": "decisions",
    "preference": "decisions",
    "commitment": "decisions",
    "cancellation": "decisions",
    # plans hall
    "intention": "plans",
    "plan": "plans",
    "goal": "plans",
    "constraint": "plans",
    # milestones hall
    "action": "milestones",
    "outcome": "milestones",
    "observation": "milestones",
    "correction": "milestones",
    "event": "milestones",
    # problems hall
    "question": "problems",
    "hypothesis": "problems",
    # context hall (default)
    "fact": "context",
    "procedure": "context",
    "relationship": "context",
    "insight": "context",
    "answer": "context",
    "meta": "context",
}


def detect_room(text: str, group_id: Optional[str] = None) -> str:
    """Detect the semantic room for a piece of text using keyword scoring.

    Scores text against all room keyword lists and returns the
    highest-scoring room. Returns "general" if no strong match.

    Args:
        text: Memory content to classify.
        group_id: Optional project group (unused currently, reserved for
                  project-specific room overrides).

    Returns:
        Room name string (e.g., "auth", "frontend", "infrastructure").
    """
    if not text:
        return "general"

    text_lower = text.lower()
    scores: dict[str, int] = {}

    for room, keywords in ROOM_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # Count occurrences, not just presence
            count = text_lower.count(kw)
            if count > 0:
                score += count
                # Bonus for longer/more specific keywords
                if len(kw) > 6:
                    score += count

        if score > 0:
            scores[room] = score

    preview = text[:80].replace("\n", " ")
    if not scores:
        # No keyword matched any room. If the text is substantive, this is a
        # candidate for room-keyword expansion.
        if len(text) >= 30:
            logger.info(
                "room_fallback: no_match group=%s len=%d preview=%r",
                group_id, len(text), preview,
            )
        return "general"

    # Return highest-scoring room
    best = max(scores, key=scores.get)

    # Require minimum score of 2 to avoid false positives on single-word matches
    if scores[best] < 2:
        if len(text) >= 30:
            logger.info(
                "room_fallback: weak_match group=%s len=%d best=%s score=%d preview=%r",
                group_id, len(text), best, scores[best], preview,
            )
        return "general"

    return best


def get_hall(memory_type: str) -> str:
    """Map a memory type to its hall.

    Args:
        memory_type: One of the 21 memory types from classifier.py.

    Returns:
        Hall name: "decisions", "plans", "milestones", "problems", or "context".
    """
    return HALL_MAP.get(memory_type, "context")
