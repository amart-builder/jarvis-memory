"""Query intent classification — rule-based, no LLM on the hot path.

Returns one of four intents so :func:`jarvis_memory.search.scored_search`
can choose an appropriate retriever mix:

* ``"entity"`` — query mentions a proper noun (a person, company, project).
  Favor Page.compiled_truth matches + graph traversal.
* ``"temporal"`` — query contains an explicit recency phrase
  (``"last week"``, ``"yesterday"``, date literal, ``"since X"``).
  Favor recency-ordered retrieval.
* ``"event"`` — query asks about a meeting/decision/handoff.
  Favor episode types tagged as such.
* ``"general"`` — catch-all. Plain semantic + keyword RRF.

Design notes
------------
* **Rule-based v1.** Spec §"Assumptions" A3 — ship regex/keyword rules,
  measure, decide whether an LLM v2 is worth the latency.
* **No LLM calls.** This function is called on every ``scored_search``
  request; it must be microseconds, not network round-trips.
* **Confidence not returned.** Router is a one-of-four switch. If you
  need confidence later, return it as a dataclass — the public API
  returns just the label here to keep callers simple.
"""
from __future__ import annotations

import re
from typing import Literal

__all__ = ["classify", "Intent"]

Intent = Literal["entity", "temporal", "event", "general"]

# Temporal phrase list. Matched case-insensitively against the query.
# Includes single words (``yesterday``), short phrases (``last week``),
# and explicit date-range words (``since``, ``before``, ``after`` when
# followed by a token — checked via separate regex below).
_TEMPORAL_PHRASES: tuple[str, ...] = (
    "yesterday",
    "today",
    "tomorrow",
    "last week",
    "last month",
    "last year",
    "this week",
    "this month",
    "this year",
    "past week",
    "past month",
    "past year",
    "past hour",
    "past day",
    "past days",
    "recent",
    "recently",
    "ago",
    "earlier",
)

# "since 2024-01-01", "since last month", "before jan", "after friday",
# "before 2025-01-01", "from 2024 to 2025"
_TEMPORAL_PREP_PATTERN = re.compile(
    r"\b(since|before|after|from|until|between)\b",
    re.IGNORECASE,
)

# ISO or slashed dates: 2025-04-20, 04/20/2025, 2025/04/20.
_DATE_PATTERN = re.compile(
    r"\b("
    r"\d{4}-\d{1,2}-\d{1,2}"  # 2025-04-20
    r"|\d{1,2}/\d{1,2}/\d{2,4}"  # 04/20/2025
    r"|\d{4}/\d{1,2}/\d{1,2}"  # 2025/04/20
    r")\b"
)

# Event keywords. Present-tense or past-tense — we just want to flag
# the *topic*, not the grammar.
_EVENT_WORDS: frozenset[str] = frozenset(
    {
        "meeting",
        "meetings",
        "met",
        "decision",
        "decisions",
        "decided",
        "handoff",
        "handoffs",
        "handed off",
        "milestone",
        "milestones",
        "approval",
        "approved",
        "rejected",
        "ship",
        "shipped",
        "deploy",
        "deployed",
        "launch",
        "launched",
        "call",
        "sync",
        "standup",
        "review",
        "interview",
    }
)

# Proper-noun-ish regex: 1–3 consecutive Capitalized tokens. Same spirit
# as ``graph._extract_proper_nouns`` but we don't need to filter sentence
# starters here — the intent is just "looks like an entity". We also
# skip matches that are entirely inside the leading token of a sentence
# by ignoring ``^[A-Z]`` at the very start of the query.
_PROPER_NOUN_PATTERN = re.compile(
    r"(?<![.!?]\s)(?<!^)\b(?:[A-Z][a-z][a-zA-Z0-9]+|[A-Z]{2,}[a-z]?)(?:\s+[A-Z][a-zA-Z0-9]+){0,2}\b"
)

# Short-circuit stopwords — the single-word query ``"Decision"`` should
# not count as an entity. Mostly our own domain's tag vocabulary.
_ENTITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "decision",
        "decisions",
        "meeting",
        "meetings",
        "handoff",
        "handoffs",
        "milestone",
        "milestones",
        "fact",
        "plan",
        "preference",
        "goal",
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "which",
    }
)


def _contains_proper_noun(query: str) -> bool:
    """Return True when the query carries an entity-ish capitalized phrase.

    A single sentence-initial capitalized word doesn't count — that's
    often just "What is X?" or "Show me Y" with unintended capitalization.
    We require the capitalized token to either be non-initial or form a
    multi-word phrase.
    """
    if not query:
        return False
    stripped = query.strip()
    if not stripped:
        return False
    # Normalize the first character for the "sentence-initial" heuristic.
    # We run the regex against ``" " + stripped`` to force the regex's
    # ``(?<!^)`` to fire even for a query whose first token is capitalized
    # only if a *second* capitalized token follows.
    padded = " " + stripped
    for m in _PROPER_NOUN_PATTERN.finditer(padded):
        token = m.group(0)
        head = token.split()[0].lower()
        if head in _ENTITY_STOPWORDS:
            continue
        return True
    # Special case: a multi-word capitalized phrase at the very start of
    # the query (e.g. "Foundry Ventures" as the whole query). The
    # ``(?<!^)`` guard above can miss this when there's no preceding
    # punctuation. Detect it manually.
    tokens = stripped.split()
    if len(tokens) >= 2:
        first_two = tokens[:2]
        if all(t and t[0].isupper() and len(t) >= 2 for t in first_two):
            if first_two[0].lower() not in _ENTITY_STOPWORDS:
                return True
    return False


def _contains_temporal(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    for phrase in _TEMPORAL_PHRASES:
        # Word-boundary match for short phrases to avoid catching
        # substring accidents (``today`` inside ``todayish`` etc).
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, q):
            return True
    if _TEMPORAL_PREP_PATTERN.search(query):
        # Only count ``since/before/after`` as temporal when followed by
        # *something* — avoid firing on a lone keyword.
        return True
    if _DATE_PATTERN.search(query):
        return True
    return False


def _contains_event_word(query: str) -> bool:
    q = query.lower()
    # Tokenize on word boundaries so "meetings" matches the plural form.
    tokens = re.findall(r"\b[a-z]+\b", q)
    if not tokens:
        return False
    return any(t in _EVENT_WORDS for t in tokens)


def classify(query: str) -> Intent:
    """Return the intent label for a user query.

    Priority order (first match wins):
      1. ``temporal`` — query asks "when" something happened.
      2. ``event`` — query asks about meetings/decisions/handoffs.
      3. ``entity`` — query contains a proper noun.
      4. ``general`` — fallback.

    Temporal beats event because "decisions last week" is first-and-foremost
    a temporal slice — the retriever should use recency filters before
    it filters by episode type. Event beats entity because "meeting with
    Foundry" is structurally an event query about a specific entity —
    searching the event retriever first (meeting-labeled episodes) is
    the safer default.

    Args:
        query: User query. Empty / whitespace-only → ``"general"``.

    Returns:
        One of ``"entity" | "temporal" | "event" | "general"``.
    """
    if not query or not query.strip():
        return "general"

    if _contains_temporal(query):
        return "temporal"
    if _contains_event_word(query):
        return "event"
    if _contains_proper_noun(query):
        return "entity"
    return "general"
