"""LongMemEval question classifier — regex/heuristic, no LLM.

Per the pre-registered protocol (docs/eval/longmemeval-v1.1-protocol.md),
we do NOT read ``question_type`` from the dataset. We classify each
question into one of LongMemEval's 6 categories from the question text
alone, then route prompts and retrieval K based on the predicted label.

Priority order (first match wins):
  1. ``single-session-assistant`` — back-references to assistant
     ("remind me", "you told me", "previous conversation/chat").
  2. ``single-session-preference`` — recommendation/advice requests
     ("any tips/advice/suggestions", "can you recommend").
  3. ``temporal-reasoning`` — ordering / time-arithmetic ONLY
     ("first/last X", "between A and B" with date-specific terms,
     "how many days before", "N (days|weeks|months) ago", age math).
  4. ``knowledge-update`` — explicit state-update markers BEFORE
     multi-session (since "how many do I currently own" is KU).
  5. ``multi-session`` — counting / aggregation across sessions
     ("how many", "how much", "how often", "the most/least", "total",
     "average").
  6. ``single-session-user`` — default fall-through.

Abstention is NOT a classifier output. In production, abstention is
handled at generation time when retrieval confidence is below a
threshold (per OMEGA's `_abs` config: min_rel=0.20, min_res=2,
max_res=5, max_tokens=256). The classifier still returns one of the
6 categories for an abstention-style question — the abstention prompt
overlay is applied separately by the adapter when retrieval is weak.

We deliberately avoid LLM-based classification: it introduces
non-determinism, cost, and a dependency on an external model that
isn't part of the memory system itself. The regex classifier IS the
classifier — what you see is what runs.
"""
from __future__ import annotations

import re
from typing import Literal, NamedTuple

QuestionType = Literal[
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "multi-session",
    "temporal-reasoning",
]


class Classification(NamedTuple):
    label: QuestionType
    rule: str  # which rule fired, for diagnostics


# ── Rule 1: single-session-assistant ──────────────────────────────────
# Back-reference to something the assistant said previously. The
# distinctive marker is reference to a PRIOR conversation OR
# second-person address to the assistant.
_ASSISTANT_REFS = (
    # explicit "remind me" patterns
    re.compile(r"\bremind me\b", re.I),
    # "you" verbs — assistant did/said something
    re.compile(r"\byou (told|recommended|suggested|mentioned|assigned|said|gave|advised|provided|noted)\b", re.I),
    # references to a prior conversation
    re.compile(r"\b(our|the) previous (chat|conversation|discussion|talk)\b", re.I),
    re.compile(r"\b(going|looking) back (to|at|through) (our|the|that)\b", re.I),
    re.compile(r"\bI('m| am) (checking|going through|looking back at|revisiting)\b", re.I),
    re.compile(r"\bwe (discussed|talked about|chatted about|covered) .* earlier\b", re.I),
    # "I think we discussed X earlier"
    re.compile(r"\bwe (discussed|talked|chatted)\b.{0,40}\b(earlier|before|previously|last time)\b", re.I),
    # remember/recall + assistant
    re.compile(r"\b(remember|recall) (that |when |how )?you\b", re.I),
    # the X you gave/told/recommended
    re.compile(r"\b(the|that) (advice|suggestion|recommendation|tip|answer|response|list|name) (you|that you)\b", re.I),
    # what (color|name|kind|type) did you say
    re.compile(r"\b(what|which|who) .{0,30}(did|do) you (say|tell|recommend|suggest|mention)\b", re.I),
)


# ── Rule 2: single-session-preference ─────────────────────────────────
# Direct recommendation/advice/tips request. The pattern is that the
# user is asking the assistant to suggest something tailored to their
# preferences (which the assistant infers from earlier conversation).
_PREFERENCE_REQS = (
    re.compile(r"\bcan you (recommend|suggest)\b", re.I),
    re.compile(r"\bcould you (recommend|suggest)\b", re.I),
    re.compile(r"\bwould you recommend\b", re.I),
    re.compile(r"\bdo you have (any |a )?(recommendation|suggestion|tip|advice)s?\b", re.I),
    re.compile(r"^(please )?(recommend|suggest)\b", re.I),
    re.compile(r"\bwhat (would|do) you recommend\b", re.I),
    # "Any tips/advice/suggestions/ideas"
    re.compile(r"\b(any|some|got any) (tips|advice|suggestions|ideas|recommendations|thoughts)\b", re.I),
    # "what's a good X"
    re.compile(r"\bwhat('s| is) a good\b", re.I),
)


# ── Rule 3: temporal-reasoning ────────────────────────────────────────
# Ordering, comparison, or explicit time-arithmetic. Sharp rules ONLY —
# we'd rather miss a temporal Q than mis-route a multi-session Q.
# Fires BEFORE knowledge-update + multi-session because temporal Qs
# often contain "how many" (days between/before/after).
_TEMPORAL_PATTERNS = (
    # "first/earliest" ordering (comparative) — high precision
    re.compile(r"\b(which|what|who|where|when) .{0,80}\b(first|earliest)\b", re.I),
    re.compile(r"\bdid I .{0,40}\b(first|earliest)\b", re.I),
    # "what is the order of"
    re.compile(r"\bwhat (is|was|are|were) the order (of|in)\b", re.I),
    # superlative ordering with "I"
    re.compile(r"\bthe (first|earliest) .{0,60}\bI\b", re.I),
    # "X did I do/use most recently" / "what did I most recently"
    re.compile(r"\bmost recent(ly)?\b", re.I),
    # explicit interval arithmetic — "how (long|many days|weeks|...) (between|since|elapsed|passed|ago|until|before|after)"
    re.compile(r"\bhow (long|many (days|weeks|months|years|hours)) (since|between|before|after|elapsed|until|passed|had passed|ago|did it take|will it (take|be))\b", re.I),
    # "how many (units) ago" — must be specifically time arithmetic
    re.compile(r"\bhow many (days|weeks|months|years|hours|minutes) ago\b", re.I),
    # "how many X have passed" / "how many X did I" with passed/elapsed elsewhere
    re.compile(r"\bhow (many|long) .{0,40}\b(passed|elapsed|gone by)\b", re.I),
    # "how many X did I do before/after Y" — comparison anchored on event
    re.compile(r"\bhow (many|much|long) .{0,60}\b(before|after) (the|my|her|his|their|I |starting|finishing|going|attending|completing|seeing|making)\b", re.I),
    # "How (long|old) was/have I"
    re.compile(r"\bhow (long|old) (had|have|was|were|am|been) I\b", re.I),
    re.compile(r"\bhow old (was|am|were|are) I\b", re.I),
    # "before/after I (event-verb)" — comparing two specific events
    re.compile(r"\b(before|after) I .{0,30}\b(started|finished|began|completed|cancelled|attended|received|booked|moved|got|signed)\b", re.I),
    # "between (event-or-date) and (event-or-date)"
    re.compile(r"\bbetween .{0,40}\band\b .{0,40}\b(I|my|day|date|time)\b", re.I),
    # explicit "N days/weeks/months/years (ago|before|after|prior)"
    re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|a few)\s+(day|week|month|year|hour|minute)s?\s+(ago|before|after|prior)\b", re.I),
    # date literal: month name + day number
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d", re.I),
    # ISO date
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # explicit weekday-of-the-week recurring schedule
    re.compile(r"\bwhat time .{0,30}\b(on|every) (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|weekday|weekend)s?\b", re.I),
    # "in March and April" — month range
    re.compile(r"\bin (January|February|March|April|May|June|July|August|September|October|November|December) and (January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I),
    # "a week/month/year ago" — singular forms
    re.compile(r"\b(a|an)\s+(day|week|month|year|while)\s+(ago|before|earlier)\b", re.I),
    # "last (week|month|year) ago" doesn't exist; "last week" alone IS temporal
    re.compile(r"\b(last|past) (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|weekend)\b", re.I),
    # "X did I do (last|this) (week|weekend|month|year)" — anchor with comparison
    # Only fires for "did I" + "last" + time unit (closer match)
    re.compile(r"\bdid I .{0,30}\b(last|this) (week|weekend|month)\b", re.I),
)


# ── Rule 4: knowledge-update ──────────────────────────────────────────
# Explicit recency / state-update markers. Fires BEFORE multi-session
# because "how many do I currently own" is KU, not multi-session.
# We undercount here on Qs without surface markers — that's accepted.
_UPDATE_PATTERNS = (
    # explicit recency adverbs/adjectives
    re.compile(r"\b(currently|right now|as of (now|today)|at the moment|these days)\b", re.I),
    re.compile(r"\b(still|no longer|anymore)\b", re.I),
    # "current X" — possessive needed to avoid noise
    re.compile(r"\b(my|the|his|her|their|our) current\b", re.I),
    # personal best / personal record
    re.compile(r"\bpersonal (best|record|high)\b", re.I),
    # "after X's recent/latest"
    re.compile(r"\b(after|since) (his|her|their|my|the) (recent|latest|new)\b", re.I),
    # change/replace/update verbs paired with first-person
    re.compile(r"\b(have I|did I) (switched|changed|replaced|updated|moved (to|from))\b", re.I),
    # "X anymore" or "X yet"
    re.compile(r"\b(do I still|am I still)\b", re.I),
    # "the same X as" — comparison to prior state
    re.compile(r"\bthe same .{0,30}\bas (me|I|before|previously|last time)\b", re.I),
    # "for the X, did I switch to" — KU about ratio/method changes
    re.compile(r"\bdid I switch (to|from|over)\b", re.I),
    # "since I started" or "have I been doing" — implies continuing state
    re.compile(r"\b(since|after) I (started|began|switched|changed|moved)\b", re.I),
    re.compile(r"\bhow long have I been\b", re.I),
    # how many X do I have/own/use right now (without temporal markers)
    re.compile(r"\bhow many .{0,40}\bdo I (currently|now|still)\b", re.I),
    # "did I (finish|complete) X?" — state of activity
    re.compile(r"\bdid I (finish|complete|stop|quit|cancel)\b", re.I),
    # "more frequently than I did previously" — comparison to prior state
    re.compile(r"\bmore (frequently|often|rarely|less) than I (did|used to)\b", re.I),
    # "have I X recently" — KU about activity since a state change
    re.compile(r"\b(have I|did I) .{0,40}\b(recently|lately)\b", re.I),
)


# ── Rule 5: multi-session (counting/aggregation across sessions) ──────
# Counting words OR aggregation language. OMEGA's K-floor lexicon plus
# our additions for aggregation Qs without "how many".
_COUNTING_PATTERNS = (
    re.compile(r"\bhow many\b", re.I),
    re.compile(r"\bhow much\b", re.I),
    re.compile(r"\bhow often\b", re.I),
    re.compile(r"\btotal (number|amount|distance|cost|count|time|hours)\b", re.I),
    re.compile(r"\bnumber of\b", re.I),
    re.compile(r"\b(list|name) (all|every)\b", re.I),
    re.compile(r"\bcount\b", re.I),
    # aggregation language
    re.compile(r"\bthe (most|least|highest|lowest|biggest|smallest|largest)\b", re.I),
    re.compile(r"\b(average|median|sum|total) (of|amount|number|distance|cost|time)\b", re.I),
    re.compile(r"\bthe average\b", re.I),
    re.compile(r"\bin (total|all)\b", re.I),
    re.compile(r"\b(across|over) (all|the past|the last|my|our)\b", re.I),
    re.compile(r"\b(in|over|during) the (past|last) (few|several|\d+)\b", re.I),
    # "what is the total X" / "what's the total X"
    re.compile(r"\bwhat('s| is) the (total|combined|overall)\b", re.I),
)


def classify(question: str) -> Classification:
    """Classify a LongMemEval question by regex priority order.

    Returns the predicted category and the name of the rule that fired
    (for diagnostics / accuracy reports). Defaults to
    ``single-session-user`` if no rule matches.
    """
    if not question or not question.strip():
        return Classification("single-session-user", "default-empty")

    q = question.strip()

    # 1. assistant back-reference
    for pat in _ASSISTANT_REFS:
        if pat.search(q):
            return Classification("single-session-assistant", f"assistant:{pat.pattern[:30]}")

    # 2. preference / recommendation request
    for pat in _PREFERENCE_REQS:
        if pat.search(q):
            return Classification("single-session-preference", f"preference:{pat.pattern[:30]}")

    # 3. temporal-reasoning (BEFORE everything else with counting language —
    #    "how many days between X and Y" is temporal, not multi-session)
    for pat in _TEMPORAL_PATTERNS:
        if pat.search(q):
            return Classification("temporal-reasoning", f"temporal:{pat.pattern[:30]}")

    # 4. knowledge-update (BEFORE multi-session — "how many do I currently own"
    #    is KU, not multi-session — recency markers win over counting words)
    for pat in _UPDATE_PATTERNS:
        if pat.search(q):
            return Classification("knowledge-update", f"update:{pat.pattern[:30]}")

    # 5. multi-session (counting/aggregation)
    for pat in _COUNTING_PATTERNS:
        if pat.search(q):
            return Classification("multi-session", f"counting:{pat.pattern[:30]}")

    # 6. default
    return Classification("single-session-user", "default")


# ── Per-category retrieval / generation config ────────────────────────
# Adapted from OMEGA's `longmemeval_official.py` lines 354-364 and
# 1471-1483, plus our own AR1 (PPR alpha=0.5) and AR2 (PPR seed
# broadening) additions per docs/eval/longmemeval-v1.1-protocol.md.

# Top-K retrieval floors per category (OMEGA lines 1471-1483).
K_FLOORS: dict[str, int] = {
    "single-session-user": 20,
    "single-session-assistant": 20,
    "single-session-preference": 20,
    "knowledge-update": 20,
    "multi-session": 25,
    "temporal-reasoning": 25,
}


# K floor specifically for counting questions (subset of multi-session).
# OMEGA bumps to 45 when "how many/much/often/total/count/number of"
# appears (lines 1473-1480).
COUNTING_K_FLOOR: int = 45


def is_counting_question(question: str) -> bool:
    """Counting questions get a K floor of 45 (OMEGA's recipe)."""
    return any(pat.search(question) for pat in _COUNTING_PATTERNS)


# Adaptive filter thresholds per category (OMEGA lines 354-364).
FILTER_CONFIG: dict[str, dict[str, float | int]] = {
    "single-session-user":       {"min_rel": 0.12, "min_res": 3, "max_res": 12, "max_tokens": 512},
    "single-session-assistant":  {"min_rel": 0.15, "min_res": 2, "max_res": 10, "max_tokens": 512},
    "single-session-preference": {"min_rel": 0.12, "min_res": 3, "max_res": 10, "max_tokens": 2048},
    "knowledge-update":          {"min_rel": 0.15, "min_res": 3, "max_res": 15, "max_tokens": 2048},
    "multi-session":             {"min_rel": 0.08, "min_res": 4, "max_res": 20, "max_tokens": 2048},
    "temporal-reasoning":        {"min_rel": 0.10, "min_res": 5, "max_res": 20, "max_tokens": 2048},
}


# Abstention overlay (applied when retrieval confidence is too low).
# OMEGA lines 1198-1199. Triggered when top-K relevance below `min_rel`
# even after retrieval — adapter checks this and switches prompt.
ABSTENTION_FILTER: dict[str, float | int] = {
    "min_rel": 0.20,
    "min_res": 2,
    "max_res": 5,
    "max_tokens": 256,
}
