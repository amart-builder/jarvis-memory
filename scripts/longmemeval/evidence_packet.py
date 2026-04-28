"""Evidence packet extractor — Stage 5 v2 Phase 3.

Pulls user-turn snippets containing high-signal content (dates, quantities,
named entities, ordinals) from retrieved sessions. Used as a "salience
rescue" prefix for MS/TR/KU prompts where the model has all gold sessions
in context but fails to enumerate scattered evidence.

Targets the dominant failure mode after Stage 4D's diagnostic re-read
(``docs/eval/codex-stage5-review.md``): 21 of 32 still-wrong questions
have all gold sessions visible in the prompt; the model just doesn't
weight them. An evidence packet pre-extracts the high-signal user turns
into a dense block at the top of the prompt — the model now reads "5 user
turns about baking with dates" and counts them off, rather than scanning
20 chronological transcripts and missing items.

Heuristic-only — no LLM call. If smoke testing shows the heuristic misses
the cases that matter, swap to LLM extraction in Phase 8 fallback.

Phase 4 (temporal two-lane) builds on this module: it partitions the
chronological notes block by date-window, but the evidence packet remains
the leading high-signal block.
"""
from __future__ import annotations

import re
from typing import Any, Optional


# ── Signal detection regexes ─────────────────────────────────────────


# ISO and written dates (March 15 / 15th of March / January 2024)
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d+(?:st|nd|rd|th)?"
        r"(?:,?\s+\d{4})?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:st|nd|rd|th)\s+(?:of\s+)?"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b",
        re.IGNORECASE,
    ),
]

# Relative temporal phrases (Stage 4D's `infer_temporal_range_anchored`
# patterns + a few more for evidence-packet scoring)
_RELATIVE_TIME = re.compile(
    r"\b(?:"
    r"yesterday|today|tonight|tomorrow|"
    r"this\s+morning|this\s+afternoon|this\s+evening|this\s+week|this\s+weekend|"
    r"last\s+(?:week|weekend|month|year|night|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"this\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"next\s+(?:week|weekend|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"a\s+(?:week|month|year)\s+ago|"
    r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago|"
    r"(?:two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?|years?)\s+ago|"
    r"in\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)|"
    r"the\s+past\s+(?:few\s+)?(?:days?|weeks?|months?|years?)|"
    r"the\s+other\s+(?:day|week|month)|"
    r"earlier\s+(?:today|this\s+week|this\s+month)|"
    r"recently"
    r")\b",
    re.IGNORECASE,
)

# Money + count-with-unit
_QUANTITY = re.compile(
    r"\$\d+(?:[,.]\d+)*|"
    r"\b\d+(?:[,.]\d+)*\s*(?:dollars?|cents?|euros?|pounds?|usd|gbp)\b|"
    r"\b\d+(?:[,.]\d+)*\s*(?:%|percent)\b|"
    r"\b\d+(?:[,.]\d+)*\s+(?:"
    r"minutes?|hours?|seconds?|days?|weeks?|months?|years?|"
    r"miles?|km|kilometers?|meters?|feet|ft|"
    r"sessions?|times?|points?|followers?|likes?|comments?|views?|"
    r"episodes?|chapters?|pages?|books?|albums?|eps?|songs?|tracks?|"
    r"hits?|runs?|laps?|reps?|sets?|attempts?|tries|wins?|losses?|"
    r"tickets?|bottles?|cups?|cans?|jars?|bags?|boxes?|tanks?|chickens?|"
    r"projects?|classes?|workshops?|conferences?|lectures?|events?|"
    r"recipes?|bakes?|drinks?|cocktails?|meals?|trips?|visits?|days|"
    r"weeks?|months?|years?|kids?|children|cars?|houses?|apartments?|"
    r"customers?|clients?|users?|members?|teammates?|colleagues?|"
    r"friends?|relatives?|cousins?"
    r")\b",
    re.IGNORECASE,
)

# Ordinal (first/second/.../Nth) — strong "Nth occurrence" signal
_ORDINAL = re.compile(
    r"\b(?:"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    r"seventeenth|eighteenth|nineteenth|twentieth|"
    r"\d+(?:st|nd|rd|th)"
    r")\b",
    re.IGNORECASE,
)

# Capitalized non-stoplist tokens — likely proper nouns
_STOPLIST = frozenset({
    "I", "A", "The", "An", "And", "But", "So", "Then", "Now", "Yes", "No",
    "OK", "Okay", "User", "Assistant", "Hi", "Hello", "Hey", "Sure",
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
})
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}\b")


# ── Scoring + extraction ─────────────────────────────────────────────


def _score_user_turn(text: str) -> int:
    """Score a user-turn string by signal density.

    Higher score = more dates/quantities/entities/ordinals = more likely
    to be relevant to a question that needs enumeration / recall of
    specific events. Pure heuristic — no semantic understanding.
    """
    score = 0
    for p in _DATE_PATTERNS:
        score += 3 * len(p.findall(text))
    score += 3 * len(_RELATIVE_TIME.findall(text))
    score += 2 * len(_QUANTITY.findall(text))
    score += 2 * len(_ORDINAL.findall(text))
    for m in _PROPER_NOUN.findall(text):
        if m not in _STOPLIST:
            score += 1
    return score


_TURN_HEAD = re.compile(r"^(user|assistant)\s*:\s*(.*)$", re.IGNORECASE)


def _split_user_turns(content: str) -> list[str]:
    """Parse role-prefixed content into a list of USER-only turn strings.

    The adapter stores sessions as ``role: content\\nrole: content\\n...``
    via ``format_session_text``. We split on each ``role:`` head and keep
    only user turns. Multi-line user turns are joined with spaces.

    Pure function — never raises on malformed input.
    """
    if not content:
        return []
    user_turns: list[str] = []
    cur_role: Optional[str] = None
    cur_buf: list[str] = []

    def _flush() -> None:
        if cur_role == "user" and cur_buf:
            text = " ".join(s for s in cur_buf if s).strip()
            if text:
                user_turns.append(text)

    for line in content.split("\n"):
        m = _TURN_HEAD.match(line)
        if m:
            _flush()
            cur_role = m.group(1).lower()
            cur_buf = [m.group(2)]
        else:
            if cur_role is not None:
                cur_buf.append(line)
    _flush()
    return user_turns


def _truncate_snippet(text: str, max_chars: int = 160) -> str:
    """Soft-truncate a user-turn snippet for the packet.

    Keeps full sentences when possible — break at the last sentence
    boundary inside ``max_chars`` if one exists; otherwise hard-cut + "...".
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - 3]
    # Prefer a sentence boundary inside the head
    for end_char in (". ", "! ", "? "):
        idx = head.rfind(end_char)
        if idx > max_chars // 2:
            return head[: idx + 1].rstrip()
    return head.rstrip() + "..."


# ── Public entry point ───────────────────────────────────────────────


def build_evidence_packet(
    hits: list[dict[str, Any]],
    query: str,
    *,
    max_snippets: int = 12,
    min_signal_score: int = 1,
) -> str:
    """Build a ``[High-signal evidence]`` block from retrieved hits.

    Format:

        [High-signal evidence — items extracted from the chronological notes]
        - 2023-05-20 [Note 3] User: "I made the apple pie..."
        - 2023-05-21 [Note 6] User: "I tried out a new bread recipe..."
        ...

        [Below: full chronological notes — VERIFY claims against these]

    Returns the empty string when no signal-bearing user turns were found
    (which causes the caller to fall through to the existing prompt
    structure unchanged — safe no-op).

    Args:
        hits: list of hit dicts with ``content`` (role-prefixed
            transcript) and ``referenced_date``.
        query: the user's question. Currently unused — extractor is
            query-agnostic — kept in signature for future query-aware
            scoring.
        max_snippets: cap on snippets in the final packet.
        min_signal_score: drop turns scoring below this threshold.

    Returns:
        Multi-line string ending with a separator instruction, or "" if
        no qualifying snippets.
    """
    # (score, hit_idx, turn_idx, date_prefix, text)
    candidates: list[tuple[int, int, int, str, str]] = []
    for hit_idx, h in enumerate(hits, start=1):
        content = str(h.get("content") or "")
        date_str = str(h.get("referenced_date") or h.get("created_at") or "")
        date_prefix = date_str[:10] if len(date_str) >= 10 else ""
        for turn_idx, turn in enumerate(_split_user_turns(content), start=1):
            score = _score_user_turn(turn)
            if score < min_signal_score:
                continue
            candidates.append((score, hit_idx, turn_idx, date_prefix, turn))

    if not candidates:
        return ""

    # Top-N by score (descending). Tiebreak: chronological hit_idx, then
    # turn_idx — keeps deterministic order for the same retrieval.
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    top = candidates[:max_snippets]
    # Re-sort the survivors chronologically so the packet reads in note
    # order — easier for the model to enumerate.
    top.sort(key=lambda x: (x[1], x[2]))

    lines = [
        "[High-signal evidence — user-turn snippets extracted from the "
        "chronological notes below. Use these to identify relevant items, "
        "then VERIFY each claim against the full notes.]",
    ]
    for _score, hit_idx, _turn_idx, date, text in top:
        date_part = f"{date} " if date else ""
        snippet = _truncate_snippet(text)
        lines.append(f'- {date_part}[Note {hit_idx}] User: "{snippet}"')
    lines.append("")
    lines.append(
        "[End of evidence packet. Full chronological notes follow — these "
        "are the ground truth.]"
    )
    return "\n".join(lines)
