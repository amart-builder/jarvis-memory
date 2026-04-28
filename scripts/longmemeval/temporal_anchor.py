"""Anchored temporal range parser + query expansion — Stage 4D port.

Ported directly from OMEGA's
``omega-memory/scripts/longmemeval_official.py`` (lines 559–879). These
two techniques are the largest deltas between OMEGA's 95.4% and our
93.4% on gpt-4.1:

  1. ``infer_temporal_range_anchored`` — given a question with a date
     anchor like "in December", "last weekend", "two weeks ago",
     "between X and Y", returns an absolute (start, end) ISO date
     window. Used to filter retrieval to date-relevant sessions for
     ALL question categories that have temporal signals — not just
     ``temporal-reasoning``.

  2. ``resolve_relative_dates`` + ``expand_query`` — at query time,
     append:
       * counting cues ("every instance all occurrences each time")
         for "how many" / "how much" questions
       * resolved absolute date keywords (e.g., "last Monday" →
         "Monday 2024-03-04 March 04") for embedding match
       * proper-noun entities for explicit lexical match

The motivation for porting these specifically: failure analysis on
Stage 1.5 still-wrongs showed gold sessions are RETRIEVED (top20 hits
=ranks 0-19) but ranked LOW for date-anchored questions, with
distractors filling the top-5/top-10 slots that the LLM actually
reads. OMEGA's anchored filter pushes date-relevant hits to the top.

These are pure functions — no I/O, no LLM calls — so they're
deterministic and unit-testable.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional


_WORD_TO_NUM: dict[str, int] = {
    "one": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20, "thirty": 30,
}

_DAY_NAMES: tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)

_MONTH_NAMES: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# Counting question signals (lowercase substring match).
_COUNTING_CUES: tuple[str, ...] = (
    "how many", "how much", "how often", "total number", "count",
)

# Common English words capitalized at the start of a question — NOT entities.
_COMMON_CAP_WORDS: frozenset[str] = frozenset({
    "I", "The", "A", "An", "My", "What", "When", "Where", "Who", "How",
    "Which", "Why", "Do", "Does", "Did", "Is", "Are", "Was", "Were",
    "Have", "Has", "Had", "Can", "Could", "Would", "Should", "Will",
    "If", "In", "On", "At", "To", "For", "Of", "And", "Or", "But",
    "Not", "That", "This", "It", "He", "She", "They", "We", "You",
    "Please", "Tell", "Me", "About",
})


def _parse_anchor(anchor_date: str) -> Optional[datetime]:
    """Parse a LongMemEval-style or ISO datetime string.

    Returns ``None`` if neither format applies. LongMemEval format is
    ``"YYYY/MM/DD (Day) HH:MM"`` (the ``(Day)`` parenthetical is stripped
    before parsing).
    """
    if not anchor_date:
        return None
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", anchor_date).strip()
    try:
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
    except ValueError:
        try:
            return datetime.fromisoformat(anchor_date)
        except ValueError:
            return None


def infer_temporal_range_anchored(
    query_text: str, anchor_date: str,
) -> Optional[tuple[str, str]]:
    """Infer an absolute ISO date window from temporal phrases in the query.

    Resolves phrases like "N weeks ago", "last Monday", "between X and Y",
    "last N months", "in March 2024" against the question's anchor_date
    (NOT ``datetime.now()`` — the question's date is the reference point).

    Returns ``(start_iso, end_iso)`` or ``None`` if no temporal signal is
    detected. The window includes a small buffer so retrieval doesn't
    miss sessions on the boundary.
    """
    anchor = _parse_anchor(anchor_date)
    if anchor is None:
        return None

    # Pattern 1: "last (Monday|...|Sunday|weekend)"
    day_match = re.search(
        r"last\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|weekend)",
        query_text, re.IGNORECASE,
    )
    if day_match:
        day_name = day_match.group(1).capitalize()
        if day_name == "Weekend":
            target_weekday = 5  # Saturday
        else:
            day_map = {d: i for i, d in enumerate(_DAY_NAMES)}
            target_weekday = day_map[day_name]
        days_back = (anchor.weekday() - target_weekday) % 7
        if days_back == 0:
            days_back = 7  # "last X" = previous one, not today
        target_date = anchor - timedelta(days=days_back)
        start = (target_date - timedelta(days=2)).isoformat()
        end = (target_date + timedelta(days=2)).isoformat()
        return (start, end)

    # Pattern 2: "N days/weeks/months/years ago" (digits or words)
    m = re.search(
        r"(\d+|[a-z]+)\s+(day|week|month|year)s?\s+ago",
        query_text, re.IGNORECASE,
    )
    if m:
        raw_n = m.group(1).lower()
        if raw_n.isdigit():
            n: Optional[int] = int(raw_n)
        else:
            n = _WORD_TO_NUM.get(raw_n)
        if n is not None:
            unit = m.group(2).lower()
            delta = _unit_to_delta(unit, n)
            if delta is not None:
                center = anchor - delta
                buffer = max(delta * 0.25, timedelta(days=3))
                start = (center - buffer).isoformat()
                end = (center + buffer).isoformat()
                return (start, end)

    # Pattern 3: "between DATE and DATE" or "from DATE to DATE"
    m = re.search(
        r"(?:between|from)\s+(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s+(?:and|to)\s+(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
        query_text, re.IGNORECASE,
    )
    if m:
        try:
            d1 = datetime.strptime(m.group(1).replace("/", "-"), "%Y-%m-%d")
            d2 = datetime.strptime(m.group(2).replace("/", "-"), "%Y-%m-%d")
            start = (min(d1, d2) - timedelta(days=1)).isoformat()
            end = (max(d1, d2) + timedelta(days=1)).isoformat()
            return (start, end)
        except ValueError:
            pass

    # Pattern 4: "last N days/weeks/months/years" / "past N ..." / "previous N ..."
    m = re.search(
        r"(?:last|past|previous)\s+(\d+|[a-z]+)\s+(day|week|month|year)s?",
        query_text, re.IGNORECASE,
    )
    if m:
        raw_n = m.group(1).lower()
        if raw_n.isdigit():
            n = int(raw_n)
        else:
            n = _WORD_TO_NUM.get(raw_n)
        if n is not None:
            unit = m.group(2).lower()
            delta = _unit_to_delta(unit, n)
            if delta is not None:
                start = (anchor - delta - timedelta(days=1)).isoformat()
                end = anchor.isoformat()
                return (start, end)

    # Pattern 5: "in [Month] [Year]" (explicit year)
    m = re.search(
        r"in\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})",
        query_text, re.IGNORECASE,
    )
    if m:
        month_name = m.group(1).capitalize()
        year = int(m.group(2))
        month_num = _MONTH_NAMES.index(month_name) + 1
        start_dt = datetime(year, month_num, 1) - timedelta(days=1)
        if month_num == 12:
            end_dt = datetime(year + 1, 1, 1) + timedelta(days=1)
        else:
            end_dt = datetime(year, month_num + 1, 1) + timedelta(days=1)
        return (start_dt.isoformat(), end_dt.isoformat())

    return None


def _unit_to_delta(unit: str, n: int) -> Optional[timedelta]:
    """Map ``("day"|"week"|"month"|"year", n)`` to a ``timedelta``.

    ``month``/``year`` use 30/365-day approximations — same as OMEGA.
    Calendar accuracy isn't critical here; the window is buffered
    by 25% on each side so a few-day drift doesn't matter.
    """
    if unit == "day":
        return timedelta(days=n)
    if unit == "week":
        return timedelta(weeks=n)
    if unit == "month":
        return timedelta(days=n * 30)
    if unit == "year":
        return timedelta(days=n * 365)
    return None


def resolve_relative_dates(query: str, anchor: datetime) -> list[str]:
    """Resolve relative time phrases to absolute date keywords.

    Returns a list of strings to APPEND to the query for better embedding/
    keyword match against session date strings. Covers the same patterns
    as :func:`infer_temporal_range_anchored` plus a few simple ones
    (yesterday, last week).

    Each returned string is a space-separated bag of date tokens
    (ISO dates + month names + day names) — embeddings and FTS both
    benefit from these as keywords.
    """
    q_lower = query.lower()
    resolved: list[str] = []

    # 1. "last (Monday|...|Sunday)" → resolved absolute date + day name
    day_match = re.search(
        r"last\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
        query, re.IGNORECASE,
    )
    if day_match:
        day_name = day_match.group(1).capitalize()
        day_map = {d: i for i, d in enumerate(_DAY_NAMES)}
        target_weekday = day_map[day_name]
        days_back = (anchor.weekday() - target_weekday) % 7
        if days_back == 0:
            days_back = 7
        target_date = anchor - timedelta(days=days_back)
        resolved.append(
            f"{day_name} {target_date.strftime('%Y-%m-%d')} "
            f"{target_date.strftime('%B %d')}"
        )

    # 2. "last weekend" → Saturday + Sunday absolute dates
    if "last weekend" in q_lower:
        sat = anchor - timedelta(days=(anchor.weekday() + 2) % 7 or 7)
        sun = sat + timedelta(days=1)
        resolved.append(
            f"Saturday Sunday {sat.strftime('%Y-%m-%d')} {sun.strftime('%Y-%m-%d')}"
        )

    # 3. "yesterday" → absolute date
    if "yesterday" in q_lower:
        yest = anchor - timedelta(days=1)
        resolved.append(f"{yest.strftime('%Y-%m-%d')} {yest.strftime('%B %d')}")

    # 4. "last week" → date range
    if "last week" in q_lower and "weekend" not in q_lower:
        start = anchor - timedelta(days=anchor.weekday() + 7)
        end = start + timedelta(days=6)
        resolved.append(f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}")

    # 5. "N days/weeks/months/years ago"
    m = re.search(
        r"(\d+|[a-z]+)\s+(day|week|month|year)s?\s+ago",
        query, re.IGNORECASE,
    )
    if m:
        raw_n = m.group(1).lower()
        if raw_n.isdigit():
            n: Optional[int] = int(raw_n)
        else:
            n = _WORD_TO_NUM.get(raw_n)
        if n is not None:
            unit = m.group(2).lower()
            delta = _unit_to_delta(unit, n)
            if delta is not None:
                center = anchor - delta
                resolved.append(
                    f"{center.strftime('%Y-%m-%d')} "
                    f"{center.strftime('%B')} {center.strftime('%d')}"
                )

    # 6. "last N days/weeks/months" or "past N months" → date range
    m = re.search(
        r"(?:last|past|previous)\s+(\d+|[a-z]+)\s+(day|week|month|year)s?",
        query, re.IGNORECASE,
    )
    if m:
        raw_n = m.group(1).lower()
        if raw_n.isdigit():
            n = int(raw_n)
        else:
            n = _WORD_TO_NUM.get(raw_n)
        if n is not None:
            unit = m.group(2).lower()
            delta = _unit_to_delta(unit, n)
            if delta is not None:
                start = anchor - delta
                resolved.append(
                    f"{start.strftime('%Y-%m-%d')} {anchor.strftime('%Y-%m-%d')} "
                    f"{start.strftime('%B')} {anchor.strftime('%B')}"
                )

    # 7. "in [Month]" (without year) → most recent occurrence ≤ anchor
    m = re.search(
        r"in\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b",
        query, re.IGNORECASE,
    )
    if m and not re.search(
        r"in\s+" + m.group(1) + r"\s+\d{4}", query, re.IGNORECASE,
    ):
        month_name = m.group(1).capitalize()
        month_num = _MONTH_NAMES.index(month_name) + 1
        if month_num <= anchor.month:
            year = anchor.year
        else:
            year = anchor.year - 1
        resolved.append(f"{month_name} {year} {year}-{month_num:02d}")

    # 8. "in [Month] [Year]" → explicit
    m = re.search(
        r"in\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})",
        query, re.IGNORECASE,
    )
    if m:
        month_name = m.group(1).capitalize()
        year = int(m.group(2))
        month_num = _MONTH_NAMES.index(month_name) + 1
        resolved.append(f"{month_name} {year} {year}-{month_num:02d}")

    return resolved


def _extract_proper_nouns(query: str) -> list[str]:
    """Extract capitalized words that look like proper nouns.

    Excludes common English words that happen to be capitalized at the
    start of a question ("What", "Why", "How", etc.). Returns a list of
    multi-word entities preserved as space-separated phrases (so
    "Rachel Smith" stays together).
    """
    if not query:
        return []
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
    return [w for w in words if w not in _COMMON_CAP_WORDS and len(w) > 1]


def expand_query(query: str, question_date: Optional[str] = None) -> str:
    """Expand a query with counting cues + resolved dates + proper nouns.

    The expansion is purely additive — appended to the original query.
    The original query is preserved verbatim so any keyword/lexical
    retriever still matches the user's exact phrasing.

    Returns the original ``query`` unchanged when no signals fire.
    """
    expansions: list[str] = []

    # 1. Counting signal
    q_lower = query.lower()
    if any(cue in q_lower for cue in _COUNTING_CUES):
        expansions.append("every instance all occurrences each time")

    # 2. Resolved relative dates
    if question_date:
        anchor = _parse_anchor(question_date)
        if anchor is not None:
            expansions.extend(resolve_relative_dates(query, anchor))

    # 3. Proper-noun entities
    entities = _extract_proper_nouns(query)
    if entities:
        expansions.append(" ".join(entities))

    if not expansions:
        return query
    return query + " " + " ".join(expansions)


def hit_in_temporal_window(
    referenced_date: str, window: tuple[str, str],
) -> bool:
    """Return True if ``referenced_date`` falls within the inclusive window.

    Parses ISO format. Returns False on any parse error so a malformed
    date never silently passes the filter.
    """
    if not referenced_date:
        return False
    try:
        d = datetime.fromisoformat(referenced_date)
        start = datetime.fromisoformat(window[0])
        end = datetime.fromisoformat(window[1])
    except (ValueError, TypeError):
        return False
    return start <= d <= end
