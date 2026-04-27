"""LongMemEval generation prompts — ported VERBATIM from OMEGA.

Source of truth: omega-memory/omega-memory/scripts/longmemeval_official.py
lines 142-341 (commit pulled 2026-04-26). These are the prompts behind
OMEGA's reported 95.4% (gpt-4.1, single-shot, temperature=0).

Per the pre-registered protocol (docs/eval/longmemeval-v1.1-protocol.md
§AR5), we copy these verbatim — they're battle-tested across 4+
benchmark iterations and contain category-specific reasoning rules
(e.g., "RECOLLECTION ≠ ACTION" for temporal, "ground truth comes from
the user's statement, not the assistant's" for multi-session).

Modifying these prompts is an explicit protocol decision — if a
modification is made, it must be documented as a new AR (e.g. "AR6:
strengthen MULTISESSION prompt with ..."). No silent edits.

The MULTISESSION prompt already includes the counting-enumeration
discipline (rule 1 of "Important:") that AR3 was meant to add — so
AR3 ships free as part of the verbatim port.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Final


# ── VANILLA: best for single-session-assistant + single-session-user ──
RAG_PROMPT_VANILLA: Final[str] = """\
I will give you several notes from past conversations between you and a user. \
Please answer the question based on the relevant notes. \
If the question cannot be answered based on the provided notes, say so.

Notes from past conversations:

{sessions}

Current Date: {question_date}
Question: {question}
Answer:"""


# ── ENHANCED: best for knowledge-update ───────────────────────────────
# Recency + aggregation + confidence rules.
RAG_PROMPT_ENHANCED: Final[str] = """\
I will give you several notes from past conversations between you and a user, \
ordered from oldest to newest. Please answer the question based on the relevant notes. \
If the question cannot be answered based on the provided notes, say so.

You MUST follow this process for EVERY question:

STEP 1 — Scan ALL notes for mentions of the queried topic. List every note that \
discusses it, with its note number and date.

STEP 2 — If the topic appears in multiple notes, compare the values. The note \
with the LATEST date is the ONLY correct one. Earlier values are SUPERSEDED and WRONG.

STEP 3 — Answer using ONLY the value from the latest note.

CRITICAL rules:
- Notes are in chronological order (oldest first). Higher note numbers are more recent.
- For questions about current state (e.g., "what is my current X?", "how many times \
have I done Y?"), the answer ALWAYS comes from the LAST note mentioning that topic.
- If a quantity changes across notes (e.g., worn 4 times → worn 6 times), the \
LATEST number replaces all earlier ones. Do NOT add or average them.
- If the question references a role, title, or name that does NOT exactly match \
what appears in the notes, say the information is not enough to answer.
- If the question asks "how many" or for a count/total, enumerate all relevant \
items and then state the final number clearly.
- Give a direct, concise answer. Do not hedge if the evidence is clear.

Notes from past conversations:

{sessions}

Current Date: {question_date}
Question: {question}
Answer:"""


# ── MULTISESSION: best for multi-session ─────────────────────────────
# Counting/aggregation rules. Includes anti-conflation, dedup, and
# "user's statement is ground truth" — addresses OMEGA's documented
# multi-session weakness directly.
RAG_PROMPT_MULTISESSION: Final[str] = """\
I will give you several notes from past conversations between you and a user, \
ordered from oldest to newest. Please answer the question based on the relevant notes. \
If the question cannot be answered based on the provided notes, say so.

Important:
- Notes are in chronological order. When the same fact appears in multiple \
notes with different values, always use the value from the MOST RECENT note.
- If the question asks "how many", for a count, or for a total:
  1. You MUST list EVERY matching item individually, citing its source as [Note #].
  2. VERIFY each item: re-read the question and confirm each item EXACTLY matches \
what was asked. If the question asks about "types of citrus fruits", only count \
distinct fruit types the user actually used, not every mention of citrus. If it \
asks about "projects I led", only count projects where the user was the leader.
  3. INCLUDE every item the user mentions that is even loosely related to the \
question. Err on the side of inclusion when borderline. Then group near-duplicates \
together as one item before counting. The user's mention is the count — do NOT \
pre-filter items because "the assistant questioned whether it's real" or because \
"the description is vague". The user's statement is ground truth.
  4. After grouping near-duplicates, count the remaining items and state the total clearly.
  5. For "how much total" questions: list each amount with its source [Note #], \
then sum them and state the total.
- When the same fact is UPDATED in a later note (e.g., a number changes from X to Y), \
use ONLY the latest value. The earlier value is superseded.
- DEDUPLICATION: When counting across notes, watch for the same event/item described \
differently (e.g., "cousin's wedding" and "Rachel's wedding at a vineyard" may be the \
same event). If two items could be the same, count them as ONE. Err on the side of \
merging duplicates rather than double-counting.
- For questions about an "increase", "decrease", or "change" in a quantity: you MUST find \
BOTH the starting value AND the ending value, then compute the DIFFERENCE. Do NOT report \
the final total as the increase. Example: if followers went from 250 to 350, the increase is 100.
- Do NOT skip notes. Scan every note for potential matches before answering.
- Give a direct, concise answer. Do not hedge if the evidence is clear.
- NEVER guess, estimate, or calculate values that are not explicitly stated in the notes. \
If the notes mention a taxi costs $X but never mention the bus/train price (or vice versa), \
say the information is not enough to answer — do NOT compute a savings amount from missing data.

ENUMERATION DISCIPLINE (Stage 2 — addresses the multi-session under-counting failure mode):
- Count by ENUMERATING. List every matching item, then count the list length. Do NOT \
estimate or eyeball the count.
- Preserve quantities, units, and dates EXACTLY as the user stated them. If the user \
said "$45.50", do not round to "$45". If the user said "March 15", do not paraphrase to \
"mid-March".
- USER STATEMENT BEATS ASSISTANT SKEPTICISM. If the user said they did the thing, count \
it. Even if the assistant in the conversation expressed doubt or asked for clarification.

FINAL ANSWER FORMAT — for any question that asks for a count, total, or "how many":
- Your final line MUST be exactly: "Total: N" where N is the integer count.
- If your enumerated list has 3 items, the count is 3 — not "2 or 3", not "approximately 3", \
not "3 or more". Be decisive.
- The "Total: N" line goes AFTER your enumerated list and any narrative explanation.

Notes from past conversations:

{sessions}

Current Date: {question_date}
Question: {question}
Answer:"""


# ── PREFERENCE: best for single-session-preference ────────────────────
# Forces personalization — generic advice is wrong.
RAG_PROMPT_PREFERENCE: Final[str] = """\
I will give you several notes from past conversations between you and a user. \
Please answer the question based on the user's stated preferences, habits, and \
personal information found in these notes. \
If the question cannot be answered based on the provided notes, say so.

Important:
- Focus on what the user explicitly said about their preferences, likes, dislikes, \
habits, routines, and personal details.
- When the same preference appears in multiple notes with different values, always \
use the value from the MOST RECENT note (higher note number = more recent).
- If the question asks for a recommendation or suggestion, USE the user's stated \
preferences to tailor your response. Do NOT say you lack information if the notes \
contain ANY relevant preferences, interests, or habits — apply them creatively.
- Even if the notes don't mention the exact topic, look for RELATED preferences \
(e.g., if asked about hotels, use stated preferences about views, amenities, \
luxury vs budget, or location preferences from ANY context).
- When the user mentions a place, activity, or event, ALWAYS check if the notes \
contain a SPECIFIC PAST EXPERIENCE with that place/activity. If so, reference it \
directly (e.g., "You mentioned enjoying X when you visited Denver before" or \
"Given your experience with Y in high school").
- Your answer MUST reference at least one specific detail from the notes. Generic \
advice that could apply to anyone is WRONG. The answer should be clearly \
personalized — someone reading it should be able to tell it was written for this \
specific user.
- Give a direct, specific answer. Quote the user's own words when possible.

Notes from past conversations:

{sessions}

Current Date: {question_date}
Question: {question}
Answer:"""


# ── TEMPORAL: best for temporal-reasoning ─────────────────────────────
# Date arithmetic + RECOLLECTION ≠ ACTION rule.
RAG_PROMPT_TEMPORAL: Final[str] = """\
I will give you several notes from past conversations between you and a user, \
ordered from oldest to newest. Each note has a date stamp. Please answer the \
question based on the relevant notes. \
If the question cannot be answered based on the provided notes, say so.

You MUST follow these steps for ALL time-based questions:

STEP 1 — Convert every relative date to an ABSOLUTE date:
  For each event mentioned in the notes, write its absolute date. Convert ALL \
relative references using the note's own date stamp:
  - "last Saturday" = the most recent Saturday BEFORE the note's date
  - "yesterday" = the day before the note's date
  - "two weeks ago" = 14 days before the note's date
  - "last month" = the calendar month before the note's date
  - "next Friday" = the first Friday AFTER the note's date

STEP 2 — Find ALL candidate events, not just the first match:
  When the question asks about something at a specific time (e.g., "two weeks ago", \
"last Saturday"), scan ALL notes and list every event that could match both the \
time reference AND the event description. Do NOT stop at the first event near \
the target date.

STEP 3 — Select the best match by verifying BOTH date AND description:
  - The event must match the question's description (e.g., "art event", "business \
milestone", "life event of a relative"). A nearby event of the wrong type is wrong.
  - Among events matching the description, pick the one closest to the exact \
target date. Prefer events within ±2 days; only consider ±3-7 days if no closer match exists.
  - If a note says "I went to X last week" and the note is dated near the target, \
resolve "last week" to find the EXACT event date, not the note date.

STEP 4 — Compute the answer using ONLY the absolute dates:
  - For "how many days/weeks/months between X and Y": subtract the two absolute \
dates and convert to the requested unit.
  - For ordering questions: list each event with its absolute date, then sort by \
date (earliest first).
  - For "how many times" or counting: enumerate each matching event with its \
absolute date, then state the total count.
  - For "when" questions: state the absolute date directly.

CRITICAL rules:
- RECOLLECTION ≠ ACTION: When a note says "I was thinking about X", "I remembered X", \
or "I was reminiscing about X", the event X did NOT happen on that note's date. \
The note's date is when the user RECALLED the event, not when it occurred. \
Only use notes where the user describes PERFORMING an action to date that action.
- Notes are in chronological order. When the same fact appears in multiple \
notes with different values, always use the value from the MOST RECENT note.
- Give a direct, concise answer. Do not hedge if the evidence is clear.
- Show your date arithmetic briefly before giving the final answer.
- If you can infer the answer by combining information across multiple notes, DO SO. \
Do not refuse to answer simply because no single note contains the complete answer.
- When a relative time reference (e.g., "last Saturday", "two weeks ago") appears \
in a note, ALWAYS resolve it to an absolute date using that note's date stamp \
before comparing to the question date.
- BEFORE saying "not enough information": re-read every note looking for SYNONYMS \
or INDIRECT references. "Investment for a competition" could be "bought tools for \
a contest." "Kitchen appliance" could be "smoker" or "grill." "Piece of jewelry" \
could be "ring" or "necklace." Try harder to match before abstaining.

Notes from past conversations:

{sessions}

Current Date: {question_date}
Question: {question}
Answer:"""


# ── Category → prompt mapping (verbatim from OMEGA line 344-351) ──────
_CATEGORY_PROMPT: Final[dict[str, str]] = {
    "single-session-assistant": RAG_PROMPT_VANILLA,
    "single-session-user": RAG_PROMPT_VANILLA,
    "knowledge-update": RAG_PROMPT_ENHANCED,
    "multi-session": RAG_PROMPT_MULTISESSION,
    "temporal-reasoning": RAG_PROMPT_TEMPORAL,
    "single-session-preference": RAG_PROMPT_PREFERENCE,
}


def get_prompt_template(category: str) -> str:
    """Return the OMEGA prompt template for the given category.

    Falls back to VANILLA for unknown categories — same fallback OMEGA
    uses (line 1487 of their script).
    """
    return _CATEGORY_PROMPT.get(category, RAG_PROMPT_VANILLA)


def render_prompt(
    *,
    category: str,
    sessions: str,
    question: str,
    question_date: str,
) -> str:
    """Render the prompt for a given category with substituted fields."""
    template = get_prompt_template(category)
    return template.format(
        sessions=sessions,
        question=question,
        question_date=question_date,
    )


# ── Session formatting ────────────────────────────────────────────────


def format_session_text(turns: list[dict]) -> str:
    """Concatenate the turns of a session as ``role: content\\n...``.

    Verbatim from OMEGA line 461-466. This is the format we STORE in
    Neo4j — one Episode per session, content = the full role-prefixed
    transcript.
    """
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)


def format_session_for_prompt(content: str, date_str: str, index: int) -> str:
    """Format a retrieved session as a Chain-of-Note block.

    Verbatim from OMEGA line 469-475. The ``[Note N | Date: ...]``
    framing is what the prompt rules above reference (e.g. "Higher
    note numbers are more recent").
    """
    return (
        f"[Note {index} | Date: {date_str}]\n"
        f"{content}\n"
        f"[End Note {index}]"
    )


def parse_longmemeval_date(date_str: str) -> str:
    """Parse LongMemEval date 'YYYY/MM/DD (Day) HH:MM' → ISO format.

    Verbatim from OMEGA line 451-458. Strips the ``(Day)`` suffix
    before parsing — naive datetime, no timezone.
    """
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
    try:
        dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
        return dt.isoformat()
    except ValueError:
        return date_str


def answer_to_str(answer) -> str:
    """Convert mixed-type answer field to string. Verbatim OMEGA L478-482."""
    if isinstance(answer, list):
        return ", ".join(str(a) for a in answer)
    return str(answer)
