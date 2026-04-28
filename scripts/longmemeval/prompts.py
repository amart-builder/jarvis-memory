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
- CUMULATIVE / IMPLIED COUNTS: if a later note describes an ADDITIONAL \
occurrence / instance / event of the same activity beyond what an earlier \
note stated, INCREMENT the count. Examples:
  · Note 1 says "I've done yoga 3 times this month"; Note 5 says "just \
finished another yoga class" → current total is 4.
  · Note 2 says "I have 5 chickens"; Note 7 says "got two more chickens" → \
current total is 7.
  Walk through the notes in order, treat each later mention of an additional \
event as +1, and report the cumulative total as of the most recent note.
- ORDINAL LANGUAGE OVERRIDES: phrases like "my Nth time", "my Nth visit", \
"the Nth one", "for the Nth time" state an exact running total. The most \
recent ordinal anywhere in the notes IS the answer — even if an earlier note \
gave a smaller explicit number. ("attended 3 sessions" earlier + "just went \
to my 5th session" later = 5.)
- IF AN EARLIER EXPLICIT COUNT AND A LATER IMPLICIT COUNT DISAGREE, the \
LATER one wins. Trust the most recent evidence, whether explicit ("now I've \
done it 6 times") or implicit ("did another one today" / "my 6th time").
- If the question references a role, title, or name that does NOT exactly match \
what appears in the notes, say the information is not enough to answer.
- "PREVIOUS" / "OLD" / "FORMER" / "ORIGINAL" QUESTIONS: when the question \
explicitly uses the words PREVIOUS, OLD, FORMER, or ORIGINAL to refer to a \
PRIOR value of something that has SINCE BEEN UPDATED, use the EARLIER value, \
NOT the latest. The "latest note wins" rule applies only when the question \
asks about CURRENT state. Example: question "what was my previous personal \
best 5K time?" with Note 2 saying 27:45 and Note 15 saying 25:30 (the new PB) \
→ answer is 27:45 (the previous PB before it was beaten). Note: this rule \
does NOT apply to "first" / "earliest" questions about events themselves \
(e.g., "what was my first job's salary?") — for those, pick the value tied \
to the FIRST event regardless of note order, using the literal question text.
- ABSTENTION ON QUESTION-SUBSTITUTION TRAPS: read the question literally. \
If the question asks about a SPECIFIC item ("the bus", "the red car", \
"my Tuesday class") and the notes contain a DIFFERENT but similar item \
("the train", "the blue car", "my Wednesday class"), DO NOT substitute. Say \
the information is not enough about the specific item asked about. \
"How much will I save by taking the bus?" — if notes don't quantify a \
bus-vs-taxi comparison, abstain — do NOT compute a train-vs-taxi answer.
- CROSS-ATTRIBUTE SUBSTITUTION TRAPS: when the question describes the topic \
WITH AN ATTRIBUTE (e.g., "my UNDERGRAD course research project poster", \
"my GRADUATE marathon", "my BLUE car", "my MORNING workout"), the attribute \
must match SEMANTICALLY in the notes — not just the noun. If the notes only \
mention the same noun with a SEMANTICALLY DIFFERENT attribute (e.g., notes \
say "thesis research poster" when question asks about "undergrad course \
research project poster"; notes say "second marathon" when question asks \
about "first marathon"; notes say "blue car" when question asks about "red \
car"), DO NOT substitute. Treat as not-enough-information. Common pitfalls: \
graduate-vs-undergraduate, first-vs-most-recent, this-year-vs-last-year. \
EXCEPTION — trivial rephrasings of the SAME attribute DO match: "morning \
workout" = "workout in the morning"; "Tuesday yoga" = "yoga on Tuesdays"; \
"my marathon last fall" = "my fall marathon". Plurality, word order, and \
common synonyms are fine — the rule targets cases where the attribute names \
a DIFFERENT instance of the noun, not a different way of saying the same \
instance.
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
- ABSTENTION ON QUESTION-SUBSTITUTION TRAPS: if the question asks about a \
SPECIFIC named item ("the bus", "Tuesday yoga", "Workshop A") and the notes \
discuss a DIFFERENT but similar item, do NOT substitute. Answer about the \
asked-about item or say the information is not enough — never silently swap.
- CROSS-ATTRIBUTE SUBSTITUTION TRAPS: if the question specifies an attribute \
("UNDERGRAD project", "FIRST marathon"), the attribute must match \
SEMANTICALLY (not just by noun). Notes about "thesis research" do NOT \
answer questions about "undergrad research" (different academic level). \
Notes about "second marathon" do NOT answer questions about "first \
marathon". Trivial rephrasings DO match — "Tuesday yoga" = "yoga on \
Tuesdays" — but a DIFFERENT instance does not. When the attribute genuinely \
mismatches, abstain rather than substitute.
- BOTH-SIDES RULE for compare/save/diff questions: questions of the form \
"how much will I save by X over Y", "what's the difference between X and Y", \
"how much more/less is X than Y" REQUIRE quantified values for BOTH X AND Y \
in the notes. If only one side is quantified, the answer is "not enough \
information." Do NOT pick a half-answer.

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


# ── MULTISESSION two-pass (Stage 4A) ──────────────────────────────────
# Failure analysis on Stage 1.5 still-wrongs: 0/17 multi-session failures
# were retrieval misses — gold sessions were all in top 20. The model
# under-counts because it prunes during enumeration ("this might be a
# duplicate, skip it" → final count too low). Splitting enumeration
# from counting fixes this: pass 1 lists every candidate without
# judgment; pass 2 dedupes + counts using the candidate list AND the
# original notes. Standard cognitive-offload pattern.
RAG_PROMPT_MULTISESSION_EXTRACT: Final[str] = """\
I will give you several notes from past conversations between you and a user, \
ordered from oldest to newest. The user is going to ask a question that \
involves counting, listing, or aggregating items across multiple notes.

YOUR JOB IN THIS PASS: extract a CANDIDATE LIST of every mention that could \
plausibly be relevant. Be MAXIMALLY INCLUSIVE — recall matters, precision \
does NOT.

Rules for this pass:
- INCLUDE items even if they look like duplicates of other items. \
A separate dedup pass handles that. If two notes mention what might be the \
same event, list them BOTH as separate candidates.
- INCLUDE items even if the match to the question is borderline or unclear. \
A separate judgment pass handles that.
- INCLUDE items even if the assistant in the conversation expressed doubt — \
the user's statement is what counts.
- Do NOT compute a total. Do NOT decide which items truly match. Do NOT \
dedupe. Just LIST.
- Format: a numbered list. For each item, quote the user's relevant words \
and cite the source as [Note #]. One item per line.
- If you find zero candidates, output the single line "No candidates found."

Notes from past conversations:

{sessions}

Current Date: {question_date}
Question: {question}

Candidate list (be liberal — include borderline items, do NOT count, do NOT \
dedupe):"""


RAG_PROMPT_MULTISESSION_COUNT: Final[str] = """\
I will give you (1) several notes from past conversations between you and a \
user, (2) a candidate list extracted from those notes by an earlier pass, \
and (3) the user's question.

YOUR JOB: produce the FINAL ANSWER. Decide which candidates truly match the \
question, merge near-duplicates, count, and answer.

Rules:
- INCLUSION RULE: for each candidate, KEEP it ONLY IF the user actually \
DID / HAD / IS the thing the question asks about. The user's STATEMENT IS \
GROUND TRUTH — but the statement must be the user CLAIMING to have done the \
thing, not merely talking around it. DROP candidates where the user only:
  (a) mentioned a related topic or domain WITHOUT claiming to have done the \
specific thing (e.g., question = "how many concerts did I attend"; user said \
"I love live music" → DROP; user said "I went to a concert last week" → KEEP);
  (b) PLANNED, INTENDED, or WANTED a future activity that may not have \
happened (e.g., "I'm thinking of trying X", "I might do X next month", "I \
should X more often" → DROP; "I tried X last weekend" → KEEP);
  (c) described someone else doing the thing (friend, family, assistant, \
hypothetical "people") → DROP. Only the USER themselves doing it counts \
unless the question explicitly asks about others;
  (d) used hypothetical / conditional / counterfactual language ("I'd love \
to", "if I had time", "imagine if I", "in theory I could") → DROP;
  (e) used PURE RECOLLECTION language with no concrete action AND the \
question is asking about RECENT or CURRENT activity (e.g., "I was thinking \
back to my marathon days" → DROP if the question is "how many marathons \
did I run THIS YEAR?"; KEEP if the question is "how many marathons have \
I run in my LIFE?" — for lifetime-total questions, recollections of \
genuinely-completed past events count).
  KEEP borderline candidates where the user clearly DID the thing but \
details are sparse. "I baked bread" with no recipe still counts as a bake. \
"I want to bake" does not. KEEP candidates where the user is recounting a \
specific, concrete past event (date, place, outcome stated) even if it's \
described retrospectively — those are real events.
- DEDUPLICATION: if two candidates describe the same underlying event/item \
(e.g., "cousin's wedding" and "Rachel's wedding at a vineyard" — likely the \
same wedding), MERGE into one. Err on the side of merging when borderline.
- USER STATEMENT BEATS ASSISTANT SKEPTICISM. If the user said they did the \
thing, count it. Even if the assistant in the conversation expressed doubt.
- Preserve quantities, units, and dates EXACTLY as the user stated them.
- For "how many"/count/total questions: \
  1. List each KEPT item briefly, citing [Note #]. \
  2. End with EXACTLY: "Total: N" where N is the integer count after \
deduplication. \
  3. Be decisive — never "approximately N" or "2 or 3".
- For "how much" sum questions: list each amount with [Note #], sum, state \
the total numerically.
- For "increase/decrease/change" questions: find BOTH starting and ending \
values, compute the difference. Do NOT report the final value as the change.
- BOTH-SIDES RULE for compare / save / diff questions ("how much will I \
save by X over Y", "how much more/less is X than Y"): you need quantified \
values for BOTH X and Y from the notes. If only one is given, abstain — \
do NOT silently substitute a similar item (e.g., do NOT use train-vs-taxi \
to answer a bus-vs-taxi question).
- ABSTENTION ON SUBSTITUTION: if the question asks about a SPECIFIC named \
item but the notes/candidates describe a similar-but-different item, do \
NOT substitute. Answer about the actual item or abstain.
- CROSS-ATTRIBUTE SUBSTITUTION (count version): if the question asks about \
items with a specific attribute (e.g., "fitness classes I attend EVERY \
WEEK", "UNDERGRAD courses I took", "MORNING runs"), only count items \
where the attribute matches. Drop candidates that mention the same noun \
with a different attribute or no attribute at all.
- If the candidate list is empty OR every candidate clearly contradicts \
the question, say the information is not enough — do NOT guess.
- NEVER fabricate values that aren't in the notes.

Notes from past conversations (for verification):

{sessions}

Candidate list (from extraction pass — items here may be duplicates or \
non-matches; you decide):

{candidate_list}

Current Date: {question_date}
Question: {question}

Final answer:"""


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

STEP 5 — VERIFY THE ARITHMETIC:
  Re-compute the answer one more time, from scratch, using the absolute dates \
you wrote in STEP 1. If the result differs from STEP 4, USE THE NEW VALUE — your \
first computation was wrong.
  Specifically: count days between dates by subtracting day numbers within the \
same month, OR by counting days through month boundaries (e.g., March 28 to April \
3 = 3 days left in March + 3 days in April = 6 days). Do NOT estimate.
  This step exists because gpt-class models reliably make off-by-one and \
month-boundary errors when computing dates. The verification pass catches them.

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
- "BEFORE / AFTER A KNOWN EVENT" QUESTIONS: when the question asks "how many \
X did I do BEFORE event Y?" or "AFTER event Y?":
  1. First find the absolute date of event Y from the notes.
  2. Then enumerate every X from ALL notes (not just notes near Y).
  3. Filter to those whose absolute date is BEFORE (or AFTER) Y's date.
  4. Count the filtered list. State the count.
- "Nth OCCURRENCE" QUESTIONS: when the question asks about "the Nth time I \
did X" (e.g., "my 10th jog"), enumerate ALL occurrences of X in chronological \
order, count to N, and report that occurrence's date. Then use that date for \
any subsequent calculation in the question.

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


def render_ms_extract_prompt(
    *, sessions: str, question: str, question_date: str,
) -> str:
    """Stage 4A pass 1: extract candidate list (high-recall, no count)."""
    return RAG_PROMPT_MULTISESSION_EXTRACT.format(
        sessions=sessions, question=question, question_date=question_date,
    )


def render_ms_count_prompt(
    *, sessions: str, candidate_list: str, question: str, question_date: str,
) -> str:
    """Stage 4A pass 2: dedupe + count from the candidate list (high-precision)."""
    return RAG_PROMPT_MULTISESSION_COUNT.format(
        sessions=sessions,
        candidate_list=candidate_list,
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
