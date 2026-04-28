"""Stage 5 v2 Phase 8 — typed observation extraction for LongMemEval.

Per Atlas's spec (`docs/eval/stage5-v2-spec.md` Phase 8) and Codex's
2026-04-28 review: at adapter ingestion, call gpt-4o-mini per session
to extract a small list of TYPED, CONCISE observations (NOT free-form
evidence packets — that approach was Phase 3, which net-regressed and
got reverted).

Why this is different from Phase 3:
  * Phase 3 dumped raw user-turn snippets into the prompt — too noisy,
    distracted reasoning-heavy questions.
  * Phase 8 produces structured rows: ``{type, key, value, date, details}``.
    The model sees compact facts ("user's 5K PB: 26:30 on 2023-07-30")
    instead of paragraph-long quotes.

Schema:
    observations: list of records, each with:
      - type: one of {event, fact, preference, update}
      - key: short lowercase topic, e.g. "5k_pb", "yoga_class", "coffee_temp"
      - value: short verbatim quote OR a number/state — never a paraphrase
      - date: ISO date (YYYY-MM-DD) if anchored in the text, else null
      - details: optional ≤80-char extra context, else empty string

Cost: ~$0.0005 per session × ~3000 sessions per 500q dataset
≈ $1.50 for full extraction. Cached on Neo4j after first ingest.

Determinism: temperature=0, seed=42, max_tokens capped per session.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


_EXTRACTION_MODEL_DEFAULT = "gpt-4o-mini"
_MAX_TOKENS_DEFAULT = 800  # ~30 obs × ~25 tokens each is plenty
_RUN_SEED = 42

_VALID_TYPES = {"event", "fact", "preference", "update"}
_MAX_KEY_LEN = 60
_MAX_VALUE_LEN = 200
_MAX_DETAILS_LEN = 80
_MAX_OBSERVATIONS_PER_SESSION = 30


# OMEGA-style strict prompt — verbatim only, typed, concise.
EXTRACTION_PROMPT_TEMPLATE = """You are an extractor. Your job is to read a chat session between a user
and an assistant, and produce a small list of typed, verbatim observations.

OUTPUT a JSON object with one key "observations". Each observation is:
{{
  "type": "event" | "fact" | "preference" | "update",
  "key":  short lowercase topic (snake_case), max {max_key} chars
  "value": short verbatim quote OR a numeric/state value, max {max_value} chars
  "date": ISO date "YYYY-MM-DD" if anchored in the text, else null
  "details": optional brief context, max {max_details} chars (use "" if none)
}}

Type meanings:
  - event:      something that happened ("user attended yoga class #5")
  - fact:       a stable user attribute ("user_age = 32")
  - preference: a stated like/dislike ("prefers iced coffee")
  - update:     a value that REPLACED an earlier value ("5K PB: 27:45 → 26:30")

Strict rules:
  1. Extract VERBATIM. Do NOT infer, summarize, or paraphrase. If the
     user did not state it explicitly, do not include it.
  2. Quote short snippets in "value" — exact phrases, not invented prose.
  3. If the same observation appears multiple times, output it only once.
  4. Numbers, ordinals ("my 5th session"), and dates are gold — capture
     them exactly as stated.
  5. Skip assistant-only content unless it relays a fact the user just
     stated and the assistant confirmed.
  6. Skip greetings, pleasantries, hypotheticals, and questions.
  7. Cap at {max_obs} observations per session. If more would qualify,
     keep the most specific (numeric/dated/ordinal) ones.

Session date (treat as the "today" for this conversation): {session_date}

--- BEGIN SESSION ---
{session_text}
--- END SESSION ---

Output ONLY the JSON object. No prose, no markdown, no preamble.
"""


@dataclass(frozen=True)
class Observation:
    """One typed observation extracted from a session."""
    type: str
    key: str
    value: str
    date: Optional[str]
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "date": self.date,
            "details": self.details,
        }

    def render_line(self) -> str:
        """Render as a single line for prompt context.

        Format examples:
          event: yoga class #5 [2023-06-12] — vinyasa style
          fact:  user_age = 32 [2023-04-11]
          pref:  coffee_temperature = iced [2023-05-01]
          update: 5K_PB: 27:45 → 26:30 [2023-07-30]
        """
        date_part = f" [{self.date}]" if self.date else ""
        details_part = f" — {self.details}" if self.details else ""
        return f"- {self.type}: {self.key} = {self.value}{date_part}{details_part}"


def build_extraction_prompt(session_text: str, session_date: str) -> str:
    """Format the extraction prompt for one session."""
    return EXTRACTION_PROMPT_TEMPLATE.format(
        session_text=session_text,
        session_date=session_date or "unknown",
        max_key=_MAX_KEY_LEN,
        max_value=_MAX_VALUE_LEN,
        max_details=_MAX_DETAILS_LEN,
        max_obs=_MAX_OBSERVATIONS_PER_SESSION,
    )


def parse_extraction_response(raw: str) -> list[Observation]:
    """Parse and validate the JSON response from the extractor.

    Defensive: handles model returning extra prose around JSON, or
    individual rows missing fields. Drops invalid rows rather than
    raising, since one bad observation shouldn't kill a whole session.
    """
    if not raw or not raw.strip():
        return []

    # Strip code-fence wrappers the model sometimes adds.
    txt = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, re.DOTALL)
    if fence_match:
        txt = fence_match.group(1).strip()

    # Find the outermost JSON object.
    first_brace = txt.find("{")
    last_brace = txt.rfind("}")
    if first_brace < 0 or last_brace <= first_brace:
        logger.debug("extractor: no JSON object found in response")
        return []
    txt = txt[first_brace : last_brace + 1]

    try:
        payload = json.loads(txt)
    except json.JSONDecodeError as e:
        logger.debug("extractor: JSON parse failed: %s", e)
        return []

    rows = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    out: list[Observation] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        obs = _coerce_observation(row)
        if obs is not None:
            out.append(obs)
        if len(out) >= _MAX_OBSERVATIONS_PER_SESSION:
            break
    return out


def _coerce_observation(row: dict[str, Any]) -> Optional[Observation]:
    """Validate one row from the model. Returns None if not coercible."""
    obs_type = str(row.get("type", "")).strip().lower()
    if obs_type not in _VALID_TYPES:
        return None

    key = str(row.get("key", "")).strip()
    if not key:
        return None
    key = key[:_MAX_KEY_LEN]

    value = row.get("value")
    if value is None:
        return None
    value = str(value).strip()[:_MAX_VALUE_LEN]
    if not value:
        return None

    date_raw = row.get("date")
    date: Optional[str] = None
    if isinstance(date_raw, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_raw.strip()):
        date = date_raw.strip()

    details = str(row.get("details", "") or "").strip()[:_MAX_DETAILS_LEN]

    return Observation(type=obs_type, key=key, value=value, date=date, details=details)


def extract_observations(
    *,
    session_text: str,
    session_date: str,
    client: Any = None,
    model: Optional[str] = None,
) -> list[Observation]:
    """Extract typed observations from one session via gpt-4o-mini.

    Args:
        session_text: full role-prefixed session text (post format_session_text).
        session_date: ISO date string for the session (for prompt context).
        client: pre-built OpenAI client. If None, builds one from env.
        model: model id; defaults to gpt-4o-mini. Overridable for tests.

    Returns:
        List of Observation records. Empty list on failure (graceful
        degradation — Phase 8 is additive, not load-bearing).
    """
    if not session_text or not session_text.strip():
        return []

    model_id = model or os.getenv("JARVIS_LME_EXTRACT_MODEL", _EXTRACTION_MODEL_DEFAULT)

    if client is None:
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai SDK not installed; observation extraction disabled")
            return []
        # Same timeout discipline as the answerer (postmortem 2026-04-28):
        # CLOSE_WAIT on OpenAI's edge has wedged us before. 60s is plenty
        # for a single session-extraction call.
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60.0,
            max_retries=2,
        )

    prompt = build_extraction_prompt(session_text, session_date)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            max_tokens=_MAX_TOKENS_DEFAULT,
            temperature=0,
            seed=_RUN_SEED,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("extraction call failed (graceful skip): %s", e)
        return []

    return parse_extraction_response(raw)


def extract_observations_batch(
    *,
    sessions: list[tuple[str, str, str]],
    client: Any = None,
    model: Optional[str] = None,
    max_workers: int = 4,
) -> dict[str, list[Observation]]:
    """Extract observations for many sessions in parallel.

    Args:
        sessions: list of ``(session_id, session_text, session_date)`` tuples.
        client: optional pre-built OpenAI client. If None, builds one.
        model: model id; defaults to gpt-4o-mini.
        max_workers: thread-pool size. 4 is the sweet spot for OpenAI rate
                     limits + a single-session timeout of 60s — at 16 in-
                     flight requests per Mini host (4 adapter workers each
                     with this 4-way pool) we stay well under TPM caps.

    Returns:
        Map of ``session_id -> list[Observation]``. Sessions whose
        extraction failed (empty/timeout) get an empty list — extraction
        is additive, never load-bearing.

    Why thread-pool not async: matches the rest of the adapter's blocking
    style (Neo4j driver is sync, Chroma is sync). Threads avoid having
    to thread an event loop through every call site.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not sessions:
        return {}

    # Build one client per call (cheap) so callers without an OpenAI
    # client don't have to construct one.
    if client is None:
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai SDK not installed; batch extraction disabled")
            return {sid: [] for sid, _, _ in sessions}
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60.0,
            max_retries=2,
        )

    out: dict[str, list[Observation]] = {sid: [] for sid, _, _ in sessions}

    def _one(item: tuple[str, str, str]) -> tuple[str, list[Observation]]:
        sid, text, date = item
        return sid, extract_observations(
            session_text=text,
            session_date=date,
            client=client,
            model=model,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, item) for item in sessions]
        for fut in as_completed(futures):
            try:
                sid, obs = fut.result()
                out[sid] = obs
            except Exception as e:  # noqa: BLE001
                logger.warning("batch extraction worker failed: %s", e)

    return out
